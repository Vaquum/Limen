"""Run one implementation in a fresh process over shared recorded inputs."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time


def _peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def _module_sources(root):
    result = {}
    for name, module in sorted(sys.modules.items()):
        if name == 'limen' or name.startswith('limen.'):
            if module.__file__ is None:
                paths = [Path(path).resolve() for path in module.__path__]
                if not paths or any(not path.is_relative_to(root) for path in paths):
                    raise ValueError(f'{name} namespace is outside the requested source tree: {paths}')
                result[name] = {'namespace_paths': [str(path) for path in paths]}
                continue
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'{name} was imported outside the requested source tree: {path}')
            result[name] = {'path': str(path),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def _measure(calculate, repeats, iterations):
    started = time.perf_counter_ns()
    result = calculate()
    cold = (time.perf_counter_ns() - started) / 1e6
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            result = calculate()
        samples.append((time.perf_counter_ns() - started) / (1e6 * iterations))
    return result, {'first_call_ms': cold, 'samples_ms': samples,
                    'iterations_per_sample': iterations}


def _trade_case(frame, signal, timer):
    import polars as pl

    from limen.backtest.trade_contract import ProductSpec, TradeInputs, TradePolicy, source_binding
    from limen.experiment._prepare_trade_context import normalize_observations

    interval = int(frame['datetime'].dt.epoch('ns').diff().drop_nulls()[0])
    observations = normalize_observations(frame, interval_seconds=interval / 1e9)
    start, end = int(observations['start_ns'][0]), int(observations['available_at_ns'][-1])
    signals = pl.DataFrame({'row_id': [f'signal:{index}' for index in range(frame.height)],
                            'available_at_ns': observations['available_at_ns'],
                            'target': signal.astype(float), 'datetime': frame['datetime']})
    binding = source_binding(observations, 'shared recorded benchmark input', start, end,
                              interval, 'recorded OHLC intervals')
    inputs = TradeInputs(10000.0, start, end, signals, observations, None, (binding,))
    policy = TradePolicy(ProductSpec('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-12, 0.0),
                         take_profit_bps=10.0, stop_loss_bps=10.0,
                         max_holding_seconds=3600.0 if timer else None,
                         timer_interval_seconds=1.0 if timer else None)
    return inputs, policy


def _capture_trade(ledger, output):
    import numpy as np

    for name in ('states', 'intents', 'fills', 'episodes', 'funding'):
        getattr(ledger, name).write_parquet(output / f'{name}.parquet')
    names = tuple(ledger.metrics)
    np.savez(output / 'arrays.npz', metric_names=np.asarray(names),
             metrics=np.asarray([ledger.metrics[name] for name in names]),
             contract_digest=np.asarray(ledger.contract_digest))


def _snapshot_case(args, frame, signal):
    import numpy as np

    from limen.backtest._snapshot_execution import snapshot_with_execution
    from limen.backtest._snapshot_ledger import snapshot_ledger

    columns = {name: frame[name].to_numpy() for name in ('open', 'high', 'low', 'close')}
    columns.update(predictions=signal, price_change=columns['close'] - columns['open'])
    options = {'fee_bps': 10.0, 'slip_bps': 5.0, 'notional_rate': 0.5}
    if args.scenario == 'snapshot_ledger':
        _, execution = snapshot_with_execution(columns, **options)

        def calculate():
            return snapshot_ledger(execution, options['notional_rate'])
    else:
        def calculate():
            return snapshot_with_execution(columns, **options)
    result, timing = _measure(calculate, args.repeats, 20)
    if args.scenario == 'snapshot_full':
        result, execution = result
    names = tuple(result)
    np.savez(args.output / 'arrays.npz', metric_names=np.asarray(names),
             metrics=np.asarray([result[name] for name in names]),
             pos=execution.pos, gross=execution.gross, net=execution.net)
    return timing


def _feature_case(args, frame):
    import numpy as np
    import polars as pl

    from limen.features.price_lines import price_lines
    from limen.features.quantile_price_lines import quantile_price_lines

    if args.variant == 'optimized':
        from limen.utils._computation_cache import computation_reuse
        scope = computation_reuse
    else:
        scope = nullcontext

    def calculate():
        outputs = []
        with scope():
            for index in range(args.rounds):
                duration = (12, 24)[index % 2]
                height = (0.001, 0.002)[(index // 2) % 2]
                threshold = (0.5, 0.75)[(index // 4) % 2]
                lines = price_lines(frame, duration, height, include_research_only=False)
                quantiles = quantile_price_lines(frame, duration, height, threshold,
                                                 include_research_only=False)
                names = [name for name in lines.columns if name not in frame.columns]
                quantile_names = [name for name in quantiles.columns if name not in frame.columns]
                outputs.append(lines.select(names).hstack(quantiles.select(quantile_names)))
        return pl.concat(outputs, how='vertical')

    result, timing = _measure(calculate, args.repeats, 1)
    result.write_parquet(args.output / 'features.parquet')
    np.savez(args.output / 'arrays.npz', rows=np.asarray(result.height),
             columns=np.asarray(result.columns))
    timing['rounds'] = args.rounds
    timing['scope'] = 'two line features per round; four detector keys and eight quantile-feature keys'
    return timing


def _event_case(args, frame, signal):
    from limen.backtest.trade_execution import trade_execution

    inputs, policy = _trade_case(frame.head(12), signal[:12], timer=True)
    result, timing = _measure(lambda: trade_execution(inputs, policy), args.repeats, 1)
    _capture_trade(result, args.output)
    timing['state_rows'] = result.states.height
    timing['scope'] = 'full event execution, all audit state rows and metrics; one-second timers'
    return timing


def _funded_case(args, frame, signal):
    import polars as pl

    from limen.backtest.funding_adapter import ADAPTER_VERSION, prepare_funding
    from limen.backtest.trade_contract import FundingPolicy, source_binding
    from limen.backtest.trade_execution import trade_execution

    recorded = json.loads(args.funding.read_text())
    records, schedule = recorded['records'], recorded['declared_schedule']
    milliseconds = 1000000
    slot_ms = schedule['settlement_interval_seconds'] * 1000
    slots = [row['fundingTime'] // slot_ms * slot_ms for row in records]
    history = pl.DataFrame({
        'event_id': [f'BTCUSDT:{row["fundingTime"]}' for row in records],
        'settlement_at': [row['fundingTime'] * milliseconds for row in records],
        'settlement_slot_ns': [slot * milliseconds for slot in slots],
        'rate_decimal': [float(row['fundingRate']) for row in records],
        'valuation_price': [float(row['markPrice']) for row in records],
        'schedule_start': [min(slots) * milliseconds] * len(records),
        'schedule_end': [(max(slots) + slot_ms) * milliseconds] * len(records),
        'settlement_interval_seconds': [schedule['settlement_interval_seconds']] * len(records),
        'settlement_phase_utc_seconds': [schedule['settlement_phase_utc_seconds']] * len(records),
    })
    params = {'mechanism': 'discrete', 'rate_unit': 'decimal',
              'rate_basis_seconds': schedule['settlement_interval_seconds'],
              **schedule, 'valuation': 'mark', 'currency': 'USDT',
              'approximation': 'recorded_exact'}
    funding = FundingPolicy('limen.backtest.funding_adapter.funding_adapter', ADAPTER_VERSION,
                            params, None, None, 'discrete',
                            schedule['settlement_interval_seconds'], 'mark', 'USDT',
                            'recorded_exact')
    frame = frame.head(168)
    inputs, policy = _trade_case(frame, signal[:frame.height], timer=False)
    events = prepare_funding(funding, history, inputs.partition_start_ns, inputs.partition_end_ns)
    binding = source_binding(events, 'shared recorded Binance funding',
                              inputs.partition_start_ns, inputs.partition_end_ns,
                              milliseconds, 'recorded settlement rates and mark prices')
    inputs = replace(inputs, funding_events=events, sources=(*inputs.sources, binding))
    policy = replace(policy, funding=funding, take_profit_bps=None, stop_loss_bps=None)
    result, timing = _measure(lambda: trade_execution(inputs, policy), args.repeats, 1)
    _capture_trade(result, args.output)
    timing['state_rows'] = result.states.height
    timing['funding_rows'] = result.funding.height
    held_funding = result.funding.filter(pl.col('quantity') != 0).height
    if not held_funding:
        raise AssertionError('Recorded funding benchmark did not fund an active position')
    timing['held_funding_rows'] = held_funding
    timing['scope'] = 'full event execution with recorded discrete funding; 168 hourly bars, no timer'
    return timing


def _target_case(args, frame, signal):
    import numpy as np

    from limen.backtest.trade_contract import contract_digest, export_trade_contract
    from limen.targets.trade_outcome import TradeOutcomeTarget, TradeTargetContext

    frame = frame.head(args.target_rows)
    inputs, policy = _trade_case(frame, signal[:frame.height], timer=False)
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns,
                                 contract_digest(export_trade_contract(policy, inputs)))

    def calculate():
        target = TradeOutcomeTarget(frame, 'outcome', trade_context=context, output='net_return')
        result = target.transform(frame, trade_context=context)
        return result['outcome'], target.outcomes.rows

    result, timing = _measure(calculate, args.repeats, 1)
    outcome, rows = result
    rows.write_parquet(args.output / 'labels.parquet')
    np.savez(args.output / 'arrays.npz', outcome=outcome.to_numpy())
    timing['candidates'] = frame.height * 2
    timing['scope'] = 'both forward trade outcomes per recorded row; TP/SL-only exits'
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--signal', type=Path, required=True)
    parser.add_argument('--funding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('baseline', 'optimized'), required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--repeats', type=int, required=True)
    parser.add_argument('--rounds', type=int, required=True)
    parser.add_argument('--target-rows', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))

    import numpy as np
    import polars as pl

    import limen

    if not Path(limen.__file__).resolve().is_relative_to(args.root):
        raise ValueError('Imported Limen is outside the requested implementation tree')

    args.output.mkdir(parents=True)
    frame, signal = pl.read_parquet(args.input), np.load(args.signal, allow_pickle=False)
    if args.scenario.startswith('snapshot_'):
        timing = _snapshot_case(args, frame, signal)
    elif args.scenario == 'line_features':
        timing = _feature_case(args, frame)
    elif args.scenario == 'timer_execution':
        timing = _event_case(args, frame, signal)
    elif args.scenario == 'funded_execution':
        timing = _funded_case(args, frame, signal)
    elif args.scenario == 'forward_labels':
        timing = _target_case(args, frame, signal)
    else:
        raise ValueError(f'Unknown scenario {args.scenario}')
    timing.update(median_ms=float(np.median(timing['samples_ms'])),
                  p95_ms=float(np.percentile(timing['samples_ms'], 95)))
    report = {'scenario': args.scenario, 'variant': args.variant, 'source_root': str(args.root),
              'python': platform.python_version(), 'numpy': np.__version__,
              'polars': pl.__version__, 'limen': limen.__version__,
              'limen_source': limen.__file__, 'loaded_module_sources': _module_sources(args.root),
              'peak_process_rss_bytes': _peak_rss_bytes(), 'timing': timing}
    report['peak_process_rss_mib'] = report['peak_process_rss_bytes'] / 1024 ** 2
    (args.output / 'timing.json').write_text(json.dumps(report, indent=2) + '\n')
    sys.stdout.write(json.dumps({name: report[name] for name in
                                ('scenario', 'variant', 'source_root',
                                 'peak_process_rss_mib', 'timing')}) + '\n')


if __name__ == '__main__':
    main()
