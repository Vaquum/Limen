from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from limen.backtest.trade_contract import ProductSpec, TradeInputs, TradePolicy, contract_digest, export_trade_contract, source_binding
from limen.experiment._prepare_trade_context import normalize_observations
from limen.features.volatility_barriers import volatility_barriers

FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'


def test_formula_warmup_and_split_boundary_invariance():
    raw = pl.read_parquet(FIXTURE, n_rows=96)
    configured = {'span': 6, 'regime_span': 24, 'min_periods': 3, 'minimum_bps': 1, 'maximum_bps': 1000}
    actual = volatility_barriers(raw, **configured)
    prior = pd.Series(raw['close'].to_list()).pct_change(fill_method=None).shift(1)
    fast = prior.ewm(span=6, adjust=True, min_periods=3).std(bias=False)
    slow = prior.ewm(span=24, adjust=True, min_periods=3).std(bias=False)
    multiplier = np.where(fast / slow < 0.75, 0.8, np.where(fast / slow > 1.25, 1.2, 1.0))
    expected = np.clip(10000 * fast * multiplier * 2, 1, 1000)
    np.testing.assert_allclose(actual['volatility_take_profit_bps'].to_numpy(), expected, equal_nan=True, atol=1e-10)
    assert actual['__barrier_ready__'].to_list() == [False] * 4 + [True] * 92
    for end in (24, 48, 72):
        assert_frame_equal(volatility_barriers(raw.head(end), **configured), actual.head(end))


def test_prefix_causality_and_unconfigured_compatibility():
    raw = pl.read_parquet(FIXTURE, n_rows=16)
    prices = normalize_observations(raw, interval_seconds=3600)
    start, end = int(prices['start_ns'][0]), int(prices['available_at_ns'][-1])
    signals = prices.select('row_id', 'available_at_ns').with_columns(pl.lit(None, dtype=pl.Float64).alias('target'))
    source = source_binding(prices, 'legacy:recorded-hourly-prefix', start, end, 1000000000, 'recorded_hourly_ohlc')
    inputs = TradeInputs(10000, start, end, signals, prices, None, (source,))
    policy = TradePolicy(ProductSpec('cash_spot', 'BTCUSDT', 'BTC', 'USDT', 0.00001, 5), fee_bps=10, slip_bps=5)
    assert contract_digest(export_trade_contract(policy, inputs)) == '9adc51c47e50e023f1d2d364400ba78e615bdf14304ab999ca66adc3b27ca9b7'
    assert 'barrier_spec' not in export_trade_contract(policy, inputs)['policy']
    assert replace(policy, take_profit_bps_column='entry_tp').take_profit_bps_column == 'entry_tp'


def recorded_source(**kwargs):
    return pl.read_parquet(FIXTURE, n_rows=120)


def _yaml():
    from limen import __version__

    return {
        'schema_version': '1.0', 'metadata': {'name': 'dynamic_barrier_proof', 'mode': 'development', 'limen_version': __version__},
        'sfd': {'manifest': {
            'type': 'rule_based', 'data_source': {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(FIXTURE.resolve())}},
            'split_dates': {'train_start': '2024-01-02', 'train_end': '2024-01-03', 'val_start': '2024-01-04', 'val_end': '2024-01-05', 'test_start': '2024-01-05', 'test_end': '2024-01-06'},
            'strategy': {'conditions': [{'id': 'entry', 'name': 'Recorded close above open', 'type': 'relative', 'column': 'close', 'operator': '>', 'other_column': 'open'}], 'entry': 'entry'},
            'reference_architecture': 'limen.sfd.reference_architecture.rule_based.rule_based',
            'backtest': {'product': {'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC', 'quote_currency': 'USDT', 'quantity_step': 0.00001, 'min_notional': 5},
                         'source_interval_seconds': 3600, 'fee_bps': 10, 'slip_bps': 5,
                         'take_profit_bps_column': 'volatility_take_profit_bps', 'stop_loss_bps_column': 'volatility_stop_loss_bps',
                         'barrier_features': {'method': 'limen.features.volatility_barriers.volatility_barriers', 'params': {'span': '{risk_span}', 'regime_span': 24, 'min_periods': 3}}},
        }, 'params': {'risk_span': [6, 12]}}, 'uel': {'n_permutations': 2, 'search_strategy': {'type': 'random', 'seed': 42}, 'prep_each_round': True},
    }


def _native():
    from limen.yaml import CompiledSFD

    manifest = CompiledSFD(_yaml()).manifest()
    raw = recorded_source()
    return manifest, raw, manifest.prepare_data(raw, {'risk_span': 6})


def test_entry_decision_distances_survive_lag_and_resize():
    import pytest
    from limen.backtest._event_plan import event_plan
    from limen.backtest.trade_execution import trade_execution
    from limen.experiment.errors import StrictModeError
    from tests.test_target_exposure import _case

    inputs, policy = _case([0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 0.0, 0.0, -0.5, -0.5, 0.0, 0.0])
    signals = inputs.signals.with_columns(pl.Series('entry_tp', [4000.0 + i for i in range(12)]), pl.Series('entry_sl', [3000.0 + i for i in range(12)]), pl.lit(True).alias('__barrier_ready__'))
    inputs = replace(inputs, signals=signals)
    policy = replace(policy, take_profit_bps_column='entry_tp', stop_loss_bps_column='entry_sl', execution_lag_seconds=900)
    ledger = trade_execution(inputs, policy)
    assert ledger.intents['status'][0] == 'replaced'
    assert ledger.episodes['take_profit_bps'].to_list() == [4001.0, 4008.0]
    assert ledger.episodes['stop_loss_bps'].to_list() == [3001.0, 3008.0]
    assert ledger.fills.filter(pl.col('episode_id') == ledger.episodes['episode_id'][0]).height == 3
    assert ledger.episodes['anchor_price'][0] == ledger.fills['fill_price'][0]
    first_plan = event_plan(inputs, policy)
    changed = replace(inputs, signals=signals.with_columns(pl.col('entry_tp') + 10))
    second_plan = event_plan(changed, replace(policy, barrier_spec={'params': {'scale': 2}}))
    assert first_plan.events is second_plan.events
    assert first_plan.binding != second_plan.binding
    warmup = replace(inputs, signals=signals.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col('entry_tp')).alias('entry_tp'), pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col('entry_sl')).alias('entry_sl'), (pl.int_range(pl.len()) != 0).alias('__barrier_ready__')))
    with pytest.raises(StrictModeError, match='Barrier warmup at entry'):
        trade_execution(warmup, policy)
    for value in (None, float('inf'), -1.0):
        broken = signals.with_columns(pl.when(pl.int_range(pl.len()) == 4).then(value).otherwise(pl.col('entry_tp')).alias('entry_tp'))
        with pytest.raises(ValueError, match=r'barrier|entry_tp|finite'):
            trade_execution(replace(inputs, signals=broken), policy)


def test_native_yaml_and_outcome_contract_parity(tmp_path):
    from click.testing import CliRunner
    from ruamel.yaml import YAML
    from limen.cli.main import cli
    from limen.yaml import validate
    from limen.targets.trade_outcome import TradeOutcomeTarget, TradeTargetContext
    from limen.backtest.trade_execution import trade_execution
    from tests.test_target_exposure import _case

    manifest, raw, data = _native()
    assert validate(_yaml()).valid
    from limen.experiment._fixed_trade_sources import fixed_trade_sources
    from limen.experiment.manifest_core import DataSourceConfig
    from copy import deepcopy
    fixed = deepcopy(manifest)
    fixed.backtest_config.execution_data_source = DataSourceConfig(recorded_source, {'kline_size': 3600})
    with fixed_trade_sources(fixed, ['risk_span']):
        a = fixed.prepare_data(raw, {'risk_span': 6})
        b = fixed.prepare_data(raw, {'risk_span': 12})
    assert a['_trade_policy'].barrier_spec != b['_trade_policy'].barrier_spec
    expected = volatility_barriers(raw, span=6, regime_span=24, min_periods=3)
    for mapping, partition in zip(data['_trade_context'].model_rows, data['_trade_context'].partitions, strict=True):
        distances = mapping.select('datetime').join(expected, on='datetime', maintain_order='left')
        assert partition.signals['volatility_stop_loss_bps'].equals(distances['volatility_stop_loss_bps'])
    assert not any('volatility_' in name or name == '__barrier_ready__' for name in data['train'].columns)
    from limen.backtest._event_plan import event_plan
    other = manifest.prepare_data(raw, {'risk_span': 12})
    left = event_plan(data['_trade_inputs'], data['_trade_policy'])
    right = event_plan(other['_trade_inputs'], other['_trade_policy'])
    assert left.events is right.events and left.binding != right.binding
    result = manifest.run_model(data, {'risk_span': 6})
    assert result['backtest_completed_trades'] >= 1
    path = tmp_path / 'barriers.yaml'
    with path.open('w') as stream:
        YAML().dump(_yaml(), stream)
    assert CliRunner().invoke(cli, ['validate', str(path)]).exit_code == 0
    assert CliRunner().invoke(cli, ['run', str(path), '--dry-run']).exit_code == 0
    inputs, policy = _case([1.0] * 8, max_holding_seconds=900)
    times = pl.read_parquet(Path(__file__).parent / 'fixtures/dollar_bar_crash_reversal_15m.parquet', n_rows=8)['datetime']
    inputs = replace(inputs, signals=inputs.signals.with_columns(times, pl.Series('tp', [20.0 + i for i in range(8)]), pl.Series('sl', [10.0 + i for i in range(8)]), pl.lit(True).alias('__barrier_ready__')))
    policy = replace(policy, take_profit_bps_column='tp', stop_loss_bps_column='sl')
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(policy, inputs)))
    target = TradeOutcomeTarget(pl.DataFrame({'datetime': times}), 'outcome', trade_context=context)
    target.transform(pl.DataFrame({'datetime': times}), trade_context=context)
    scalar = replace(policy, take_profit_bps_column=None, stop_loss_bps_column=None, take_profit_bps=20, stop_loss_bps=10)
    ledger = trade_execution(inputs, scalar)
    assert target.outcomes.rows['long_return'][0] == ledger.episodes['net_pnl'][0] / ledger.episodes['entry_equity'][0]
    warm = inputs.signals.with_columns((pl.int_range(pl.len()) >= 2).alias('__barrier_ready__'), *[pl.when(pl.int_range(pl.len()) < 2).then(None).otherwise(pl.col(name)).alias(name) for name in ('tp', 'sl')])
    warm_inputs = replace(inputs, signals=warm)
    warm_context = replace(context, inputs=warm_inputs, contract_digest=contract_digest(export_trade_contract(policy, warm_inputs)))
    target.transform(pl.DataFrame({'datetime': times}), trade_context=warm_context)
    assert target.outcomes.rows['long_available'].head(2).to_list() == [False, False]
    assert target.outcomes.rows['long_unavailable_reason'].head(2).to_list() == ['barrier_warmup', 'barrier_warmup']
    assert target.outcomes.rows['long_available'].slice(2).any()


def test_recording_and_replay_require_original_distances():
    from types import SimpleNamespace
    import pytest
    from limen.experiment._recording import project_record
    from limen.log._snapshot_backtest_round import snapshot_backtest_round

    manifest, raw, data = _native()
    result = manifest.run_model(data, {'risk_span': 6})
    log = SimpleNamespace(manifest=manifest, data=raw, round_params=[{'risk_span': 6}], preds=[result['_preds']], _alignment=[data['_alignment']])
    metrics = snapshot_backtest_round(log, 0, lambda x: x)
    assert metrics['completed_trades'] == result['backtest_completed_trades']
    contract = data['trade_contract']
    record = {'trade_contract': contract, 'trade_ledger': data['_alignment']['trade_ledger']}
    omitted = project_record(record, dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), False))
    assert 'signal_rows' not in omitted['trade_contract']
    assert 'trade_ledger' not in omitted
    assert 'barrier_distances' in omitted['trade_contract']
    for alignment in ({}, {'trade_contract': omitted['trade_contract']}):
        log._alignment = [alignment]
        with pytest.raises(ValueError, match='original bound'):
            snapshot_backtest_round(log, 0, lambda x: x)


def test_research_promotion_and_factorize_reject(tmp_path):
    import json
    from types import SimpleNamespace
    from unittest.mock import patch
    import pytest
    from limen.inference import Trainer, Sensor
    from limen.experiment._parallel_rounds import begin_run
    from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy
    from limen.experiment._factorized_admission import validate_domain

    config = _yaml()
    (tmp_path / 'metadata.json').write_text(json.dumps({'yaml_reference': config}))
    before = (tmp_path / 'metadata.json').read_bytes()
    with patch.object(Trainer, '_load_round_data', side_effect=AssertionError('late rejection')), pytest.raises(ValueError, match='research-only dynamic barriers/allocation cannot be promoted'):
        Trainer(tmp_path)
    assert (tmp_path / 'metadata.json').read_bytes() == before
    with pytest.raises(ValueError, match='research-only dynamic barriers/allocation cannot be promoted'):
        Sensor(config, RuleBasedStrategy(), {}, {'risk_span': 6})
    manifest, _, _ = _native()
    for workers, resume in ((2, False), (1, True)):
        host = SimpleNamespace(manifest=manifest, _experiment_dir=tmp_path)
        with pytest.raises(ValueError, match='fresh serial'):
            begin_run(host, workers, True, resume, False, False, None, (None, None, None))
    with pytest.raises(ValueError, match=r'event|snapshot'):
        _ = validate_domain(manifest, {'risk_span': [6]}, frozenset())


def test_generic_interval_metadata_survives_promotion():
    from limen.inference import Sensor
    from limen.sfd.reference_architecture.random_binary import RandomBinary
    from limen.yaml import CompiledSFD

    config = _yaml()
    declared = config['sfd']['manifest']
    declared['type'] = 'ml'
    declared.pop('strategy')
    declared['indicators'] = [{'func': 'limen.indicators.window_return', 'params': {'period': 1}}]
    declared['target'] = {'name': 'up', 'class': 'limen.targets.QuantileBinaryTarget', 'fit_params': {'source_column': 'ret_1', 'quantile': 0.5}, 'transform_params': {'shift': -1}}
    declared['reference_architecture'] = 'limen.sfd.reference_architecture.random_binary.random_binary'
    for key in ('take_profit_bps_column', 'stop_loss_bps_column', 'barrier_features'):
        declared['backtest'].pop(key)
    manifest = CompiledSFD(config).manifest()
    raw = recorded_source()
    data = manifest.prepare_data(raw, {})
    model = RandomBinary().train(data)
    sensor = Sensor(config, model, data['_fitted_params'], {})
    future = pl.read_parquet(FIXTURE, n_rows=168)
    predicted = sensor.predict_all(future)
    assert predicted[-1].reason is None
    assert predicted[-1].available_at_ns == int(future['datetime'].dt.epoch('ns')[-1]) + 3600_000_000_000
