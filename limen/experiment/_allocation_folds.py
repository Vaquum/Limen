from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from typing import cast
from importlib import import_module

import numpy as np
import polars as pl

from limen.backtest.trade_contract import JsonValue, TradeInputs, contract_digest, json_value, source_binding
from limen.backtest.trade_execution import trade_execution
from limen.experiment._allocation import AllocationBatch, AllocationContext, OpportunityLabels, component_identity
from limen.experiment._prepare_trade_context import normalize_observations
from limen.experiment.errors import StrictModeError
from limen.experiment.manifest_core import Manifest, MLManifest, RuleBasedManifest
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.sfd.reference_architecture.direction_sizing import Array
from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy


_core = vars(import_module('limen.experiment.manifest_core'))
apply_feature_transforms = cast(Callable[..., pl.LazyFrame], _core['_apply_feature_transforms'])
finalize_rule_based_data = cast(Callable[..., dict[str, object]], _core['_finalize_rule_based_data'])
process_bars = cast(Callable[..., tuple[object, pl.DataFrame]], _core['_process_bars'])
resolve_params = cast(Callable[..., dict[str, object]], _core['_resolve_params'])
resolve_split = cast(Callable[..., list[pl.DataFrame]], _core['_resolve_split'])


def fit_primary(manifest: object, data: Mapping[str, object], params: Mapping[str, object]) -> ReferenceModel:
    manifest = cast(Manifest, manifest)
    kwargs = manifest.resolve_model_kwargs(dict(params))
    if isinstance(manifest, RuleBasedManifest):
        return RuleBasedStrategy(**kwargs).train(dict(data))
    return LightGBMBinary().train(dict(data), **kwargs)


def _rule_features(manifest: RuleBasedManifest, raw: pl.DataFrame, params: Mapping[str, object]) -> pl.DataFrame:
    _, bars = process_bars(manifest, raw, dict(params))
    frame = apply_feature_transforms(manifest, bars.lazy(), dict(params)).collect().fill_nan(None).drop_nulls()
    data = finalize_rule_based_data(manifest, [frame, frame.clear(), frame.clear()], frame['datetime'].to_list(), dict(params))
    return cast(pl.DataFrame, data['train']).with_columns(frame['datetime'])


def _inference(manifest: MLManifest, history: pl.DataFrame, fitted: Mapping[str, object], params: Mapping[str, object]) -> pl.DataFrame:
    features, _ = manifest.sensor_input_prep(history, dict(fitted), dict(params))
    return features.drop([name for name in ('__trade_available_at_ns__', manifest.target_column) if name is not None and name in features.columns]).fill_nan(None).drop_nulls()


def training_opportunities(context: AllocationContext, *, folds: int, validation_fraction: float, include_primary_proba: bool) -> OpportunityLabels:
    raw, manifest, params = context.raw_train, cast(Manifest, context.manifest), context.round_params
    if folds > raw.height:
        raise StrictModeError('Allocation fold count exceeds original raw training rows')
    if include_primary_proba and not isinstance(manifest, MLManifest):
        raise ValueError('Rule primary supplies no learned probability')
    prices = normalize_observations(raw, interval_seconds=manifest.backtest_config.source_interval_seconds if manifest.backtest_config else None)
    prices = prices.with_columns(pl.Series('row_id', [f'source:{value}' for value in prices['start_ns']]))
    rule_features = _rule_features(manifest, raw, params) if isinstance(manifest, RuleBasedManifest) else None
    labels: list[dict[str, object]] = []
    features: list[pl.DataFrame] = []
    bindings: list[dict[str, JsonValue]] = []
    armed = False
    prior = 1
    for block_id, indices in enumerate(np.array_split(np.arange(raw.height), folds)):
        if block_id == 0:
            continue
        start, end = int(indices[0]), int(indices[-1]) + 1
        block_start = int(prices['start_ns'][start])
        block_end = int(prices['available_at_ns'][end - 1])
        if isinstance(manifest, MLManifest):
            prefix = raw.head(start - 1)
            clone = deepcopy(manifest)
            clone.allocation_config = None
            clone.backtest_config = None
            clone.split_dates = None
            clone.split_config = (max(1, int(prefix.height * (1 - validation_fraction))), max(1, int(prefix.height * validation_fraction)), 0)
            prepared = clone.prepare_data(prefix, dict(params))
            if not len(cast(pl.DataFrame, prepared['x_train'])) or not len(cast(pl.DataFrame, prepared['x_val'])):
                raise StrictModeError(f'OOF block {block_id}: insufficient prefix primary fit rows')
            primary = fit_primary(clone, prepared, params)
            frame = _inference(clone, raw.head(end), cast(Mapping[str, object], prepared['_fitted_params']), params)
            frame = frame.filter(pl.col('datetime').is_between(raw['datetime'][start], raw['datetime'][end - 1]))
            names = cast(pl.DataFrame, prepared['x_train']).columns
            prediction = primary.predict({'x_test': frame.select(names)})
            meta = frame.select(names)
            target = clone.target_class_config
            target_binding = None if target is None else {'method': f'{target.target_class.__module__}.{target.target_class.__qualname__}', 'fit': json_value(target.fit_params), 'transform': json_value(target.transform_params)}
        else:
            if rule_features is None or not isinstance(manifest, RuleBasedManifest) or manifest.strategy is None:
                raise ValueError('Missing original causal rule preparation')
            prefix_features = _rule_features(manifest, raw.head(end), params)
            earlier = rule_features.filter(pl.col('datetime') < raw['datetime'][end - 1])
            if not prefix_features.filter(pl.col('datetime') < raw['datetime'][end - 1]).equals(earlier):
                raise ValueError('Primary rule features depend on observations after their decision')
            frame = prefix_features.filter(pl.col('datetime').is_between(raw['datetime'][start], raw['datetime'][end - 1]))
            prediction = RuleBasedStrategy().predict({'test': frame, 'strategy': {'conditions': manifest.strategy.conditions, 'entry': manifest.strategy.entry}})
            excluded = {condition['id'] for condition in manifest.strategy.conditions}
            meta = frame.select([name for name, dtype in frame.schema.items() if dtype.is_numeric() and name not in excluded])
            target_binding = None
        if not frame.height:
            raise StrictModeError(f'OOF block {block_id}: no feature-valid primary predictions')
        decisions = np.asarray(prediction['_preds'], dtype=np.float64)
        if decisions.shape != (frame.height,) or not np.isin(decisions, [0, 1]).all():
            raise ValueError('OOF primary must emit aligned binary decisions')
        if include_primary_proba:
            probabilities = np.asarray(prediction['_probs'], dtype=np.float64)
            if probabilities.shape != decisions.shape or not np.isfinite(probabilities).all():
                raise ValueError('OOF primary probabilities must be finite and aligned')
            meta = meta.with_columns(pl.Series('primary_probability', probabilities))
        membership = frame['datetime'].dt.epoch('ns').to_list()
        bindings.append({'block': block_id, 'start_ns': block_start, 'end_ns': block_end,
                         'fit_prefix_rows': start - 1 if isinstance(manifest, MLManifest) else 0,
                         'target': target_binding, 'member_count': len(membership), 'membership_digest': contract_digest({'members': membership}),
                         'primary_digest': contract_digest({'predictions': decisions.tolist()}),
                         'features': source_binding(meta, f'fold:{block_id}', block_start, block_end, 1, 'causal_oof_primary_features').checksum})
        for local, decision in enumerate(decisions):
            if decision == 0:
                armed, prior = True, 0
                continue
            opening = armed and prior == 0
            prior = 1
            if not opening:
                continue
            row, available = _opportunity(context, prices, frame, decisions, local, block_id, block_end)
            row['available'] = available
            labels.append(row)
            features.append(meta.slice(local, 1))
    if not labels:
        raise StrictModeError('No causal OOF primary opportunities after observed flat')
    rows = pl.DataFrame(labels)
    feature_frame = pl.concat(features)
    return OpportunityLabels(rows, feature_frame, {'folds': bindings, 'candidate_exposure': min(1, context.policy.max_exposure),
        'ordered_completion': contract_digest({'rows': json_value(rows.select('opportunity_id', 'decision_row_id', 'label_available_ns').to_dicts())}),
        'economics': json_value(context.policy), 'raw_train': source_binding(raw, 'original_train', int(prices['start_ns'][0]), int(prices['available_at_ns'][-1]), 1, 'original_training_source').checksum})


def _opportunity(context: AllocationContext, prices: pl.DataFrame, frame: pl.DataFrame, decisions: Array, local: int, block_id: int, block_end: int) -> tuple[dict[str, object], bool]:
    following_flat = np.flatnonzero(decisions[local:] == 0)
    count = int(following_flat[0]) + 1 if following_flat.size else frame.height - local
    subset = frame.slice(local, count)
    support_end = min(block_end, int(subset['datetime'].dt.epoch('ns')[-1]) + int(prices['available_at_ns'][0] - prices['start_ns'][0]))
    times = subset['datetime'].dt.epoch('ns')
    signal_rows = pl.DataFrame({'datetime': subset['datetime'], 'row_id': [f'source:{value}' for value in times],
                                'available_at_ns': times + int(prices['available_at_ns'][0] - prices['start_ns'][0])})
    if context.barriers is not None:
        signal_rows = signal_rows.join(context.barriers, on='datetime', how='left', maintain_order='left')
    policy = replace(context.policy, notional_rate=1.0)
    exposure = decisions[local:local + count] * min(1.0, policy.max_exposure)
    signals = signal_rows.drop('datetime').with_columns(pl.Series('target', exposure))
    start_ns = int(times[0])
    observations = prices.filter((pl.col('start_ns') >= start_ns) & (pl.col('available_at_ns') <= support_end))
    funding = context.train_inputs.funding_events
    if funding is not None:
        funding = funding.filter((pl.col('end_ns') >= start_ns) & (pl.col('start_ns') <= support_end)).with_columns(pl.col('start_ns').clip(lower_bound=start_ns), pl.col('end_ns').clip(upper_bound=support_end), pl.col('time_ns').clip(lower_bound=start_ns, upper_bound=support_end))
    source = source_binding(observations, f'primary-opportunity:{start_ns}', start_ns, support_end, 1, 'recorded_primary_opportunity_prices')
    partition = TradeInputs(context.initial_equity, start_ns, support_end, signals, observations, funding, (source,))
    ledger = trade_execution(partition, policy)
    opening = signal_rows.row(0, named=True)
    row: dict[str, object] = {'opportunity_id': f'opportunity:{start_ns}', 'opening_block': block_id,
        'decision_row_id': opening['row_id'], 'decision_available_at_ns': opening['available_at_ns'],
        'first_fill_ns': None, 'closed_at_ns': None, 'label_available_ns': None, 'net_pnl': None,
        'entry_equity': None, 'entry_notional': None, 'net_return_on_entry_notional': None,
        'unavailable_reason': 'never_filled', 'exit_reason': None,
        'take_profit_bps': opening.get(policy.take_profit_bps_column) if policy.take_profit_bps_column else policy.take_profit_bps,
        'stop_loss_bps': opening.get(policy.stop_loss_bps_column) if policy.stop_loss_bps_column else policy.stop_loss_bps}
    if ledger.episodes.height:
        episode = ledger.episodes.row(0, named=True)
        entry = ledger.fills.filter(pl.col('episode_id') == episode['episode_id']).row(0, named=True)
        notional = abs(float(entry['quantity_delta'])) * float(entry['fill_price'])
        closed = episode['closed_at_ns']
        available = closed is not None and int(closed) <= block_end
        row.update(first_fill_ns=episode['first_fill_ns'], closed_at_ns=closed, label_available_ns=closed if available else None,
                   net_pnl=episode['net_pnl'], entry_equity=episode['entry_equity'], entry_notional=notional,
                   net_return_on_entry_notional=float(episode['net_pnl']) / notional if available else None,
                   unavailable_reason=None if available else 'open_at_opening_block_end', exit_reason=episode['exit_reason'])
        return row, available
    return row, False


__all__: list[str] = []

def admit(manifest: Manifest, params: Mapping[str, object]) -> None:
    from limen.sfd.reference_architecture.lightgbm_binary import lightgbm_binary
    from limen.sfd.reference_architecture.rule_based import rule_based
    from limen.targets.quantile_binary import QuantileBinaryTarget
    from limen.indicators.window_return import window_return

    if manifest.allocation_config is None:
        return
    policy, config = manifest.resolve_trade_policy(params), manifest.backtest_config
    if policy is None or config is None or policy.prediction_mode != 'target_exposure' or policy.notional_rate != 1 or policy.execution_lag_seconds != 0 or config.source_interval_seconds is None or config.execution_data_source is not None:
        raise ValueError('Allocation requires regular co-timed event target exposure, notional_rate=1 and zero lag')
    if manifest.split_walk_forward is not None or getattr(getattr(manifest, '_holdout_binding', None), 'cutoff_ns', None) is not None or manifest.bar_formation is not None or manifest.pre_split_data_selector is not None or manifest.test_data_source_config is not None:
        raise ValueError('Allocation requires one original chronological source and true outer partitions')
    if any(name in vars(manifest) for name in ('prepare_data', '_prepare_data_with_fitted', 'run_model')):
        raise ValueError('Allocation cannot refit replaced preparation/model methods')
    if isinstance(manifest, MLManifest):
        if any(getattr(manifest, key) is not None for key in ('scaler', 'pca_compression_config', 'ablation_config', 'prediction_calibration_config', 'objective', 'data_dict_extension')) or manifest.feature_shims:
            raise ValueError('Allocation prefix fitting lacks causal state for configured fitted preprocessing or selection')
        target = manifest.target_class_config
        if manifest.architecture_params or manifest.architecture_function is not lightgbm_binary or target is None or target.target_class is not QuantileBinaryTarget:
            raise ValueError('Allocation needs an original binary primary with refittable forward target support')
        fitting = resolve_params(target.fit_params, dict(params))
        transform = resolve_params(target.transform_params, dict(params))
        shift = transform.get('shift')
        if isinstance(shift, bool) or not isinstance(shift, int) or shift >= 0 or fitting.get('source_column') != f'ret_{-shift}':
            raise ValueError('Allocation target must declare a negative forward horizon and matching return source')
        if not any(entry.func is window_return and resolve_params(entry.params, dict(params)).get('period', 24) == -shift for entry in manifest.feature_transforms):
            raise ValueError('Allocation forward target requires its original causal window_return source')
    elif not isinstance(manifest, RuleBasedManifest) or manifest.architecture_function is not rule_based or manifest.target_class_config is not None:
        raise ValueError('Allocation requires an unchanged binary rule or refittable ML primary')


def attach(manifest: Manifest, raw: pl.DataFrame, params: Mapping[str, object], data: dict[str, object]) -> dict[str, object]:
    from copy import deepcopy
    from limen.experiment._prepare_trade_context import PreparedTradeContext
    from limen.experiment._dynamic_barriers import prepare_barriers

    if manifest.allocation_config is None:
        return data
    trade = data.get('_trade_context')
    if not isinstance(trade, PreparedTradeContext) or manifest.backtest_config is None:
        raise ValueError('Allocation requires original event source context')
    if any(partition.observations['available_at_ns'].ne(partition.observations['end_ns']).any() for partition in trade.partitions):
        raise ValueError('Allocation requires co-timed regular close observations')
    interval = manifest.backtest_config.source_interval_seconds
    if interval is None:
        raise ValueError('Missing declared allocation source interval')
    train = resolve_split(manifest, raw)[0]
    history = raw.filter(pl.col('datetime') <= train['datetime'][-1])
    barriers = prepare_barriers(manifest.backtest_config, trade.policy, history)
    batches: dict[str, AllocationBatch] = {}
    for name, partition in zip(('train', 'val', 'test'), trade.partitions, strict=True):
        if name == 'train':
            continue
        if isinstance(manifest, MLManifest):
            frame = cast(pl.DataFrame, data[f'x_{name}'])
            primary: Mapping[str, object] = {'x_test': frame}
            meta = frame
        else:
            frame = cast(pl.DataFrame, data[name])
            primary = {'test': frame, 'strategy': data['strategy']}
            conditions = cast(Mapping[str, object], data['strategy'])['conditions']
            excluded = {item['id'] for item in cast(list[dict[str, object]], conditions)}
            meta = frame.select([key for key, dtype in frame.schema.items() if dtype.is_numeric() and key not in excluded])
        signals = partition.signals
        ids = tuple(f'source:{int(value) - int(interval * 1000000000)}' for value in signals['available_at_ns'])
        stop = tuple(signals[trade.policy.stop_loss_bps_column].to_list()) if trade.policy.stop_loss_bps_column else (trade.policy.stop_loss_bps,) * signals.height
        ready = tuple(signals['__barrier_ready__'].to_list()) if '__barrier_ready__' in signals.columns else (True,) * signals.height
        batches[name] = AllocationBatch(name, partition.partition_start_ns, partition.partition_end_ns, ids, tuple(signals['available_at_ns'].to_list()), primary, meta, stop, ready)
    clone = deepcopy(manifest)
    clone.allocation_config = None
    context = AllocationContext(clone, train, dict(params), trade.policy, batches, barriers, trade.partitions[0].initial_equity, trade.partitions[0])
    data['_allocation_context'], data['_allocation_batches'] = context, batches
    return data


def run(manifest: Manifest, data: dict[str, object], params: Mapping[str, object]) -> dict[str, object]:
    from limen.experiment._resolve_trade_policy import resolve_json
    from limen.experiment._prepare_trade_context import finish_trade_result

    config = manifest.allocation_config
    if config is None:
        raise ValueError('Missing configured allocation stage')
    context = data.get('_allocation_context')
    if not isinstance(context, AllocationContext) or dict(context.round_params) != dict(params):
        raise ValueError('Cached allocation context does not match current round')
    factories = {key: value for key, value in config.params.items() if key in ('probability_factory', 'sizing_factory')}
    resolved = cast(dict[str, object], resolve_json({key: value for key, value in config.params.items() if key not in factories}, params))
    for factory in factories.values():
        _ = component_identity(factory)
    return finish_trade_result(data, config.method(data, context=context, **resolved, **factories))
