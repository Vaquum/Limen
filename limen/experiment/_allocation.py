from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from limen.backtest.trade_contract import TradeInputs, TradePolicy, contract_digest, json_value, source_binding
from limen.sfd.reference_architecture.direction_sizing import _identity

if TYPE_CHECKING:
    from limen.experiment.manifest_core import Manifest


@dataclass(frozen=True)
class AllocationConfig:
    method: Callable[..., dict[str, object]]
    params: Mapping[str, object]

    def __post_init__(self) -> None:
        _identity(self.method)


@dataclass(frozen=True)
class AllocationBatch:
    batch_id: str
    partition_start_ns: int
    partition_end_ns: int
    row_ids: tuple[str, ...]
    available_at_ns: tuple[int, ...]
    primary_inputs: Mapping[str, object]
    meta_features: pl.DataFrame
    stop_loss_bps: tuple[float | None, ...]
    barrier_ready: tuple[bool, ...]

    def digest(self) -> str:
        identities = pl.DataFrame({'row_id': self.row_ids, 'available_at_ns': self.available_at_ns,
                                   'stop_loss_bps': self.stop_loss_bps, 'barrier_ready': self.barrier_ready})
        if identities.height != len(self.stop_loss_bps) or identities.height != len(self.barrier_ready) or identities.height != self.meta_features.height or len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError('Allocation batch requires complete unique ordered membership')
        features = source_binding(self.meta_features, self.batch_id, self.partition_start_ns, self.partition_end_ns, 1, 'allocation_features').checksum
        primary = {}
        for key, value in self.primary_inputs.items():
            if isinstance(value, pl.DataFrame):
                primary[key] = source_binding(value, key, self.partition_start_ns, self.partition_end_ns, 1, 'allocation_primary_inputs').checksum
            else:
                primary[key] = contract_digest({'value': json_value(value)})
        return contract_digest({'batch_id': self.batch_id, 'start': self.partition_start_ns, 'end': self.partition_end_ns,
                                'rows': json_value(identities.to_dicts()), 'features': features, 'primary': primary})


@dataclass(frozen=True)
class OpportunityLabels:
    rows: pl.DataFrame
    features: pl.DataFrame
    binding: Mapping[str, object]


@dataclass(frozen=True)
class AllocationContext:
    manifest: Manifest
    raw_train: pl.DataFrame
    round_params: Mapping[str, object]
    policy: TradePolicy
    batches: Mapping[str, AllocationBatch]
    barriers: pl.DataFrame | None
    initial_equity: float
    train_inputs: TradeInputs


__all__: list[str] = []


def admit(manifest: Manifest, params: Mapping[str, object]) -> None:
    from limen.experiment.manifest_core import MLManifest, RuleBasedManifest, _resolve_params
    from limen.sfd.reference_architecture.lightgbm_binary import lightgbm_binary
    from limen.sfd.reference_architecture.rule_based import rule_based
    from limen.targets.quantile_binary import QuantileBinaryTarget
    from limen.indicators.window_return import window_return

    if manifest.allocation_config is None:
        return
    policy, config = manifest.resolve_trade_policy(params), manifest.backtest_config
    if policy is None or config is None or policy.prediction_mode != 'target_exposure' or policy.notional_rate != 1 or policy.execution_lag_seconds != 0 or config.source_interval_seconds is None or config.execution_data_source is not None:
        raise ValueError('Allocation requires regular co-timed event target exposure, notional_rate=1 and zero lag')
    if manifest.split_walk_forward is not None or getattr(manifest, '_holdout_binding', None) is not None or manifest.bar_formation is not None or manifest.pre_split_data_selector is not None or manifest.test_data_source_config is not None:
        raise ValueError('Allocation requires one original chronological source and true outer partitions')
    if any(name in vars(manifest) for name in ('prepare_data', '_prepare_data_with_fitted', 'run_model')):
        raise ValueError('Allocation cannot refit replaced preparation/model methods')
    if isinstance(manifest, MLManifest):
        if any(getattr(manifest, key) is not None for key in ('scaler', 'pca_compression_config', 'ablation_config', 'prediction_calibration_config', 'objective', 'data_dict_extension')) or manifest.feature_shims:
            raise ValueError('Allocation prefix fitting lacks causal state for configured fitted preprocessing or selection')
        target = manifest.target_class_config
        if manifest.architecture_params or manifest.architecture_function is not lightgbm_binary or target is None or target.target_class is not QuantileBinaryTarget:
            raise ValueError('Allocation needs an original binary primary with refittable forward target support')
        fitting = _resolve_params(target.fit_params, dict(params))
        transform = _resolve_params(target.transform_params, dict(params))
        shift = transform.get('shift')
        if isinstance(shift, bool) or not isinstance(shift, int) or shift >= 0 or fitting.get('source_column') != f'ret_{-shift}':
            raise ValueError('Allocation target must declare a negative forward horizon and matching return source')
        if not any(entry.func is window_return and _resolve_params(entry.params, dict(params)).get('period', 24) == -shift for entry in manifest.feature_transforms):
            raise ValueError('Allocation forward target requires its original causal window_return source')
    elif not isinstance(manifest, RuleBasedManifest) or manifest.architecture_function is not rule_based or manifest.target_class_config is not None:
        raise ValueError('Allocation requires an unchanged binary rule or refittable ML primary')


def attach(manifest: Manifest, raw: pl.DataFrame, params: Mapping[str, object], data: dict[str, object]) -> dict[str, object]:
    from copy import deepcopy
    from typing import cast
    from limen.experiment.manifest_core import MLManifest, _resolve_split
    from limen.experiment._prepare_trade_context import PreparedTradeContext
    from limen.experiment._dynamic_barriers import prepare_barriers

    if manifest.allocation_config is None:
        return data
    trade = data.get('_trade_context')
    if not isinstance(trade, PreparedTradeContext) or manifest.backtest_config is None:
        raise ValueError('Allocation requires original event source context')
    if any(partition.observations['available_at_ns'].ne(partition.observations['end_ns']).any() for partition in trade.partitions):
        raise ValueError('Allocation requires co-timed regular close observations')
    train = _resolve_split(manifest, raw)[0]
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
        ids = tuple(f'source:{int(value) - int(manifest.backtest_config.source_interval_seconds * 1000000000)}' for value in signals['available_at_ns'])
        stop = tuple(signals[trade.policy.stop_loss_bps_column].to_list()) if trade.policy.stop_loss_bps_column else (trade.policy.stop_loss_bps,) * signals.height
        ready = tuple(signals['__barrier_ready__'].to_list()) if '__barrier_ready__' in signals.columns else (True,) * signals.height
        batches[name] = AllocationBatch(name, partition.partition_start_ns, partition.partition_end_ns, ids, tuple(signals['available_at_ns'].to_list()), primary, meta, stop, ready)
    clone = deepcopy(manifest)
    clone.allocation_config = None
    context = AllocationContext(clone, train, dict(params), trade.policy, batches, barriers, trade.partitions[0].initial_equity, trade.partitions[0])
    data['_allocation_context'], data['_allocation_batches'] = context, batches
    return data


def run(manifest: Manifest, data: dict[str, object], params: Mapping[str, object]) -> dict[str, object] | None:
    from typing import cast
    from limen.experiment._resolve_trade_policy import resolve_json
    from limen.experiment._prepare_trade_context import finish_trade_result

    config = manifest.allocation_config
    if config is None:
        return None
    context = data.get('_allocation_context')
    if not isinstance(context, AllocationContext) or dict(context.round_params) != dict(params):
        raise ValueError('Cached allocation context does not match current round')
    factories = {key: value for key, value in config.params.items() if key in ('probability_factory', 'sizing_factory')}
    resolved = cast(dict[str, object], resolve_json({key: value for key, value in config.params.items() if key not in factories}, params))
    for factory in factories.values():
        _identity(factory)
    manifest._apply_backtest_cost(data, dict(params))
    return finish_trade_result(data, config.method(data, context=context, **resolved, **factories))
