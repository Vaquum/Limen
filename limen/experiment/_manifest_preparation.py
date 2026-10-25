from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, timezone
from typing import Protocol, TypeVar, cast
import polars as pl

from limen.backtest.trade_contract import TradePolicy
from limen.experiment._prepare_trade_context import FrozenTradeSources, PreparedTradeContext, prepare_trade_context, source_interval, validate_cached_context
from limen.experiment._prepare_backtest_data import prepare_backtest_data
from limen.experiment._resolve_backtest_config import BACKTEST_KEYS, configured_barriers
from limen.experiment._resolve_trade_policy import BacktestConfig, SourceConfig

from limen.sfd.rule_based.config import RuleBasedConfig
from limen.experiment._walk_forward_split import WalkForwardConfig

_M = TypeVar('_M')


def admit_native_date_ends(dates: tuple[date, ...] | None,
                           data_source: SourceConfig | None, test_source: SourceConfig | None) -> None:
    '''Refuse declared native bars whose nominal support crosses a date-window end.'''
    if dates is None:
        return
    from limen.data import HistoricalData
    for source_name, source in (('data_source', data_source), ('test_data_source', test_source)):
        if source is None or getattr(source.method, '__func__', source.method) is not HistoricalData.get_spot_klines:
            continue
        interval = source.params.get('kline_size')
        if interval is None:
            interval = 60
        if type(interval) is not int or interval < 1:
            raise ValueError(f'Manifest {source_name} get_spot_klines kline_size must be a positive integer')
        for name, end in zip(('train_end', 'val_end', 'test_end'), dates[1::2], strict=True):
            stamp = end if isinstance(end, datetime) else datetime.combine(end, datetime.min.time(), timezone.utc)
            if stamp.utcoffset() is None:
                raise TypeError(f'Manifest {source_name} {name}={end!r} must be timezone-aware for native spot klines')
            elapsed = stamp.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
            end_ns = ((elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds) * 1000
            if end_ns % (interval * 1_000_000_000):
                raise ValueError(f'Manifest {source_name} {name}={end!r} must align to the UTC Unix-epoch grid for get_spot_klines kline_size={interval} seconds')


class RulePreparation(Protocol):
    @property
    def strategy(self) -> RuleBasedConfig | None: ...
    @property
    def split_walk_forward(self) -> WalkForwardConfig | None: ...


class SetupManifest(Protocol):
    @property
    def backtest_config(self) -> BacktestConfig | None: ...
    @property
    def data_source_config(self) -> SourceConfig | None: ...
    def resolve_trade_policy(self, round_params: Mapping[str, object]) -> TradePolicy | None: ...
    def resolve_backtest_config(self, round_params: Mapping[str, object]) -> dict[str, float | None]: ...


def finish_prepare_setup(manifest: SetupManifest, raw_splits: list[pl.DataFrame],
                         pairs: list[tuple[list[datetime] | list[int], pl.DataFrame]],
                         round_params: dict[str, object], *, raw_history: pl.DataFrame | None = None) -> tuple[list[pl.DataFrame], list[datetime] | list[int], pl.DataFrame | None, list[pl.DataFrame] | None, PreparedTradeContext | None]:
    all_datetimes = cast(list[datetime] | list[int], [dt for datetimes, _ in pairs for dt in datetimes])
    splits = [frame for _, frame in pairs]
    policy = manifest.resolve_trade_policy(round_params)
    frozen = cast(FrozenTradeSources | None, getattr(manifest, '_frozen_trade_sources', None))
    if callable(frozen):
        frozen = frozen(manifest.backtest_config, round_params)
    from limen.experiment._holdout_sources import verify_sources
    frozen = verify_sources(manifest, raw_splits, splits, round_params, frozen)
    trade = None
    if policy is not None and manifest.backtest_config is not None:
        from limen.experiment._dynamic_barriers import dynamic_barriers, prepare_barriers
        interval = manifest.backtest_config.source_interval_seconds or getattr(manifest, '_holdout_interval_seconds', None) or source_interval(manifest.data_source_config)
        if any(not split.height for split in raw_splits):
            raise ValueError('Event execution requires nonempty declared partitions')
        barriers = None
        if dynamic_barriers(manifest.backtest_config):
            if raw_history is None:
                raise ValueError('Dynamic barriers require original unsplit admitted history')
            history = raw_history.filter(pl.col('datetime') <= raw_splits[-1]['datetime'][-1])
            barriers = prepare_barriers(manifest.backtest_config, policy, history)
        trade = prepare_trade_context(manifest.backtest_config, policy, raw_splits, splits, round_params, interval_seconds=interval, frozen_sources=frozen, barriers=barriers)
        from limen.experiment._holdout_research import binding_of
        held = binding_of(manifest)
        if held is not None and held.cutoff_ns is not None:
            from limen.experiment._resolve_trade_policy import holdout_event_policy
            from limen.experiment._holdout_source import source_workflow_active
            policy = holdout_event_policy(manifest.backtest_config, round_params, sources=source_workflow_active.get())
            for inputs in trade.partitions:
                if inputs.partition_end_ns > held.cutoff_ns:
                    raise ValueError('Holdout event partition crosses the reserved cutoff')
        splits = [split.drop([name for name in ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns') if name in split.columns]) for split in splits]
    prepared = prepare_backtest_data(splits, all_datetimes, configured=policy is None and configured_barriers(manifest.backtest_config))
    return (*prepared, trade)


def finish_sensor_input(manifest: _M, data: pl.DataFrame, lookback: int,
                        params: dict[str, object], fitted: dict[str, object], decisions: pl.DataFrame | None, *,
                        scale: Callable[[_M, pl.DataFrame, dict[str, object], dict[str, object], bool], tuple[pl.DataFrame, dict[str, object]]],
                        compress: Callable[[_M, pl.DataFrame, dict[str, object], dict[str, object]], pl.DataFrame]) -> tuple[pl.DataFrame, int]:
    data = _assessment_schema(manifest, data, 'input')
    data, _ = scale(manifest, data, params, fitted, False)
    data = data.fill_nan(None)
    data = compress(manifest, data, params, fitted)
    data = _assessment_schema(manifest, data, 'model')
    if decisions is not None:
        data = data.join(decisions, on='datetime', how='left', maintain_order='left')
    return data, lookback


def _assessment_schema(manifest: object, frame: pl.DataFrame, stage: str) -> pl.DataFrame:
    schemas = cast(dict[str, list[dict[str, str]]] | None, getattr(manifest, '_holdout_assessment_schemas', None))
    if schemas is None:
        return frame
    expected = schemas[stage]
    excluded = {'datetime', getattr(manifest, 'target_column', None), '__trade_available_at_ns__'}
    actual = {name: str(dtype) for name, dtype in frame.schema.items() if name not in excluded}
    if actual != {field['name']: field['dtype'] for field in expected}:
        message = f'Holdout {stage} schema mismatch: expected {expected}, received {actual}'
        record = cast(Callable[[str], None], vars(manifest)['_holdout_assessment_failure'])
        record(message)
        raise ValueError(message)
    return frame.select([name for name in frame.columns if name in excluded] + [field['name'] for field in expected])


def apply_backtest_cost(manifest: SetupManifest, data: dict[str, object], round_params: Mapping[str, object]) -> None:
    from limen.experiment._backtest_provenance import preflight_backtest
    policy = manifest.resolve_trade_policy(round_params)
    validate_cached_context(manifest.backtest_config, policy, data, round_params)
    if configured_barriers(manifest.backtest_config) and policy is None:
        data['_backtest_configured'] = True
    else:
        _ = data.pop('_backtest_configured', None)
    _ = data.pop('_trade_ledger', None)
    preflight_backtest(data)
    resolved = manifest.resolve_backtest_config(round_params)
    for key in BACKTEST_KEYS:
        if key in resolved:
            data[f'backtest_{key}'] = resolved[key]
        else:
            _ = data.pop(f'backtest_{key}', None)


def rule_predicates(manifest: RulePreparation, data: pl.DataFrame, params: dict[str, object], *, split: str = 'input') -> tuple[pl.DataFrame, tuple[str, ...]]:
    from limen.sfd.rule_based.predicates import build_predicate
    config = manifest.strategy
    if config is None:
        raise ValueError('_finalize_rule_based_data manifest has no strategy configured')
    conditions = [condition for condition in config.conditions if 'type' in condition]
    names = tuple(cast(str, condition['id']) for condition in conditions)
    collisions = sorted(set(data.columns) & set(names))
    if collisions:
        raise ValueError(f"Rule-based condition ids collide with existing columns in {split!r} split: {collisions}. Rename the affected conditions.")
    return data.with_columns([build_predicate(condition, params).fill_null(False).alias(condition['id']) for condition in conditions]), names


def finish_rule_data(manifest: RulePreparation, splits: list[pl.DataFrame], datetimes: list[datetime] | list[int], params: dict[str, object]) -> dict[str, object]:
    from limen.data.utils import split_data_to_rule_based_prep_output
    from limen.experiment._fold_partitions import retained
    data = cast(dict[str, object], split_data_to_rule_based_prep_output(splits, datetimes))
    if manifest.split_walk_forward is not None:
        cast(dict[str, object], data['_alignment'])['test_datetimes'] = splits[2]['datetime'].to_list()
    names: tuple[str, ...] = ()
    for split in ('train', 'val', 'test'):
        frame, names = rule_predicates(manifest, cast(pl.DataFrame, data[split]), params, split=split)
        data[split] = frame
        data['x_' + split] = frame.select(names).cast(pl.Float64).to_numpy()
    assert manifest.strategy is not None
    data['strategy'] = {'conditions': manifest.strategy.conditions, 'entry': manifest.strategy.entry}
    return retained(manifest, splits, data, feature_columns=names)
