from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from importlib import import_module
from inspect import getsourcefile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl

from limen.backtest.trade_contract import JsonValue, TradePolicy, finite_number

if TYPE_CHECKING:
    from limen.experiment._resolve_trade_policy import BacktestConfig

__all__: list[str] = []


def dynamic_barriers(config: object) -> bool:
    return any(getattr(config, name, None) is not None for name in ('take_profit_bps_column', 'stop_loss_bps_column', 'barrier_features'))


def reject_promotion(manifest: object) -> None:
    if dynamic_barriers(getattr(manifest, 'backtest_config', None)) or getattr(manifest, 'allocation_config', None) is not None:
        raise ValueError('research-only dynamic barriers/allocation cannot be promoted')


def admit_research_run(manifest: object, workers: int, resume: bool) -> None:
    configured = dynamic_barriers(getattr(manifest, 'backtest_config', None)) or getattr(manifest, 'allocation_config', None) is not None
    if configured and (workers != 1 or resume):
        raise ValueError('Research-only dynamic barriers/allocation require fresh serial runs; resume and native workers>1 are unsupported')


def barrier_policy(config: BacktestConfig, params: Mapping[str, object]) -> dict[str, object]:
    from limen.experiment._resolve_trade_policy import resolve_json

    columns = [getattr(config, f'{name}_column') for name in ('take_profit_bps', 'stop_loss_bps')]
    if any(value is not None and (not isinstance(value, str) or not value.isidentifier() or value.startswith('__')) for value in columns):
        raise ValueError('Barrier column names must be literal public identifiers')
    if dynamic_barriers(config) and (not any(columns) or config.barrier_features is None):
        raise ValueError('Dynamic columns require an isolated barrier_features producer')
    spec = None
    if config.barrier_features is not None:
        method = config.barrier_features.method
        module, name = getattr(method, '__module__', ''), getattr(method, '__qualname__', '')
        if not module or not name or '<' in name or '.' in name or getattr(import_module(module), name) is not method:
            raise ValueError('Barrier producer must have an importable source identity')
        path = getsourcefile(method)
        if path is None:
            raise ValueError('Barrier producer requires readable source')
        spec = {'method': f'{module}.{name}', 'source_sha256': sha256(Path(path).read_bytes()).hexdigest(), 'params': resolve_json(config.barrier_features.params, params)}
    interval = config.source_interval_seconds
    if interval is not None and (isinstance(interval, str) or finite_number(interval, 'source interval') <= 0):
        raise ValueError('source_interval_seconds must be positive literal metadata')
    return dict(take_profit_bps_column=columns[0], stop_loss_bps_column=columns[1], barrier_spec=spec, source_interval_seconds=interval)


def prepare_barriers(config: BacktestConfig, policy: TradePolicy, raw: pl.DataFrame) -> pl.DataFrame | None:
    from limen.backtest.execution_events import validate_signal_barriers
    from limen.backtest.trade_contract import TradeInputs

    if not dynamic_barriers(config):
        return None
    if config.barrier_features is None or policy.barrier_spec is None:
        raise ValueError('Missing private barrier producer')
    if raw['datetime'].is_duplicated().any() or not raw['datetime'].is_sorted():
        raise ValueError('Private barriers require unique ordered unsplit history')
    source = raw.select('datetime', 'open', 'high', 'low', 'close')
    resolved = cast(Mapping[str, object], policy.barrier_spec['params'])
    produced = config.barrier_features.method(source.clone(), **resolved)
    if not isinstance(produced, pl.DataFrame) or not produced['datetime'].equals(source['datetime']):
        raise ValueError('Barrier producer changed original source membership')
    columns = [name for name in (policy.take_profit_bps_column, policy.stop_loss_bps_column) if name is not None]
    if set(['__barrier_ready__', *columns]) - set(produced.columns):
        raise ValueError('Barrier producer omitted declared distances/readiness')
    selected = produced.select('datetime', '__barrier_ready__', *columns)
    validation = selected.with_row_index('row_id').with_columns(pl.lit(0, dtype=pl.Int64).alias('available_at_ns'))
    validate_signal_barriers(TradeInputs(1, 0, 0, validation, pl.DataFrame(), None, ()), policy)
    if policy.barrier_spec['method'] == 'limen.features.volatility_barriers.volatility_barriers':
        warmup = int(cast(int, resolved.get('min_periods', 24))) + 1
        if selected['__barrier_ready__'].to_list() != [index >= warmup for index in range(raw.height)]:
            raise ValueError('Builtin barrier readiness disagrees with declared warmup')
    return selected


def reject_yaml_promotion(reference: Mapping[str, object]) -> None:
    sfd = reference.get('sfd')
    manifest = sfd.get('manifest') if isinstance(sfd, Mapping) else None
    if isinstance(manifest, Mapping):
        config = manifest.get('backtest')
        configured = isinstance(config, Mapping) and any(config.get(key) is not None for key in ('take_profit_bps_column', 'stop_loss_bps_column', 'barrier_features'))
        if configured or manifest.get('allocation') is not None:
            raise ValueError('research-only dynamic barriers/allocation cannot be promoted')
