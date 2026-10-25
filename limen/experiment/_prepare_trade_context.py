from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from inspect import isfunction, signature
from typing import TYPE_CHECKING, cast

import polars as pl

from limen.backtest._event_plan import event_plan
from limen.backtest.execution_events import OBSERVATION_COLUMNS
from limen.backtest.funding_adapter import prepare_funding, resolve_adapter
from limen.backtest.trade_contract import NANOSECONDS, RULE_VERSION, JsonValue, TradeInputs, TradeLedger, TradePolicy, contract_digest, export_trade_contract, json_value, source_binding
from limen.experiment._fold_preparation import FoldCompression, FoldFeatures, FoldPreparation, FoldScaler, PreparedFolds
from limen.experiment._resolve_trade_policy import BacktestConfig, SourceConfig, resolve_json, resolve_number

if TYPE_CHECKING:
    from limen.experiment._column_shims import FittedColumnShims
    from limen.targets.trade_outcome import TradeTargetContext, OutcomeLabels


_METADATA = ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns')
FrozenTradeSources = tuple[pl.DataFrame | None, pl.DataFrame | None] | Callable[[BacktestConfig | None, Mapping[str, object]], tuple[pl.DataFrame | None, pl.DataFrame | None] | None]


@dataclass(frozen=True)
class PreparedTradeContext:
    policy: TradePolicy
    partitions: tuple[TradeInputs, ...]
    model_rows: tuple[pl.DataFrame, ...]
    source_settings: tuple[tuple[Callable[..., object], JsonValue] | None, ...]
    native_support: tuple[pl.DataFrame, ...]


def source_settings(config: BacktestConfig, params: Mapping[str, object]) -> tuple[tuple[Callable[..., object], JsonValue] | None, ...]:
    sources = (config.execution_data_source, config.funding.data_source if config.funding is not None else None)
    return tuple(None if source is None else (source.method, resolve_json(source.params, params)) for source in sources)


def load_source(source: SourceConfig, params: Mapping[str, object]) -> pl.DataFrame:
    values = cast(dict[str, object], resolve_json(source.params, params))
    method = source.method
    if isfunction(method) and '.' in method.__qualname__ and '<locals>' not in method.__qualname__ and next(iter(signature(method).parameters), None) == 'self':
        module = import_module('limen.experiment.manifest_core')
        configure = cast(Callable[[Callable[..., object], dict[str, object]], SourceConfig], vars(module)['DataSourceConfig'])
        resolve = cast(Callable[[SourceConfig], object], vars(module)['DataSourceResolver'].resolve)
        result = resolve(configure(method, values))
    else:
        result = method(**values)
    if not isinstance(result, pl.DataFrame):
        raise ValueError('Trade source must return a Polars DataFrame')
    return result


_load = load_source


def source_interval(source: SourceConfig | None) -> object:
    value = None if source is None else source.params.get('kline_size')
    return value if value is not None or source is None else source.params.get('klines_size')


def normalize_observations(data: pl.DataFrame, *, interval_seconds: object = None, timestamp_role: str = 'start') -> pl.DataFrame:
    if timestamp_role not in ('start', 'end'):
        raise ValueError('Source timestamp role must be start or end')
    if set(_METADATA) & set(data.columns) and not set(_METADATA) <= set(data.columns):
        raise ValueError('Explicit source timing requires all four metadata columns')
    if set(OBSERVATION_COLUMNS) <= set(data.columns):
        return data.select(OBSERVATION_COLUMNS)
    if not {'datetime', 'open', 'high', 'low', 'close'} <= set(data.columns):
        raise ValueError('Recorded execution data requires OHLC, datetime and causal interval metadata')
    if not isinstance(data['datetime'].dtype, pl.Datetime):
        raise ValueError('Model-bar datetime must declare UTC datetime units')
    starts = data['datetime'].dt.epoch('ns')
    if set(_METADATA) <= set(data.columns):
        timing = data.select(_METADATA)
        anchor = 'start_ns' if timestamp_role == 'start' else 'end_ns'
        if not timing[anchor].equals(starts.rename(anchor)):
            raise ValueError('Execution/model source start timestamps disagree')
    else:
        if interval_seconds is None and 'base_interval' in data.columns and data['base_interval'].n_unique() == 1:
            interval_seconds = data['base_interval'][0]
        interval = round(resolve_number(interval_seconds, {}, 'recorded regular source interval') * NANOSECONDS)
        if interval <= 0:
            raise ValueError('Missing interval metadata or source interval is not verified regular')
        differences = starts.diff().drop_nulls()
        invalid = differences.ne(interval) if timestamp_role == 'start' else ((differences < interval) | (differences % interval != 0))
        if invalid.any():
            raise ValueError('Missing interval metadata or source interval is not verified regular')
        if timestamp_role == 'end':
            starts = starts - interval
        timing = pl.DataFrame({'start_ns': starts, 'end_ns': starts + interval, 'open_available_at_ns': starts, 'available_at_ns': starts + interval})
    return timing.with_columns(pl.Series('row_id', [f'row:{index}' for index in range(data.height)]), *[data[key] for key in ('open', 'high', 'low', 'close')]).select(OBSERVATION_COLUMNS)


def admit_formation(function: Callable[..., object]) -> None:
    from limen.data.bars import liquidity_bars, trade_bars, volume_bars
    if function not in (liquidity_bars, trade_bars, volume_bars):
        raise ValueError('Event execution requires recoverable contiguous native support and constituent counts from a standard formation')


def _bar_support(raw: pl.DataFrame, bars: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    if not raw.height or not bars.height:
        raise ValueError('Event execution requires nonempty declared partitions')
    timing = prices.select(_METADATA)
    if any(dtype != pl.Int64 for dtype in timing.dtypes) or timing.null_count().sum_horizontal()[0]:
        raise ValueError('Native support requires complete integer UTC nanosecond clocks')
    if ((prices['start_ns'] > prices['end_ns']) | (prices['open_available_at_ns'] < prices['start_ns']) | (prices['available_at_ns'] < prices['end_ns']) | (prices['available_at_ns'] < prices['open_available_at_ns'])).any():
        raise ValueError('Native support requires source availability at or after interval end and causal interval/open clocks')
    if (prices['start_ns'].slice(1) < prices['end_ns'].head(-1)).any():
        raise ValueError('Native support intervals overlap or are out of order')
    times = raw['datetime'].dt.epoch('ns')
    if times.null_count() or times.is_duplicated().any() or not times.is_sorted():
        raise ValueError('Native support requires unique ordered source identities')
    positions = {value: index for index, value in enumerate(times)}
    counts = bars['bar_count'].to_list() if 'bar_count' in bars.columns else [1] * bars.height
    available = cast(list[int], prices['available_at_ns'].to_list())
    values = {name: cast(list[float], prices[name].to_list()) for name in ('open', 'high', 'low', 'close')}
    formed_values = {name: bars[name].to_list() for name in values if name in bars.columns}
    rows: list[dict[str, object]] = []
    previous = 0
    for row, (instant, count) in enumerate(zip(bars['datetime'].dt.epoch('ns'), counts, strict=True)):
        first = positions.get(instant)
        if first is None or type(count) is not int or count < 1 or first < previous or first + count > prices.height:
            raise ValueError('Model bar has no recoverable nonoverlapping native membership')
        last = first + count
        expected = {'open': values['open'][first], 'high': max(values['high'][first:last]), 'low': min(values['low'][first:last]), 'close': values['close'][last - 1]}
        if any(column[row] != expected[name] for name, column in formed_values.items()):
            raise ValueError('Model bar disagrees with its contiguous native support')
        rows.append({'native_first': first, 'native_last': last - 1, 'available_at_ns': max(available[first:last])})
        previous = last
    result = pl.DataFrame(rows)
    if not result['available_at_ns'].is_sorted() or result['available_at_ns'].max() > prices['available_at_ns'][-1]:
        raise ValueError('Derived decisions overtake another bar or escape the true source partition')
    return result


def prepare_trade_context(config: BacktestConfig, policy: TradePolicy, raw_splits: list[pl.DataFrame], bars: list[pl.DataFrame], params: Mapping[str, object], *, interval_seconds: object = None, frozen_sources: FrozenTradeSources | None = None, barriers: pl.DataFrame | None = None) -> PreparedTradeContext:
    interval_seconds = None if interval_seconds is None else resolve_number(interval_seconds, params, 'recorded source interval')
    if callable(frozen_sources):
        frozen_sources = frozen_sources(config, params)
    execution = None
    if config.execution_data_source is not None:
        declared_interval = source_interval(config.execution_data_source)
        execution_interval = None if declared_interval is None else resolve_number(declared_interval, params, 'execution source interval')
        raw_execution = load_source(config.execution_data_source, params) if frozen_sources is None else frozen_sources[0]
        if not isinstance(raw_execution, pl.DataFrame):
            raise ValueError('Declared execution source requires its frozen observation frame')
        execution = normalize_observations(raw_execution, interval_seconds=execution_interval)
    history = None
    if config.funding is not None and config.funding.data_source is not None:
        history = load_source(config.funding.data_source, params) if frozen_sources is None else frozen_sources[1]
        if not isinstance(history, pl.DataFrame):
            raise ValueError('Declared funding source requires its frozen observation frame')
    initial = resolve_number(config.initial_equity, params, 'initial equity')
    inputs: list[TradeInputs] = []
    model_rows: list[pl.DataFrame] = []
    native_support: list[pl.DataFrame] = []
    for index, (raw, model) in enumerate(zip(raw_splits, bars, strict=True)):
        raw_prices = normalize_observations(raw, interval_seconds=interval_seconds, timestamp_role=config.source_timestamp_role)
        mapping = _bar_support(raw, model, raw_prices).with_columns(model['datetime'], pl.Series('row_id', [f'partition:{index}:bar:{position}' for position in range(model.height)]))
        if barriers is not None:
            if 'bar_count' in model.columns and model['bar_count'].ne(1).any():
                raise ValueError('Dynamic barriers reject resampled/ambiguous model bars')
            mapping = mapping.join(barriers, on='datetime', how='left', maintain_order='left')
        if mapping['datetime'].is_duplicated().any():
            raise ValueError('Trade model rows require unambiguous source identity')
        start, end = int(raw_prices['start_ns'][0]), int(raw_prices['available_at_ns'][-1])
        observations = raw_prices if execution is None else execution.filter((pl.col('start_ns') <= end) & (pl.col('end_ns') >= start))
        funding = prepare_funding(policy.funding, history, start, end) if policy.funding is not None else None
        sources = [source_binding(observations, f'execution:partition:{index}', start, end, _precision(observations), 'causal_recorded_prices')]
        native = raw_prices.select('row_id', *_METADATA)
        sources.append(source_binding(mapping.select('datetime', 'row_id', 'native_first', 'native_last', 'available_at_ns'), f'model:partition:{index}', start, end, _precision(raw_prices), 'model_bar_membership_and_availability'))
        sources.append(source_binding(native, f'native:partition:{index}', start, end, 1, 'native_constituent_clocks'))
        if funding is not None:
            sources.append(source_binding(funding, f'funding:partition:{index}', start, end, 1, 'normalized_funding_support'))
        signals = mapping.drop('datetime', 'native_first', 'native_last').with_columns(pl.lit(None, dtype=pl.Float64).alias('target'))
        partition = TradeInputs(initial, start, end, signals, observations, funding, tuple(sources))
        _ = event_plan(partition, policy)
        inputs.append(partition)
        model_rows.append(mapping)
        native_support.append(native)
    return PreparedTradeContext(policy, tuple(inputs), tuple(model_rows), source_settings(config, params), tuple(native_support))


def _precision(observations: pl.DataFrame) -> int:
    differences = observations['available_at_ns'].diff().drop_nulls()
    positive = differences.filter(differences > 0)
    return int(cast(int, positive.min())) if positive.len() else 1


def validate_cached_context(config: BacktestConfig | None, policy: TradePolicy | None, data: Mapping[str, object], params: Mapping[str, object]) -> None:
    inputs = data.get('_trade_inputs')
    context = data.get('_trade_context')
    equity_matches = policy is None or (config is not None and isinstance(inputs, TradeInputs) and inputs.initial_equity == resolve_number(config.initial_equity, params, 'initial equity'))
    sources_match = policy is None or (config is not None and isinstance(context, PreparedTradeContext) and context.source_settings == source_settings(config, params))
    if policy != data.get('_trade_policy') or not equity_matches or not sources_match:
        raise ValueError('Cached trade preparation does not match this round; refresh preparation')


def attach_trade_context(data: Mapping[str, object], context: PreparedTradeContext | None, retained: list[pl.DataFrame]) -> None:
    if context is None:
        return
    partitions = tuple(select_trade_rows(inputs, mapping, rows) for inputs, mapping, rows in zip(context.partitions, context.model_rows, retained, strict=True))
    mutable = cast(dict[str, object], data)
    for name, rows in zip(('train', 'val', 'test'), retained, strict=True):
        frame = mutable.get(name)
        if isinstance(frame, pl.DataFrame):
            mutable[name] = frame.with_columns(rows['datetime'])
    mutable['_trade_context'] = replace(context, partitions=partitions)
    mutable['_trade_policy'] = context.policy
    mutable['_trade_inputs'] = partitions[2]
    contract = export_trade_contract(context.policy, partitions[2])
    mutable['trade_contract'] = contract
    mutable['trade_contract_digest'] = contract_digest(contract)
    alignment = mutable.get('_alignment')
    if not isinstance(alignment, dict):
        raise ValueError('Trade execution requires round alignment metadata')
    cast(dict[str, object], alignment).update({'trade_contract': contract, 'trade_contract_digest': contract_digest(contract), '_trade_inputs': partitions[2], '_trade_policy': context.policy})


def select_trade_rows(inputs: TradeInputs, mapping: pl.DataFrame, rows: pl.DataFrame) -> TradeInputs:
    identities = rows.select('datetime').join(mapping, on='datetime', how='left', maintain_order='left')
    if identities['row_id'].null_count() or identities['row_id'].is_duplicated().any():
        raise ValueError('Prepared inference rows do not match source identity')
    signals = identities.select('row_id').join(inputs.signals, on='row_id', how='left', maintain_order='left')
    return replace(inputs, signals=signals)


def persist_ledger(data: Mapping[str, object], ledger: TradeLedger) -> None:
    alignment = data.get('_alignment')
    if not isinstance(alignment, dict):
        raise ValueError('Trade ledger requires bound round alignment')
    cast(dict[str, object], alignment)['trade_ledger'] = json_value({key: getattr(ledger, key).to_dicts() for key in ('states', 'intents', 'fills', 'episodes', 'funding')})
    cast(dict[str, object], data)['_trade_ledger'] = ledger


def finish_trade_result(data: Mapping[str, object], result: dict[str, object]) -> dict[str, object]:
    policy = data.get('_trade_policy')
    if not isinstance(policy, TradePolicy):
        return result
    mode = result.get('_prediction_mode', getattr(result.get('_model'), 'prediction_mode', 'binary'))
    if mode != policy.prediction_mode:
        raise ValueError('Model output mode does not match manifest trade policy')
    if data.get('_trade_ledger') is None:
        from limen.backtest.execution_events import with_predictions
        from limen.backtest.trade_execution import trade_execution

        predictions = result.get('_preds')
        if predictions is None:
            raise ValueError('Event execution requires causal model predictions')
        inputs = data.get('_trade_inputs')
        if not isinstance(inputs, TradeInputs):
            raise ValueError('Event execution requires its original source context')
        ledger = trade_execution(with_predictions(inputs, predictions), policy)
        persist_ledger(data, ledger)
        metrics = {f'backtest_{key}': value for key, value in ledger.metrics.items()}
        result = {key: value for key, value in result.items() if not key.startswith('backtest_')}
        result.update(metrics)
    return result


def target_context(context: PreparedTradeContext | None, index: int) -> TradeTargetContext:
    from limen.targets.trade_outcome import TradeTargetContext

    if context is None:
        raise ValueError('Trade outcomes require configured execution economics before target fitting')
    original = context.partitions[index]
    signals = context.model_rows[index].select('datetime', 'row_id').join(original.signals, on='row_id', how='left', maintain_order='left')
    inputs = replace(original, signals=signals)
    return TradeTargetContext(context.policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(context.policy, inputs)))


def sensor_decisions(raw: pl.DataFrame, bars: pl.DataFrame, *, interval_seconds: object = None, timestamp_role: str = 'start') -> pl.DataFrame:
    prices = normalize_observations(raw, interval_seconds=interval_seconds, timestamp_role=timestamp_role)
    support = _bar_support(raw, bars, prices)
    return bars.select('datetime').with_columns(support['available_at_ns'].rename('__trade_available_at_ns__'))


def resolve_component_kwargs(architecture: object, data: Mapping[str, object], kwargs: dict[str, object], params: Mapping[str, object]) -> None:
    from limen.yaml.resolver import resolve

    supports_labels = bool(getattr(architecture, 'requires_trade_outcomes', False))
    if '_trade_labels' in data and not supports_labels:
        raise ValueError('This architecture cannot fit unavailable trade labels; use an architecture with private availability masks')
    for key in ('direction_params', 'sizing_params'):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = resolve_json(kwargs[key], params)
    if supports_labels:
        for key in ('direction_factory', 'sizing_factory'):
            if key not in kwargs:
                continue
            value = kwargs[key]
            if isinstance(value, str):
                kwargs[key] = resolve(value)
            if not callable(kwargs[key]):
                raise ValueError(f'{key} must resolve to a component factory')


def validate_inference_contract(policy: TradePolicy | None, mode: str, contract: Mapping[str, JsonValue], binding: object) -> None:
    if policy is None or contract.get('rule_version') != RULE_VERSION or json_value(policy) != contract.get('policy') or policy.prediction_mode != mode:
        raise ValueError('Sensor rules, funding or output mode differ from the frozen contract')
    if policy.funding is not None:
        _ = resolve_adapter(policy.funding)
    model_binding = cast(Mapping[str, object], binding) if isinstance(binding, Mapping) else None
    if (mode == 'target_exposure' or binding is not None) and (model_binding is None or model_binding.get('trade_contract_digest') != contract_digest(contract)):
        raise ValueError('Sensor model does not belong to the frozen trade contract')


__all__ = ['FoldCompression', 'FoldFeatures', 'FoldPreparation', 'FoldScaler', 'PreparedFolds', 'PreparedTradeContext', 'attach_trade_context', 'finish_trade_result', 'normalize_observations', 'persist_ledger', 'prepare_trade_context', 'select_trade_rows', 'source_interval']

def attach_outcomes(data: dict[str, object], labels: Sequence[OutcomeLabels], raw: pl.DataFrame | None, retained: pl.DataFrame, *, scale: FoldScaler, compress: FoldCompression, deterministic: bool, dropped: Sequence[str] = ()) -> None:
    from limen.experiment._fold_preparation import prepare_folds
    from limen.targets.trade_outcome import OutcomeLabels

    context = data.get('_trade_context')
    if not isinstance(context, PreparedTradeContext) or raw is None:
        raise ValueError('Missing training-only trade outcome preparation')
    binding = contract_digest({'partitions': [contract_digest(export_trade_contract(context.policy, partition)) for partition in context.partitions]})
    data['_trade_labels'] = OutcomeLabels(pl.concat([label.rows for label in labels]), binding)
    fitted = cast(Mapping[str, object], data['_fitted_params'])
    shims = cast('FittedColumnShims | None', fitted.get('_column_shims'))
    data['_fitted_params'] = {key: value for key, value in fitted.items() if not key.startswith('_target_cls_')}
    data['_fold_preparation'] = prepare_folds(raw, retained, context.model_rows[0], scale=scale, compress=compress, deterministic=deterministic, shims=shims, dropped=dropped)
