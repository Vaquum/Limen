from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import polars as pl

from limen.backtest.execution_events import OBSERVATION_COLUMNS, validate_observations
from limen.backtest.funding_adapter import prepare_funding, resolve_adapter
from limen.backtest.trade_contract import NANOSECONDS, RULE_VERSION, JsonValue, TradeInputs, TradeLedger, TradePolicy, contract_digest, export_trade_contract, json_value, source_binding
from limen.experiment._resolve_trade_policy import BacktestConfig, SourceConfig, resolve_json, resolve_number

if TYPE_CHECKING:
    from limen.targets.trade_outcome import TradeTargetContext, OutcomeLabels


_METADATA = ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns')


@dataclass(frozen=True)
class PreparedTradeContext:
    policy: TradePolicy
    partitions: tuple[TradeInputs, ...]
    model_rows: tuple[pl.DataFrame, ...]
    source_settings: tuple[tuple[Callable[..., object], JsonValue] | None, ...]


def _source_settings(config: BacktestConfig, params: Mapping[str, object]) -> tuple[tuple[Callable[..., object], JsonValue] | None, ...]:
    sources = (config.execution_data_source, config.funding.data_source if config.funding is not None else None)
    return tuple(None if source is None else (source.method, resolve_json(source.params, params)) for source in sources)


def _load(source: SourceConfig, params: Mapping[str, object]) -> pl.DataFrame:
    values = cast(dict[str, object], resolve_json(source.params, params))
    result = source.method(**values)
    if not isinstance(result, pl.DataFrame):
        raise ValueError('Trade source must return a Polars DataFrame')
    return result


def normalize_observations(data: pl.DataFrame, *, interval_seconds: object = None) -> pl.DataFrame:
    if set(OBSERVATION_COLUMNS) <= set(data.columns):
        return data.select(OBSERVATION_COLUMNS)
    if not {'datetime', 'open', 'high', 'low', 'close'} <= set(data.columns):
        raise ValueError('Recorded execution data requires OHLC, datetime and causal interval metadata')
    if not isinstance(data['datetime'].dtype, pl.Datetime):
        raise ValueError('Model-bar datetime must declare UTC datetime units')
    starts = data['datetime'].dt.epoch('ns')
    if set(_METADATA) <= set(data.columns):
        timing = data.select(_METADATA)
        if not timing['start_ns'].equals(starts.rename('start_ns')):
            raise ValueError('Execution/model source start timestamps disagree')
    else:
        if interval_seconds is None and 'base_interval' in data.columns and data['base_interval'].n_unique() == 1:
            interval_seconds = data['base_interval'][0]
        interval = round(resolve_number(interval_seconds, {}, 'recorded regular source interval') * NANOSECONDS)
        if interval <= 0 or starts.diff().drop_nulls().ne(interval).any():
            raise ValueError('Missing interval metadata or source interval is not verified regular')
        timing = pl.DataFrame({'start_ns': starts, 'end_ns': starts + interval, 'open_available_at_ns': starts, 'available_at_ns': starts + interval})
    return timing.with_columns(pl.Series('row_id', [f'row:{index}' for index in range(data.height)]), *[data[key] for key in ('open', 'high', 'low', 'close')]).select(OBSERVATION_COLUMNS)


def prepare_trade_context(config: BacktestConfig, policy: TradePolicy, raw_splits: list[pl.DataFrame], bars: list[pl.DataFrame], params: Mapping[str, object], *, interval_seconds: object = None) -> PreparedTradeContext:
    interval_seconds = None if interval_seconds is None else resolve_number(interval_seconds, params, 'recorded source interval')
    execution = None
    if config.execution_data_source is not None:
        declared_interval = config.execution_data_source.params.get('klines_size')
        execution_interval = None if declared_interval is None else resolve_number(declared_interval, params, 'execution source interval')
        execution = normalize_observations(_load(config.execution_data_source, params), interval_seconds=execution_interval)
    history = _load(config.funding.data_source, params) if config.funding is not None and config.funding.data_source is not None else None
    initial = resolve_number(config.initial_equity, params, 'initial equity')
    inputs: list[TradeInputs] = []
    model_rows: list[pl.DataFrame] = []
    for index, (raw, model) in enumerate(zip(raw_splits, bars, strict=True)):
        raw_prices = normalize_observations(raw, interval_seconds=interval_seconds)
        if not raw_prices.height or not model.height:
            raise ValueError('Event execution requires nonempty declared partitions')
        starts = raw['datetime'].dt.epoch('ns').to_list()
        counts = model['bar_count'].to_list() if 'bar_count' in model.columns else [1] * model.height
        positions = {int(value): position for position, value in enumerate(starts)}
        rows: list[dict[str, object]] = []
        for position, (time, count) in enumerate(zip(model['datetime'].dt.epoch('ns'), counts, strict=True)):
            first = positions[int(time)]
            last = first + int(count) - 1
            if last >= raw_prices.height:
                raise ValueError('Bar membership exceeds its true source partition')
            rows.append({'datetime': model['datetime'][position], 'row_id': f'partition:{index}:bar:{position}', 'available_at_ns': int(raw_prices['available_at_ns'][last])})
        mapping = pl.DataFrame(rows).with_columns(model['datetime'])
        if mapping['datetime'].is_duplicated().any():
            raise ValueError('Trade model rows require unambiguous source identity')
        start, end = int(raw_prices['start_ns'][0]), int(raw_prices['available_at_ns'][-1])
        observations = raw_prices if execution is None else execution.filter((pl.col('start_ns') <= end) & (pl.col('end_ns') >= start))
        funding = prepare_funding(policy.funding, history, start, end) if policy.funding is not None else None
        sources = [source_binding(observations, f'execution:partition:{index}', start, end, _precision(observations), 'causal_recorded_prices')]
        sources.append(source_binding(mapping, f'model:partition:{index}', start, end, _precision(raw_prices), 'model_bar_membership_and_availability'))
        if funding is not None:
            sources.append(source_binding(funding, f'funding:partition:{index}', start, end, 1, 'normalized_funding_support'))
        signals = mapping.select('row_id', 'available_at_ns').with_columns(pl.lit(None, dtype=pl.Float64).alias('target'))
        partition = TradeInputs(initial, start, end, signals, observations, funding, tuple(sources))
        validate_observations(partition, policy)
        inputs.append(partition)
        model_rows.append(mapping)
    return PreparedTradeContext(policy, tuple(inputs), tuple(model_rows), _source_settings(config, params))


def _precision(observations: pl.DataFrame) -> int:
    differences = observations['available_at_ns'].diff().drop_nulls()
    positive = differences.filter(differences > 0)
    return int(cast(int, positive.min())) if positive.len() else 1


def validate_cached_context(config: BacktestConfig | None, policy: TradePolicy | None, data: Mapping[str, object], params: Mapping[str, object]) -> None:
    inputs = data.get('_trade_inputs')
    context = data.get('_trade_context')
    equity_matches = policy is None or (config is not None and isinstance(inputs, TradeInputs) and inputs.initial_equity == resolve_number(config.initial_equity, params, 'initial equity'))
    sources_match = policy is None or (config is not None and isinstance(context, PreparedTradeContext) and context.source_settings == _source_settings(config, params))
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
    signals = identities.select('row_id', 'available_at_ns').with_columns(pl.lit(None, dtype=pl.Float64).alias('target'))
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
    signals = context.model_rows[index].with_columns(pl.lit(None, dtype=pl.Float64).alias('target'))
    inputs = replace(original, signals=signals)
    return TradeTargetContext(context.policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(context.policy, inputs)))


@dataclass(frozen=True)
class FoldFeatures:
    train: pl.DataFrame
    predict: pl.DataFrame


@runtime_checkable
class FoldPreparation(Protocol):
    raw_features: pl.DataFrame
    row_ids: tuple[str, ...]
    deterministic: bool
    def fit_transform(self, train_rows: Sequence[str], predict_rows: Sequence[str]) -> FoldFeatures: ...


@dataclass(frozen=True)
class PreparedFolds:
    raw_features: pl.DataFrame
    row_ids: tuple[str, ...]
    deterministic: bool
    transform: Callable[[Sequence[str], Sequence[str]], tuple[pl.DataFrame, pl.DataFrame]]

    def fit_transform(self, train_rows: Sequence[str], predict_rows: Sequence[str]) -> FoldFeatures:
        train, predict = self.transform(train_rows, predict_rows)
        if train.height != len(train_rows) or predict.height != len(predict_rows):
            raise ValueError('Fold preprocessing changed causal feature-valid membership')
        return FoldFeatures(train, predict)


def sensor_decisions(raw: pl.DataFrame, bars: pl.DataFrame, *, interval_seconds: object = None) -> pl.DataFrame:
    prices = normalize_observations(raw, interval_seconds=interval_seconds)
    if (prices['available_at_ns'] < prices['end_ns']).any():
        raise ValueError('Sensor decisions require source availability at or after interval end')
    positions = {int(value): index for index, value in enumerate(raw['datetime'].dt.epoch('ns'))}
    counts = bars['bar_count'].to_list() if 'bar_count' in bars.columns else [1] * bars.height
    available: list[int] = []
    for time, count in zip(bars['datetime'].dt.epoch('ns'), counts, strict=True):
        last = positions[int(time)] + int(count) - 1
        if last >= prices.height:
            raise ValueError('Sensor model-bar membership exceeds its recorded source')
        available.append(int(prices['available_at_ns'][last]))
    return bars.select('datetime').with_columns(pl.Series('__trade_available_at_ns__', available, dtype=pl.Int64))


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


__all__ = ['PreparedTradeContext', 'attach_trade_context', 'finish_trade_result', 'normalize_observations', 'persist_ledger', 'prepare_trade_context', 'select_trade_rows']

class FoldScaler(Protocol):
    def __call__(self, data: pl.DataFrame, *, all_fitted_params: dict[str, object], is_training: bool) -> tuple[pl.DataFrame, dict[str, object]]: ...


class FoldCompression(Protocol):
    def __call__(self, split_data: list[pl.DataFrame], *, all_fitted_params: dict[str, object]) -> tuple[list[pl.DataFrame], dict[str, object]]: ...


def prepare_folds(raw: pl.DataFrame, retained: pl.DataFrame, identities: pl.DataFrame, *, scale: FoldScaler, compress: FoldCompression, deterministic: bool) -> PreparedFolds:
    selected = retained.select('datetime').join(raw, on='datetime', how='left', maintain_order='left')
    mapping = retained.select('datetime').join(identities, on='datetime', how='left', maintain_order='left')
    row_ids = tuple(str(value) for value in mapping['row_id'])
    positions = {identity: index for index, identity in enumerate(row_ids)}

    def transform(train_rows: Sequence[str], predict_rows: Sequence[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
        train = selected[[positions[row] for row in train_rows]]
        predict = selected[[positions[row] for row in predict_rows]]
        fitted: dict[str, object] = {}
        train, fitted = scale(train, all_fitted_params=fitted, is_training=True)
        context_rows = max((int(getattr(value, 'context_rows', 0)) for value in fitted.values()), default=0)
        prefix = selected[[positions[row] for row in train_rows]].tail(context_rows)
        predict, fitted = scale(pl.concat([prefix, predict]), all_fitted_params=fitted, is_training=False)
        predict = predict.slice(prefix.height)
        transformed, _ = compress([train, predict, predict], all_fitted_params=fitted)
        features = [name for name in transformed[0].columns if name != 'datetime']
        if any(transformed[index].select(features).null_count().sum_horizontal()[0] for index in (0, 1)):
            raise ValueError('Fold preprocessing has unavailable features; provide sufficient causal context')
        return transformed[0].select(features), transformed[1].select(features)

    return PreparedFolds(selected.drop('datetime'), row_ids, deterministic, transform)


def attach_outcomes(data: dict[str, object], labels: Sequence[OutcomeLabels], raw: pl.DataFrame | None, retained: pl.DataFrame, *, scale: FoldScaler, compress: FoldCompression, deterministic: bool) -> None:
    from limen.targets.trade_outcome import OutcomeLabels

    context = data.get('_trade_context')
    if not isinstance(context, PreparedTradeContext) or raw is None:
        raise ValueError('Missing training-only trade outcome preparation')
    binding = contract_digest({'partitions': [contract_digest(export_trade_contract(context.policy, partition)) for partition in context.partitions]})
    data['_trade_labels'] = OutcomeLabels(pl.concat([label.rows for label in labels]), binding)
    fitted = cast(Mapping[str, object], data['_fitted_params'])
    data['_fitted_params'] = {key: value for key, value in fitted.items() if not key.startswith('_target_cls_')}
    data['_fold_preparation'] = prepare_folds(raw, retained, context.model_rows[0], scale=scale, compress=compress, deterministic=deterministic)
