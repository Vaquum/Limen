from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import polars as pl

from limen.backtest.execution_events import OBSERVATION_COLUMNS, validate_observations
from limen.backtest.funding_adapter import prepare_funding
from limen.backtest.trade_contract import NANOSECONDS, TradeInputs, TradeLedger, TradePolicy, contract_digest, export_trade_contract, json_value, source_binding
from limen.experiment._resolve_trade_policy import BacktestConfig, SourceConfig, resolve_json, resolve_number

if TYPE_CHECKING:
    from limen.sfd.reference_architecture.direction_sizing import FoldFeatures
    from limen.targets.trade_outcome import TradeTargetContext


_METADATA = ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns')


@dataclass(frozen=True)
class PreparedTradeContext:
    policy: TradePolicy
    partitions: tuple[TradeInputs, ...]
    model_rows: tuple[pl.DataFrame, ...]


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
    execution = _load(config.execution_data_source, params) if config.execution_data_source is not None else None
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
        observations = raw_prices if execution is None else normalize_observations(execution).filter(pl.col('start_ns').is_between(start, end) & (pl.col('available_at_ns') <= end))
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
    return PreparedTradeContext(policy, tuple(inputs), tuple(model_rows))


def _precision(observations: pl.DataFrame) -> int:
    differences = observations['available_at_ns'].diff().drop_nulls()
    positive = differences.filter(differences > 0)
    return int(cast(int, positive.min())) if positive.len() else 1


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
class PreparedFolds:
    raw_features: pl.DataFrame
    row_ids: tuple[str, ...]
    deterministic: bool
    transform: Callable[[Sequence[str], Sequence[str]], tuple[pl.DataFrame, pl.DataFrame]]

    def fit_transform(self, train_rows: Sequence[str], predict_rows: Sequence[str]) -> FoldFeatures:
        from limen.sfd.reference_architecture.direction_sizing import FoldFeatures

        train, predict = self.transform(train_rows, predict_rows)
        if train.height != len(train_rows) or predict.height != len(predict_rows):
            raise ValueError('Fold preprocessing changed causal feature-valid membership')
        return FoldFeatures(train, predict)


def sensor_decisions(raw: pl.DataFrame, bars: pl.DataFrame, *, interval_seconds: object = None) -> pl.DataFrame:
    prices = normalize_observations(raw, interval_seconds=interval_seconds)
    positions = {int(value): index for index, value in enumerate(raw['datetime'].dt.epoch('ns'))}
    counts = bars['bar_count'].to_list() if 'bar_count' in bars.columns else [1] * bars.height
    available: list[int] = []
    for time, count in zip(bars['datetime'].dt.epoch('ns'), counts, strict=True):
        last = positions[int(time)] + int(count) - 1
        if last >= prices.height:
            raise ValueError('Sensor model-bar membership exceeds its recorded source')
        available.append(int(prices['available_at_ns'][last]))
    return bars.select('datetime').with_columns(pl.Series('__trade_available_at_ns__', available, dtype=pl.Int64))


__all__ = ['PreparedTradeContext', 'attach_trade_context', 'finish_trade_result', 'normalize_observations', 'persist_ledger', 'prepare_trade_context', 'select_trade_rows']
