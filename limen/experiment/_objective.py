from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from copy import deepcopy
import inspect
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.backtest.execution_events import SIGNAL_COLUMNS, validate_observations, with_predictions
from limen.backtest.trade_contract import TradeInputs, TradePolicy, finite_number, source_binding
from limen.backtest.trade_execution import trade_execution
from limen.calibration.pipeline import CalibrationConfigProtocol
from limen.experiment._objective_optimizer import check_optimizer
from limen.experiment._prepare_trade_context import PreparedTradeContext
from limen.experiment._manifest_preparation import apply_backtest_cost
from limen.experiment._validation_calibration import ValidationReceipt, resolve_config
from limen.experiment._validation_decisions import ModelManifest, validation_inputs, validation_header, validation_evidence


@dataclass(frozen=True)
class ObjectiveConfig:
    metric: str = 'backtest_total_return'
    direction: str = 'maximize'

    def __post_init__(self) -> None:
        if self.metric != 'backtest_total_return' or self.direction not in ('maximize', 'minimize'):
            raise ValueError('Objective requires backtest_total_return and maximize or minimize')

    @property
    def column(self) -> str:
        return 'val_backtest_total_return'

    @property
    def maximize(self) -> bool:
        return self.direction == 'maximize'

    def as_dict(self) -> dict[str, str]:
        return {'metric': self.metric, 'direction': self.direction}


@dataclass(frozen=True)
class ValidationScorer:
    inputs: TradeInputs
    policy: TradePolicy

    def __call__(self, _labels: object, predictions: object) -> float:
        ledger = trade_execution(with_predictions(self.inputs, predictions), self.policy)
        return finite_number(ledger.metrics['total_return'], 'objective validation return')


def _row_count(value: object) -> int:
    if isinstance(value, (pl.DataFrame, pl.Series)):
        return len(value)
    if isinstance(value, np.ndarray) and value.ndim:
        return cast(npt.NDArray[np.generic], value).shape[0]
    raise ValueError('Objective requires prepared validation features and labels')


def check_calibration(config: CalibrationConfigProtocol | None) -> None:
    if config is not None:
        check_optimizer(config)


def threshold_params(config: CalibrationConfigProtocol, scorer: ValidationScorer,
                     objective: ObjectiveConfig) -> dict[str, object]:
    check_calibration(config)
    return {**config.threshold_params, 'metric': scorer, '_objective_maximize': objective.maximize}


def _validate_funding_coverage(funding: pl.DataFrame, start: int, end: int) -> None:
    cursor = start
    for row in funding.filter(pl.col('kind') == 'accrual').select('start_ns', 'end_ns').sort('start_ns').iter_rows():
        left, right = int(row[0]), int(row[1])
        if right <= left or left > cursor or (left < cursor and cursor != start):
            raise ValueError('Objective funding support has a gap/overlap or invalid interval')
        cursor = right
    if cursor < end:
        raise ValueError('Objective funding support does not cover the partition')


def prepare_objective(data: Mapping[str, object], architecture: object,
                      calibration: CalibrationConfigProtocol | None) -> ValidationScorer:
    del architecture
    check_calibration(calibration)
    context = data.get('_trade_context')
    if not isinstance(context, PreparedTradeContext):
        raise ValueError('Objective requires configured event execution')
    inputs = context.partitions[1]
    count = _row_count(data.get('x_val', data.get('val')))
    if not count or ('y_val' in data and count != _row_count(data['y_val'])) or count != inputs.signals.height:
        raise ValueError('Objective validation rows must be nonempty and aligned')
    if inputs.signals['row_id'].null_count() or inputs.signals['row_id'].is_duplicated().any():
        raise ValueError('Objective validation row identity is ambiguous')
    start, end = inputs.partition_start_ns, inputs.partition_end_ns
    causal_open = ((pl.col('start_ns') != pl.col('end_ns'))
                   & (pl.col('open_available_at_ns') == pl.col('start_ns'))
                   & (pl.col('open_available_at_ns') >= start))
    observations = inputs.observations.filter(
        (pl.col('start_ns') < end) & ((pl.col('available_at_ns') <= end) | causal_open)
    )
    sources = [source_binding(observations, 'objective:validation:execution', start, end, inputs.sources[0].precision_ns, 'causal_recorded_prices'),
               source_binding(inputs.signals.select(SIGNAL_COLUMNS), 'objective:validation:model', start, end, inputs.sources[1].precision_ns, 'retained_model_rows')]
    funding = inputs.funding_events
    if funding is not None:
        funding = funding.filter(
            ((pl.col('kind') == 'accrual') & (pl.col('start_ns') < end) & (pl.col('end_ns') > start))
            | ((pl.col('kind') != 'accrual') & (pl.col('time_ns') >= start) & (pl.col('time_ns') <= end))
        )
        if context.policy.funding is not None and context.policy.funding.mechanism == 'continuous':
            accrual = pl.col('kind') == 'accrual'
            if context.policy.funding.params.get('history_interpretation') == 'integrated':
                if funding.filter(accrual & (pl.col('end_ns') > end)).height:
                    raise ValueError('Objective integrated funding extends into held-out evidence; provide causal quoted support')
            else:
                funding = funding.with_columns(*[
                    pl.when(accrual).then(pl.col(column).clip(start, end)).otherwise(pl.col(column)).alias(column)
                    for column in ('start_ns', 'end_ns', 'time_ns')
                ])
            _validate_funding_coverage(funding, start, end)
        sources.append(source_binding(funding, 'objective:validation:funding', start, end, 1, 'causal_funding_support'))
    selected = replace(inputs, observations=observations, funding_events=funding, sources=tuple(sources))
    validate_observations(selected, context.policy)
    return ValidationScorer(selected, context.policy)


def score_objective(data: Mapping[str, object], result: Mapping[str, object], scorer: ValidationScorer,
                    *, receipt: ValidationReceipt | None = None, inputs: Mapping[str, object] | None = None,
                    header: Mapping[str, object] | None = None) -> float:
    inputs = validation_inputs(data, scorer) if inputs is None else inputs
    header = validation_header(data, scorer, inputs) if header is None else header
    predictions, probabilities, payload = validation_evidence(inputs, result, scorer, header)
    if receipt is not None:
        receipt.verify(predictions, probabilities, payload)
    return scorer(None, predictions)


def run_model(manifest: ModelManifest, data: dict[str, object], round_params: dict[str, object]) -> dict[str, object]:
    from limen.experiment._allocation import run as run_allocation
    from limen.experiment._prepare_trade_context import finish_trade_result, resolve_component_kwargs

    function = manifest.architecture_function
    if function is None:
        raise ValueError('Manifest run_model requires a configured architecture_function')
    allocation = manifest.allocation_config is not None
    objective = getattr(manifest, 'objective', None)
    if objective is not None and not isinstance(objective, ObjectiveConfig):
        raise TypeError('Manifest objective must be an ObjectiveConfig or None')
    kwargs = {} if allocation else manifest.resolve_model_kwargs(round_params)
    config = None if allocation else getattr(manifest, 'prediction_calibration_config', None)
    if objective is not None and config is None:
        config = kwargs.get('prediction_calibration_config')
    if config is not None:
        config = resolve_config(config, round_params)
        accepts = any(name == 'prediction_calibration_config' or parameter.kind == inspect.Parameter.VAR_KEYWORD for name, parameter in inspect.signature(function).parameters.items())
        if config is not None and (objective is None or config.calibration_func is not None or config.threshold_func is not None) and not accepts:
            raise ValueError('MLManifest Calibration is configured but the architecture function does not accept `prediction_calibration_config`')
        if accepts and (config is not None or objective is not None):
            kwargs['prediction_calibration_config'] = config
        elif objective is not None:
            _ = kwargs.pop('prediction_calibration_config', None)
    apply_backtest_cost(manifest, data, round_params)
    if hasattr(manifest, 'prediction_calibration_config') and not allocation:
        resolve_component_kwargs(function, data, kwargs, round_params)
    scorer = None if objective is None else prepare_objective(data, function, config)
    inputs = None if scorer is None else validation_inputs(data, scorer)
    header = None if scorer is None else validation_header(data, scorer, cast(Mapping[str, object], inputs))
    receipt = None if scorer is None or config is None or objective is None or (config.calibration_func is None and config.threshold_func is None) else ValidationReceipt(config, scorer, objective.maximize, validation_inputs=data['x_val'], validation_labels=data['y_val'])
    if header is not None:
        data['_validation_context'] = deepcopy(header)
    if receipt is not None:
        kwargs['prediction_calibration_config'] = receipt.resolved()
    result: dict[str, object] = {}
    try:
        result = run_allocation(manifest, data, round_params) if allocation else function(data, **kwargs)
        if scorer is not None and objective is not None:
            result[objective.column] = score_objective(data, result, scorer, receipt=receipt, inputs=inputs, header=header)
        _ = result.pop('_validation', None)
        return result if allocation else finish_trade_result(data, result)
    finally:
        _ = data.pop('_validation_context', None)
        if receipt is not None:
            receipt.finish(result.get('_model'))


__all__ = ['ObjectiveConfig', 'ValidationScorer', 'prepare_objective', 'score_objective', 'threshold_params']
