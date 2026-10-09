from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib import import_module

import numpy as np
import polars as pl

from limen.backtest.execution_events import validate_observations, with_predictions
from limen.backtest.funding_adapter import _coverage
from limen.backtest.trade_contract import TradeInputs, TradePolicy, finite_number, source_binding
from limen.backtest.trade_execution import trade_execution
from limen.calibration._objective_threshold import validate_probabilities
from limen.calibration.pipeline import CalibrationConfigProtocol
from limen.calibration.threshold import grid_threshold_optimizer
from limen.experiment._prepare_trade_context import PreparedTradeContext
from limen.sfd.reference_architecture.base import ReferenceModel


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

    def __call__(self, labels: object, predictions: object) -> float:
        ledger = trade_execution(with_predictions(self.inputs, predictions), self.policy)
        return finite_number(ledger.metrics['total_return'], 'objective validation return')


def _row_count(value: object) -> int:
    if isinstance(value, (pl.DataFrame, pl.Series)):
        return len(value)
    if isinstance(value, np.ndarray) and value.ndim:
        return len(value)
    raise ValueError('Objective requires prepared validation features and labels')


def check_calibration(config: CalibrationConfigProtocol | None) -> None:
    if config is not None:
        if config.threshold_func is not None and config.threshold_func is not grid_threshold_optimizer:
            raise ValueError('Objective supports only grid_threshold_optimizer')
        if 'metric' in config.threshold_params or '_objective_maximize' in config.threshold_params:
            raise ValueError('Objective conflicts with an explicit threshold metric/direction')


def threshold_params(config: CalibrationConfigProtocol, scorer: ValidationScorer,
                     objective: ObjectiveConfig) -> dict[str, object]:
    check_calibration(config)
    return {**config.threshold_params, 'metric': scorer, '_objective_maximize': objective.maximize}


def prepare_objective(data: Mapping[str, object], architecture: object,
                      calibration: CalibrationConfigProtocol | None) -> ValidationScorer:
    name = getattr(architecture, '__name__', '')
    if name not in ('logreg_binary', 'lightgbm_binary', 'tabpfn_binary') or architecture is not getattr(import_module(f'limen.sfd.reference_architecture.{name}'), name):
        raise ValueError('Objective requires a supported built-in binary ML architecture')
    check_calibration(calibration)
    context = data.get('_trade_context')
    if not isinstance(context, PreparedTradeContext) or context.policy.prediction_mode != 'binary':
        raise ValueError('Objective requires configured binary event execution')
    inputs = context.partitions[1]
    count = _row_count(data.get('x_val'))
    if not count or count != _row_count(data.get('y_val')) or count != inputs.signals.height:
        raise ValueError('Objective validation rows must be nonempty and aligned')
    if inputs.signals['row_id'].null_count() or inputs.signals['row_id'].is_duplicated().any():
        raise ValueError('Objective validation row identity is ambiguous')
    start, end = inputs.partition_start_ns, inputs.partition_end_ns
    observations = inputs.observations.filter((pl.col('start_ns') < end) & (pl.col('available_at_ns') <= end))
    sources = [source_binding(observations, 'objective:validation:execution', start, end, 1, 'causal_recorded_prices'),
               source_binding(inputs.signals, 'objective:validation:model', start, end, 1, 'retained_model_rows')]
    funding = inputs.funding_events
    if funding is not None:
        funding = funding.filter(
            ((pl.col('kind') == 'accrual') & (pl.col('start_ns') < end) & (pl.col('end_ns') > start) & (pl.col('end_ns') <= end))
            | ((pl.col('kind') != 'accrual') & (pl.col('time_ns') >= start) & (pl.col('time_ns') <= end))
        )
        if context.policy.funding is not None and context.policy.funding.mechanism == 'continuous':
            _coverage([(int(left), int(right)) for left, right in funding.filter(pl.col('kind') == 'accrual').select('start_ns', 'end_ns').iter_rows()], start, end)
        sources.append(source_binding(funding, 'objective:validation:funding', start, end, 1, 'causal_funding_support'))
    selected = replace(inputs, observations=observations, funding_events=funding, sources=tuple(sources))
    validate_observations(selected, context.policy)
    return ValidationScorer(selected, context.policy)


def score_objective(data: Mapping[str, object], result: Mapping[str, object],
                    scorer: ValidationScorer) -> float:
    model = result.get('_model')
    if not isinstance(model, ReferenceModel):
        raise ValueError('Objective requires its fitted reference model')
    prediction = model.predict({'x_test': data['x_val']})
    probabilities = validate_probabilities(prediction.get('_probs'))
    if probabilities.size != scorer.inputs.signals.height:
        raise ValueError('Objective probabilities do not match validation rows')
    return scorer(data.get('y_val'), prediction.get('_preds'))


__all__ = ['ObjectiveConfig', 'ValidationScorer', 'prepare_objective', 'score_objective', 'threshold_params']
