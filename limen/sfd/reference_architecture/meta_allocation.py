from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import ClassVar, Literal, Protocol, cast
from importlib import import_module

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from limen.backtest.trade_contract import JsonValue, finite_number, json_value
from limen.experiment._allocation import AllocationBatch, AllocationContext, OpportunityLabels, component_identity, primary_parameters
from limen.experiment.errors import StrictModeError
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.sfd.reference_architecture.direction_sizing import Array, ComponentEstimator, ComponentFactory
_components = vars(import_module('limen.sfd.reference_architecture.direction_sizing'))
DIRECTION_FACTORY = cast(ComponentFactory, _components['_DIRECTION_FACTORY'])
SIZING_FACTORY = cast(ComponentFactory, _components['_SIZING_FACTORY'])
component_predict = cast(Callable[[ComponentEstimator, Array], Array], _components['_predict'])


_MAX_INTERNAL_VALIDATION_FRACTION = 0.5


class _Scaler(Protocol):
    def fit(self, x: Array) -> _Scaler: ...
    def transform(self, x: Array) -> Array: ...


class _ProbabilityEstimator(ComponentEstimator, Protocol):
    classes_: Array
    def predict_proba(self, x: Array) -> Array: ...


class MetaAllocationModel(ReferenceModel):
    prediction_mode: ClassVar[Literal['binary', 'target_exposure']] = 'target_exposure'

    @override
    def train(self, data: dict[str, object], **params: object) -> MetaAllocationModel:
        context = params.pop('context')
        if not isinstance(context, AllocationContext):
            raise ValueError('Meta allocation requires its original prepared primary context')
        self.context = context
        allowed = {'folds', 'internal_validation_fraction', 'min_train_samples', 'features', 'include_primary_proba', 'probability_factory', 'sizing_factory', 'probability_params', 'sizing_params', 'probability_threshold', 'return_scale', 'max_size', 'risk_budget_fraction', 'seed'}
        if set(params) - allowed:
            raise ValueError(f'Unknown meta allocation params: {sorted(set(params) - allowed)}')
        self.threshold = finite_number(params.get('probability_threshold', 0.5), 'take probability threshold')
        self.return_scale = finite_number(params.get('return_scale', 0.01), 'return scale')
        self.max_size = finite_number(params.get('max_size', 1), 'maximum size')
        risk = params.get('risk_budget_fraction')
        self.risk = None if risk is None else finite_number(risk, 'risk budget fraction')
        fraction = finite_number(params.get('internal_validation_fraction', 0.2), 'internal validation fraction')
        seed, folds, minimum = params.get('seed', 42), params.get('folds', 5), params.get('min_train_samples', 20)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (seed, folds, minimum)) or cast(int, folds) <= 1 or cast(int, minimum) <= 1:
            raise ValueError('Seed/folds/minimum require integers; folds/minimum at least two')
        if not 0 < self.threshold < 1 or self.return_scale <= 0 or not 0 < self.max_size <= 1 or not 0 < fraction < _MAX_INTERNAL_VALIDATION_FRACTION or (self.risk is not None and not 0 < self.risk <= 1):
            raise ValueError('Invalid meta allocation probability, scale, size, validation or risk range')
        if self.risk is not None and context.policy.stop_loss_bps is None and context.policy.stop_loss_bps_column is None:
            raise ValueError('Risk budget requires a declared stop loss')
        include = params.get('include_primary_proba', False)
        if not isinstance(include, bool):
            raise ValueError('include_primary_proba must be boolean')
        self.include_primary_proba = include
        from limen.experiment._allocation_folds import fit_primary, training_opportunities
        labels = training_opportunities(context, folds=cast(int, folds), validation_fraction=fraction, include_primary_proba=include)
        self.primary = fit_primary(context.manifest, data, context.round_params)
        self._fit(labels, params, cast(int, minimum), cast(int, seed))
        self.batch_digests = {name: batch.digest() for name, batch in context.batches.items()}
        self.batch_outputs = {name: self._compose(batch) for name, batch in context.batches.items()}
        self.learning_binding = {'architecture': component_identity(meta_allocation), 'primary': component_identity(vars(context.manifest)['architecture_function']),
                                 'primary_params': json_value(primary_parameters(context.manifest, context.round_params)), 'training': json_value(labels.binding),
                                 'probability_factory': component_identity(self.probability_factory), 'sizing_factory': component_identity(self.sizing_factory),
                                 'allocation_params': json_value({key: value for key, value in params.items() if key not in ('probability_factory', 'sizing_factory')}),
                                 'features': list(self.feature_names), 'batch_digests': self.batch_digests,
                                 'mapping_version': 'primary_opportunity_signed_net_notional_v1'}
        self.deterministic = bool(self.primary.deterministic and self.probability_factory.deterministic and self.sizing_factory.deterministic)
        data['_learning_binding'] = self.learning_binding
        alignment = cast(dict[str, object], data['_alignment'])
        alignment['learning_binding'] = self.learning_binding
        self.training_rows = labels.rows
        return self

    def _fit(self, labels: OpportunityLabels, params: Mapping[str, object], minimum: int, seed: int) -> None:
        available = labels.rows['available'].to_numpy()
        returns = labels.rows['net_return_on_entry_notional'].to_numpy().astype(np.float64)
        y = (returns > 0).astype(np.float64)
        if np.count_nonzero(available) < minimum or np.unique(y[available]).size <= 1:
            raise StrictModeError(f'Insufficient completed OOF opportunities/classes: available={np.count_nonzero(available)} required={minimum}')
        selected = cast(Sequence[str] | None, params.get('features'))
        names = list(labels.features.columns if selected is None else selected)
        if self.include_primary_proba and 'primary_probability' not in names:
            names.append('primary_probability')
        if not names or len(set(names)) != len(names) or set(names) - set(labels.features.columns):
            raise ValueError('Meta features must be unique existing numeric primary feature names')
        self.feature_names = tuple(names)
        x = labels.features.select(names).to_numpy().astype(np.float64)[available]
        if not np.isfinite(x).all():
            raise ValueError('Meta opportunity features must be finite')
        self.scaler = cast(_Scaler, StandardScaler()).fit(x)
        x = self.scaler.transform(x)
        self.probability_factory = cast(ComponentFactory, params.get('probability_factory', DIRECTION_FACTORY))
        self.sizing_factory = cast(ComponentFactory, params.get('sizing_factory', SIZING_FACTORY))
        self.probability_model = cast(_ProbabilityEstimator, self.probability_factory(seed=seed, **cast(Mapping[str, JsonValue], params.get('probability_params') or {'C': 1, 'max_iter': 1000})))
        self.sizing_model = self.sizing_factory(seed=seed, **cast(Mapping[str, JsonValue], params.get('sizing_params') or {'alpha': 1}))
        _ = self.probability_model.fit(x, y[available])
        _ = self.sizing_model.fit(x, returns[available])
        if not np.array_equal(self.probability_model.classes_, [0, 1]):
            raise ValueError('Probability model must supply classes 0 and 1')

    def _allocation(self, probability: float, gain: float, stop: float | None) -> float:
        if not np.isfinite(probability) or not 0 <= probability <= 1 or not np.isfinite(gain):
            raise ValueError('Meta predictions must be finite probabilities and signed returns')
        risk_cap = np.inf
        policy = self.context.policy
        if self.risk is not None:
            if stop is None or not np.isfinite(stop) or stop <= 0:
                raise ValueError('Risk cap requires a positive decision-time stop')
            risk_cap = self.risk / ((stop + 2 * policy.fee_bps + 2 * policy.slip_bps) / 10000)
        exposure = min(self.max_size, policy.max_exposure, max(0, gain / self.return_scale), risk_cap) if probability >= self.threshold else 0.0
        return 0.0 if exposure <= policy.flat_threshold else float(exposure)

    @override
    def predict(self, data: dict[str, object]) -> dict[str, object]:
        batch = data.get('allocation_batch')
        if not isinstance(batch, AllocationBatch) or self.batch_digests.get(batch.batch_id) != batch.digest():
            raise ValueError('Allocation prediction requires its complete original bound batch')
        return deepcopy(self.batch_outputs[batch.batch_id])

    def _compose(self, batch: AllocationBatch) -> dict[str, object]:
        primary = self.primary.predict(dict(batch.primary_inputs))
        decisions = np.asarray(primary['_preds'], dtype=np.float64)
        if decisions.shape != (len(batch.row_ids),) or not np.isin(decisions, [0, 1]).all():
            raise ValueError('Primary predictions must be aligned binary long/flat decisions')
        features = batch.meta_features
        if self.include_primary_proba:
            import polars as pl
            probability = np.asarray(primary.get('_probs'), dtype=np.float64)
            if probability.shape != decisions.shape or not np.isfinite(probability).all():
                raise ValueError('Original primary probabilities are unavailable or malformed')
            features = features.with_columns(pl.Series('primary_probability', probability))
        x = features.select(self.feature_names).to_numpy().astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Meta inference features must be finite')
        x = self.scaler.transform(x)
        probabilities = np.asarray(self.probability_model.predict_proba(x), dtype=np.float64)
        if probabilities.shape != (len(decisions), 2) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError('Malformed take probabilities')
        gains = component_predict(self.sizing_model, x)
        exposure = np.zeros(len(decisions), dtype=np.float64)
        held, previous = 0.0, 0.0
        for i, decision in enumerate(decisions):
            if decision == 0:
                held = 0.0
            elif previous == 0:
                if not batch.barrier_ready[i]:
                    raise StrictModeError(f'Barrier warmup at primary entry row={batch.row_ids[i]}')
                held = self._allocation(float(probabilities[i, 1]), float(gains[i]), batch.stop_loss_bps[i])
            exposure[i], previous = held, decision
        return {'_preds': exposure, 'primary': primary, 'take_probability': probabilities[:, 1], 'signed_notional_return': gains}

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        output = self.predict({'allocation_batch': self.context.batches['test']})
        result: dict[str, object] = {'_preds': output['_preds'], '_prediction_mode': self.prediction_mode,
                                    'allocation_training_completed': int(self.training_rows['available'].sum()),
                                    'allocation_training_unavailable': int((~self.training_rows['available']).sum())}
        alignment = cast(dict[str, object], data['_alignment'])
        _ = alignment.pop('model_outputs', None)
        if data.get('_record_model_outputs'):
            alignment['model_outputs'] = {'architecture': 'meta_allocation', 'primary': json_value({key: np.asarray(value).tolist() for key, value in cast(Mapping[str, object], output['primary']).items()}),
                                          'take_probability': json_value(np.asarray(output['take_probability']).tolist()),
                                          'signed_notional_return': json_value(np.asarray(output['signed_notional_return']).tolist()),
                                          'allocated_exposure': json_value(np.asarray(output['_preds']).tolist())}
        if inline_metrics:
            result.update(self._compute_backtest(cast(Array, result['_preds']), data))
        ledger = alignment.get('trade_ledger')
        if isinstance(ledger, dict):
            ledger['allocation_training'] = json_value(self.training_rows.to_dicts())
        return result


def meta_allocation(data: dict[str, object], *, context: AllocationContext, folds: int = 5, internal_validation_fraction: float = 0.2, min_train_samples: int = 20, features: Sequence[str] | None = None, include_primary_proba: bool = False, probability_factory: ComponentFactory = DIRECTION_FACTORY, sizing_factory: ComponentFactory = SIZING_FACTORY, probability_params: Mapping[str, JsonValue] | None = None, sizing_params: Mapping[str, JsonValue] | None = None, probability_threshold: float = 0.5, return_scale: float = 0.01, max_size: float = 1.0, risk_budget_fraction: float | None = None, seed: int = 42) -> dict[str, object]:
    params = dict(locals())
    del params['data']
    model = MetaAllocationModel().train(data, **params)
    return {**model.evaluate(data), '_model': model}


__all__ = ['MetaAllocationModel', 'meta_allocation']
