from __future__ import annotations
from limen.experiment._boolean_controls import manifest_controls
from limen.experiment._validation_calibration import resolve_config

from collections.abc import Callable, Iterable, Mapping
import copy
from dataclasses import fields
from datetime import date
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import math
import numbers
import pickle
import platform
from pathlib import Path
from threading import Lock
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.utils import validation as _validation

from limen.cohort import Cohort
from limen.experiment import _holdout_events as events
from limen.experiment import _holdout_history as history
from limen.experiment._holdout_source import source_spec
from limen.experiment._holdout_report import digest, normalized
from limen.experiment._column_shims import FittedColumnShims
from limen.experiment.manifest_core import MLManifest
from limen.inference.sensor import Sensor, contract_digest
from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
from limen.sfd.reference_architecture.logreg_binary import LogRegBinary
from limen.targets.next_bar_down import NextBarDownTarget
from limen.targets.next_bar_up import NextBarUpTarget
from limen.targets.threshold_binary import ThresholdBinaryTarget
import limen.targets as _targets
from limen.scalers import registry as _scalers
from limen.yaml.compiler import CompiledSFD

check_is_fitted = cast(Callable[[object], None], vars(_validation)['check_is_fitted'])
_TOKEN = object()
_VOTE_THRESHOLD = 0.5
_FITTED_TYPES = (*cast(dict[str, type[object]], vars(_scalers)['SCALER_REGISTRY']).values(), PCA, FittedColumnShims,
                 *(value for value in vars(_targets).values() if isinstance(value, type) and value.__module__.startswith('limen.targets.')))
_IMPLEMENTATIONS: dict[type[object], dict[str, object]] = {cls: {name: getattr(cls, name) for name in names} for cls, names in (
    (Sensor, ('predict', '_get_manifest', '_last_bar_inside_training_window')),
    (Cohort, ('_aggregate_bar_predictions', '_fallback_vote_threshold', '_validate_probability_range')),
    (LogRegBinary, ('predict',)), (LightGBMBinary, ('predict',)),
    (LogisticRegression, ('predict', 'predict_proba')),
    (CalibratedClassifierCV, ('predict_proba',)),
    (FrozenEstimator, ('__getattr__',)),
    (MLManifest, ('sensor_input_prep', '_prepare_column_shims')),
    (NextBarUpTarget, ('transform',)), (NextBarDownTarget, ('transform',)))}
_IMPLEMENTATIONS.update({cls: {name: getattr(cls, name) for name in ('transform',) if hasattr(cls, name)} for cls in _FITTED_TYPES})


def unchanged(value: object) -> None:
    for name, original in _IMPLEMENTATIONS.get(type(value), {}).items():
        if name in vars(value) or getattr(type(value), name) is not original:
            raise ValueError(f'Holdout refuses replaced {type(value).__name__}.{name}')


def scenario_declarations(values: list[Mapping[str, object]], *, allow_empty: bool = False) -> list[dict[str, object]]:
    keys = {'fee_bps', 'slip_bps', 'notional_rate', 'execution_lag_bars', 'take_profit_bps', 'stop_loss_bps'}
    if not isinstance(cast(object, values), list) or (not values and not allow_empty):
        raise ValueError('Holdout requires a nonempty scenario list')
    result: list[dict[str, object]] = []
    for value in values:
        if not isinstance(cast(object, value), Mapping) or set(value) != keys:
            raise ValueError('Holdout scenario requires exactly all six declared keys')
        row: dict[str, object] = {}
        for key, item in value.items():
            if key == 'execution_lag_bars':
                if type(item) is not int or item < 1:
                    raise ValueError('Holdout execution lag must be a positive integer')
                row[key] = item
            elif key in ('take_profit_bps', 'stop_loss_bps') and item is None:
                row[key] = None
            else:
                if isinstance(item, bool) or not isinstance(item, numbers.Real) or not math.isfinite(float(item)):
                    raise ValueError(f'Holdout {key} must be finite real')
                number = float(item)
                if number < 0 or (key in ('notional_rate', 'take_profit_bps', 'stop_loss_bps') and number <= 0) or (key == 'notional_rate' and number > 1):
                    raise ValueError(f'Invalid holdout {key}')
                row[key] = number
        result.append(row)
    return result


def members(selection: Sensor | Cohort) -> list[Sensor]:
    if type(selection) is Sensor:
        return [selection]
    if type(selection) is not Cohort:
        raise ValueError('Holdout accepts exact Sensor or Cohort only')
    unchanged(selection)
    values = cast(list[Sensor], vars(selection)['_members'])
    if not values or len(values) != len(selection.member_ids) or selection.aggregation_mode not in ('majority_vote', 'probability_weighted') or getattr(selection, '_VOTE_THRESHOLD', None) != _VOTE_THRESHOLD:
        raise ValueError('Unsupported holdout Cohort membership/aggregation')
    if [(item.manifest_id, item.permutation_id) for item in values] != selection.member_ids:
        raise ValueError('Holdout Cohort member bindings disagree')
    return values


def _member_state(sensor: Sensor) -> tuple[object, ...]:
    return tuple(vars(sensor)[key] for key in ('_yaml_reference', '_round_params', '_fitted_params', '_model', '_trade_contract'))


def state_digest(selection: Sensor | Cohort) -> str:
    state = tuple(_member_state(item) for item in members(selection))
    aggregate = ('sensor',) if type(selection) is Sensor else (cast(Cohort, selection).aggregation_mode, cast(Cohort, selection).member_ids, getattr(selection, '_VOTE_THRESHOLD', None), *((vars(selection)['_mixed_policy'],) if vars(selection).get('_mixed_policy') is not None else ()))
    return sha256(pickle.dumps((state, aggregate), protocol=5)).hexdigest()


def manifest_projection(manifest: MLManifest) -> object:
    return normalized({field.name: getattr(manifest, field.name) for field in fields(manifest)})


def provenance(sensor: Sensor) -> dict[str, object]:
    value = cast(dict[str, object], vars(sensor)['_fitted_params']).get('_holdout')
    if not isinstance(value, dict):
        raise ValueError('Holdout requires Trainer ProvenanceV1, not legacy fitted state')
    proof = cast(dict[str, object], value)
    if proof.get('schema_version') != 1 or proof.get('role') != 'trainer' or proof.get('schema_reason') is not None:
        raise ValueError('Holdout requires completed Trainer schemas')
    for name in ('input_schema', 'model_schema'):
        schema = proof.get(name)
        if not isinstance(schema, list) or not schema:
            raise ValueError('Holdout requires ordered input/model schemas')
        for field in cast(list[dict[str, object]], schema):
            if set(field) != {'name', 'dtype'} or not isinstance(field['name'], str) or not isinstance(field['dtype'], str):
                raise ValueError('Malformed holdout feature schema')
            if name == 'model_schema' and field['dtype'] not in ('Float32', 'Float64', 'Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64', 'Boolean'):
                raise ValueError('Holdout model features must be numeric')
        if len({field['name'] for field in cast(list[dict[str, object]], schema)}) != len(cast(list[object], schema)):
            raise ValueError('Duplicate holdout schema columns')
    return proof


def _fitted_value(value: object) -> None:
    if type(value) in _FITTED_TYPES:
        unchanged(value)
        _fitted_value(vars(value))
    elif isinstance(value, Mapping):
        for item in cast(Mapping[str, object], value).values():
            _fitted_value(item)
    elif isinstance(value, (list, tuple)):
        for item in cast(list[object] | tuple[object, ...], value):
            _fitted_value(item)
    elif isinstance(value, np.ndarray):
        if cast(npt.NDArray[np.generic], value).dtype.kind not in ('b', 'i', 'u', 'f', 'U', 'S'):
            raise ValueError('Unsupported holdout fitted array')
    elif type(value) not in (pl.DataFrame, pl.Series):
        if callable(value):
            raise ValueError('Holdout refuses fitted callbacks')
        _ = normalized(value)


def validate_member(sensor: Sensor, locator: Mapping[str, object] | None) -> dict[str, object]:
    if type(sensor) is not Sensor:
        raise ValueError('Holdout accepts ordinary exact binary Sensors only')
    unchanged(sensor)
    wrapper = cast(LogRegBinary | LightGBMBinary, vars(sensor)['_model'])
    if type(wrapper) not in (LogRegBinary, LightGBMBinary):
        raise ValueError('Holdout supports fitted LogRegBinary/LightGBMBinary only')
    if set(vars(wrapper)) - {'model', 'prediction_calibration_config', '_fitted_calibrator', '_calibration_threshold', '_val_score'}:
        raise ValueError('Holdout refuses undeclared model-wrapper state')
    unchanged(wrapper)
    model = getattr(wrapper, 'model', None)
    if type(wrapper) is LogRegBinary:
        if type(model) is not LogisticRegression:
            raise ValueError('Holdout requires exact sklearn LogisticRegression')
        unchanged(model)
    else:
        from lightgbm import LGBMClassifier
        if type(model) is not LGBMClassifier or any(name in vars(model) for name in ('predict', 'predict_proba')):
            raise ValueError('Holdout requires exact LGBMClassifier')
        if any(getattr(LGBMClassifier, name).__module__ != 'lightgbm.sklearn' or getattr(LGBMClassifier, name).__qualname__ != f'LGBMClassifier.{name}' for name in ('predict', 'predict_proba')):
            raise ValueError('Holdout refuses replaced LightGBM prediction methods')
    check_is_fitted(model)
    if list(cast(Iterable[int], getattr(model, 'classes_', []))) != [0, 1]:
        raise ValueError('Holdout estimator classes must be [0,1]')
    config = wrapper.prediction_calibration_config
    calibrator = vars(wrapper).get('_fitted_calibrator')
    if config is not None:
        if calibrator is None:
            raise ValueError('Holdout refuses configured but unfitted calibration')
        if config.calibration_func is None:
            if calibrator is not model:
                raise ValueError('Holdout threshold-only calibrator must be the fitted estimator')
        elif type(calibrator) is CalibratedClassifierCV:
            unchanged(calibrator)
            check_is_fitted(calibrator)
            estimator = vars(calibrator).get('estimator')
            calibrated = cast(list[object], vars(calibrator).get('calibrated_classifiers_', []))
            if type(estimator) is not FrozenEstimator or vars(estimator).get('estimator') is not model or not calibrated or any(vars(item).get('estimator') is not estimator for item in calibrated):
                raise ValueError('Holdout calibration must wrap the selected frozen estimator')
            unchanged(estimator)
        else:
            raise ValueError('Holdout supports fitted sklearn calibration only')
        threshold = vars(wrapper).get('_calibration_threshold')
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real) or not math.isfinite(float(threshold)):
            raise ValueError('Holdout requires a finite fitted threshold')
    manifest = cast(MLManifest | None, vars(sensor)['_manifest'])
    fresh = CompiledSFD(cast(dict[str, object], vars(sensor)['_yaml_reference'])).manifest()
    if type(fresh) is not MLManifest or (manifest is not None and (type(manifest) is not MLManifest or manifest_projection(manifest) != manifest_projection(fresh))):
        raise ValueError('Holdout cached manifest differs from the declared pipeline')
    effective = fresh if manifest is None else manifest
    unchanged(effective)
    fitted = cast(dict[str, object], vars(sensor)['_fitted_params'])
    names = {'_holdout', '_scaler', '_pca', '_pca_input_feature_names', '_pca_feature_names', '_pca_n_components', '_column_shims', f'_target_cls_{effective.target_column}'}
    if effective.scaler is not None:
        names.update(entry[0] for entry in effective.scaler[0])
    if set(fitted) - names:
        raise ValueError('Holdout refuses undeclared fitted state')
    _fitted_value(fitted)
    if any(name in vars(effective) for name in ('sensor_input_prep', '_prepare_column_shims', '_process_bars')):
        raise ValueError('Holdout refuses replaced manifest methods')
    if effective.decoder_lookback != 1 or effective.bar_formation is not None or effective.split_walk_forward is not None or effective.test_data_source_config is not None or effective.data_dict_extension is not None or effective.split_dates is None or any(type(item) is not date for item in effective.split_dates):
        raise ValueError('Holdout requires ordinary dated native binary inference')
    params = cast(dict[str, object], vars(sensor)['_round_params'])
    _ = manifest_controls(effective, params)
    events.check_frozen_backtest(effective.backtest_config, params)
    resolved = effective.prediction_calibration_config
    expected = None if resolved is None else resolve_config(resolved, params)
    if normalized(config) != normalized(expected):
        raise ValueError('Holdout fitted calibration differs from declared configuration')
    proof = provenance(sensor)
    interval = proof.get('interval_seconds')
    if type(interval) is not int:
        raise ValueError('Holdout requires recorded native interval')
    source = source_spec(effective, history.ProjectBinding(Path.cwd(), None, 0, None, interval))
    return {'yaml': normalized(cast(dict[str, object], vars(sensor)['_yaml_reference'])), 'params': normalized(cast(dict[str, object], vars(sensor)['_round_params'])), 'pipeline': manifest_projection(effective),
            'source': normalized(source), 'manifest_id': sensor.manifest_id, 'permutation_id': sensor.permutation_id,
            'provenance': normalized(proof), 'test_end': effective.split_dates[-1].isoformat(), **events.execution_key(sensor, locator)}


class FrozenHoldoutSelection:
    __slots__ = ('_lock', '_snapshot', '_spec', '_spec_digest', '_state_digest')
    def __new__(cls, token: object, /) -> FrozenHoldoutSelection:
        if token is not _TOKEN:
            raise TypeError('Use freeze_holdout_selection')
        return super().__new__(cls)

    def __init__(self, token: object, /) -> None:
        if token is not _TOKEN:
            raise TypeError('Use freeze_holdout_selection')
        super().__init__()
        self._snapshot: tuple[Sensor | Cohort, NextBarUpTarget | NextBarDownTarget | ThresholdBinaryTarget | None] | None = None
        self._spec: dict[str, object] | None = None
        self._state_digest: str | None = None
        self._spec_digest: str | None = None
        self._lock = Lock()


def _model_witness(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _model_witness(item) for key, item in cast(Mapping[str, object], value).items()}
    if isinstance(value, (tuple, list)):
        return tuple(_model_witness(item) for item in cast(tuple[object, ...] | list[object], value))
    if type(value).__module__ == 'lightgbm.basic' and type(value).__name__ == 'Booster':
        from lightgbm import Booster
        booster = cast(Booster, value)
        return (booster.model_to_string(num_iteration=-1), booster.num_trees(), booster.current_iteration(), max(0, booster.best_iteration))
    if type(value).__module__.startswith(('sklearn.', 'lightgbm.', 'limen.sfd.reference_architecture.')) and hasattr(value, '__dict__'):
        return (type(value).__module__, type(value).__qualname__, _model_witness(vars(value)))
    return value


def _copy_witness(selection: Sensor | Cohort) -> bytes:
    return pickle.dumps(tuple((*_member_state(item)[:3], _model_witness(vars(item)['_model']), vars(item)['_trade_contract'], vars(item).get('_trade_digest')) for item in members(selection)), protocol=5)


def owned_copy(selection: Sensor | Cohort) -> Sensor | Cohort:
    before = _copy_witness(selection)
    copied: list[Sensor] = []
    for original in members(selection):
        member = copy.copy(original)
        vars(member).clear()
        vars(member).update(zip(('_yaml_reference', '_round_params', '_fitted_params', '_model', '_trade_contract'), copy.deepcopy(_member_state(original)), strict=True))
        if type(vars(member)['_model']) is LightGBMBinary:
            vars(member)['_model'].model.booster_.best_iteration = vars(original)['_model'].model.booster_.best_iteration
        member.permutation_id, member.manifest_id = original.permutation_id, original.manifest_id
        vars(member)['_trade_digest'] = contract_digest(vars(member)['_trade_contract']) if vars(member)['_trade_contract'] is not None else None
        cached = cast(MLManifest | None, vars(original)['_manifest'])
        if cached is None:
            manifest = CompiledSFD(cast(dict[str, object], vars(member)['_yaml_reference'])).manifest()
        else:
            manifest = copy.copy(cached)
            values = {field.name: getattr(cached, field.name) for field in fields(cached)}
            if cached.data_source_config is not None:
                source = copy.copy(cached.data_source_config)
                source.method = getattr(source.method, '__func__', source.method)
                values['data_source_config'] = source
            vars(manifest).clear()
            vars(manifest).update(copy.deepcopy(values))
            if '_holdout_binding' in vars(cached):
                vars(manifest)['_holdout_binding'] = vars(cached)['_holdout_binding']
        vars(member)['_manifest'] = manifest
        _ = validate_member(member, None)
        copied.append(member)
    result: Sensor | Cohort = copied[0]
    if type(selection) is Cohort:
        result = copy.copy(selection)
        vars(result).clear()
        vars(result).update(_members=copied, member_ids=copy.deepcopy(selection.member_ids), aggregation_mode=selection.aggregation_mode, architecture_id=selection.architecture_id, _mixed_policy=copy.deepcopy(vars(selection).get('_mixed_policy')))
    if _copy_witness(result) != before:
        raise ValueError('Holdout fitted-state copy changed selected state')
    return result


def freeze_holdout_selection(selection: Sensor | Cohort, *, context_start: str,
                             scenarios: list[Mapping[str, object]],
                             evaluation_target: NextBarUpTarget | NextBarDownTarget | ThresholdBinaryTarget | None = None, round_locator: Mapping[str, object] | None = None) -> FrozenHoldoutSelection:
    """Own the fitted selection and preregister its final-assessment specification."""
    _ = history.utc_day(context_start)
    declarations = scenario_declarations(scenarios, allow_empty=events.freeze_kind(selection, round_locator) == 'contract')
    specs = [validate_member(member, round_locator) for member in members(selection)]
    if evaluation_target is not None:
        if type(evaluation_target) in (NextBarUpTarget, NextBarDownTarget):
            if set(vars(evaluation_target)) != {'target_name'} or not isinstance(cast(object, evaluation_target.target_name), str):
                raise ValueError('Holdout target must be an initialized exact next-bar binary target')
        elif type(evaluation_target) is ThresholdBinaryTarget:
            state = cast(dict[str, object], vars(evaluation_target))
            if (set(state) != {'target_name', 'source_column', 'threshold'} or not isinstance(state['target_name'], str) or not state['target_name']
                    or not isinstance(state['source_column'], str) or state['source_column'] not in ('open', 'high', 'low', 'close')
                    or type(state['threshold']) not in (int, float) or not math.isfinite(cast(float, state['threshold']))
                    or ThresholdBinaryTarget.transform.__defaults__ != (-1,)):
                raise ValueError('Holdout threshold target must declare a fixed OHLC comparison with horizon one')
        else:
            raise ValueError('Holdout target must be an initialized exact next-bar binary target')
        unchanged(evaluation_target)
    versions: dict[str, object] = {'python': platform.python_version()}
    for package in ('vaquum-limen', 'numpy', 'polars', 'scikit-learn', 'lightgbm'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    snapshot = (owned_copy(selection), copy.deepcopy(evaluation_target))
    spec: dict[str, object] = {'schema_version': 1, 'members': specs, 'context_start': context_start, 'scenarios': declarations,
                    'aggregation': 'sensor' if type(selection) is Sensor else cast(Cohort, selection).aggregation_mode,
                    'order': [spec['permutation_id'] for spec in specs], 'threshold': 0.5,
                    'target': None if evaluation_target is None else {'class': normalized(type(evaluation_target)), 'params': copy.deepcopy(vars(evaluation_target)), **({'shift': -1} if type(evaluation_target) is ThresholdBinaryTarget else {})}, 'versions': versions}
    result = FrozenHoldoutSelection(_TOKEN)
    for key, value in zip(('_snapshot', '_spec', '_state_digest', '_spec_digest'), (snapshot, spec, state_digest(snapshot[0]), digest(spec)), strict=True):
        object.__setattr__(result, key, value)
    return result


def frozen_parts(handle: FrozenHoldoutSelection) -> tuple[tuple[Sensor | Cohort, NextBarUpTarget | NextBarDownTarget | ThresholdBinaryTarget | None], dict[str, object], str, str, Lock]:
    parts = tuple(object.__getattribute__(handle, key) for key in ('_snapshot', '_spec', '_state_digest', '_spec_digest', '_lock'))
    if any(value is None for value in parts):
        raise ValueError('Holdout handle was not initialized by freeze_holdout_selection')
    return cast(tuple[tuple[Sensor | Cohort, NextBarUpTarget | NextBarDownTarget | ThresholdBinaryTarget | None], dict[str, object], str, str, Lock], parts)
