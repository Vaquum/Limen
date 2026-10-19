from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import math
import numbers
import pickle
import platform
from pathlib import Path
from threading import Lock
from typing import cast

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from limen.cohort import Cohort
from limen.experiment import _holdout_history as history
from limen.experiment._holdout_source import source_spec
from limen.experiment.manifest_core import MLManifest
from limen.inference.sensor import Sensor
from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
from limen.sfd.reference_architecture.logreg_binary import LogRegBinary
from limen.targets.next_bar_down import NextBarDownTarget
from limen.targets.next_bar_up import NextBarUpTarget
from limen.yaml.compiler import CompiledSFD

_TOKEN = object()
_VOTE_THRESHOLD = 0.5
_IMPLEMENTATIONS = {cls: {name: getattr(cls, name) for name in names} for cls, names in (
    (Sensor, ('predict', '_get_manifest', '_last_bar_inside_training_window')),
    (Cohort, ('_aggregate_bar_predictions', '_fallback_vote_threshold', '_validate_probability_range')),
    (LogRegBinary, ('predict',)), (LightGBMBinary, ('predict',)),
    (LogisticRegression, ('predict', 'predict_proba')),
    (CalibratedClassifierCV, ('predict_proba',)),
    (MLManifest, ('sensor_input_prep', '_prepare_column_shims')),
    (NextBarUpTarget, ('transform',)), (NextBarDownTarget, ('transform',)))}


def unchanged(value: object) -> None:
    for name, original in _IMPLEMENTATIONS.get(type(value), {}).items():
        if name in vars(value) or getattr(type(value), name) is not original:
            raise ValueError(f'Holdout refuses replaced {type(value).__name__}.{name}')


def normalized(value: object) -> object:
    result: object
    if value is None or type(value) in (str, bool, int):
        result = value
    elif isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Holdout specification must be finite')
    elif isinstance(value, (date, datetime)):
        result = value.isoformat()
    elif isinstance(value, Mapping):
        items = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in items):
            raise ValueError('Holdout specification requires string keys')
        result = {str(key): normalized(item) for key, item in items.items()}
    elif isinstance(value, (list, tuple)):
        result = [normalized(item) for item in cast(list[object] | tuple[object, ...], value)]
    elif inspect.ismethod(value):
        result = normalized(value.__func__)
    elif inspect.isfunction(value) or isinstance(value, type):
        if not value.__module__.startswith(('limen.', 'sklearn.')):
            raise ValueError('Holdout specification refuses custom callables')
        name = f'{value.__module__}.{value.__qualname__}'
        if inspect.isfunction(value) and value.__closure__:
            result = {'callable': name, 'defaults': normalized(value.__defaults__),
                    'closure': {key: normalized(cell.cell_contents) for key, cell in zip(value.__code__.co_freevars, value.__closure__, strict=True)}}
        else:
            result = name
    elif is_dataclass(value) and not isinstance(value, type) and type(value).__module__.startswith('limen.'):
        result = {field.name: normalized(getattr(value, field.name)) for field in fields(value)}
    else:
        raise ValueError(f'Unsupported holdout specification value: {type(value).__name__}')
    return result


def encoded(value: object) -> bytes:
    return json.dumps(normalized(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value: object) -> str:
    return sha256(encoded(value)).hexdigest()


def scenario_declarations(values: list[Mapping[str, object]]) -> list[dict[str, object]]:
    keys = {'fee_bps', 'slip_bps', 'notional_rate', 'execution_lag_bars', 'take_profit_bps', 'stop_loss_bps'}
    if not isinstance(values, list) or not values:
        raise ValueError('Holdout requires a nonempty scenario list')
    result: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != keys:
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
    values = selection._members
    if not values or len(values) != len(selection.member_ids) or selection.aggregation_mode not in ('majority_vote', 'probability_weighted') or selection._VOTE_THRESHOLD != _VOTE_THRESHOLD:
        raise ValueError('Unsupported holdout Cohort membership/aggregation')
    if [(item.manifest_id, item.permutation_id) for item in values] != selection.member_ids:
        raise ValueError('Holdout Cohort member bindings disagree')
    return values


def state_digest(selection: Sensor | Cohort) -> str:
    state = tuple((item._yaml_reference, item._round_params, item._fitted_params, item._model, item._trade_contract) for item in members(selection))
    aggregate = ('sensor',) if type(selection) is Sensor else (selection.aggregation_mode, selection.member_ids, selection._VOTE_THRESHOLD)
    return sha256(pickle.dumps((state, aggregate), protocol=5)).hexdigest()


def manifest_projection(manifest: MLManifest) -> object:
    return normalized({field.name: getattr(manifest, field.name) for field in fields(manifest)})


def provenance(sensor: Sensor) -> dict[str, object]:
    value = sensor._fitted_params.get('_holdout')
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
        if len({field['name'] for field in cast(list[dict[str, object]], schema)}) != len(schema):
            raise ValueError('Duplicate holdout schema columns')
    return proof


def validate_member(sensor: Sensor) -> dict[str, object]:
    if type(sensor) is not Sensor or sensor.trade_contract is not None:
        raise ValueError('Holdout accepts ordinary exact binary Sensors only')
    unchanged(sensor)
    wrapper = sensor._model
    if type(wrapper) not in (LogRegBinary, LightGBMBinary):
        raise ValueError('Holdout supports fitted LogRegBinary/LightGBMBinary only')
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
        if model.booster_.best_iteration > 0:
            raise ValueError('Holdout refuses positive native LightGBM best_iteration')
    check_is_fitted(model)
    if list(getattr(model, 'classes_', [])) != [0, 1]:
        raise ValueError('Holdout estimator classes must be [0,1]')
    config = wrapper.prediction_calibration_config
    calibrator = wrapper._fitted_calibrator
    if config is not None:
        if calibrator is None:
            raise ValueError('Holdout refuses configured but unfitted calibration')
        if config.calibration_func is None:
            if calibrator is not model:
                raise ValueError('Holdout threshold-only calibrator must be the fitted estimator')
        elif type(calibrator) is CalibratedClassifierCV:
            unchanged(calibrator)
            check_is_fitted(calibrator)
        else:
            raise ValueError('Holdout supports fitted sklearn calibration only')
        threshold = wrapper._calibration_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real) or not math.isfinite(float(threshold)):
            raise ValueError('Holdout requires a finite fitted threshold')
    manifest = sensor._manifest
    fresh = CompiledSFD(sensor._yaml_reference).manifest()
    if type(fresh) is not MLManifest or (manifest is not None and (type(manifest) is not MLManifest or manifest_projection(manifest) != manifest_projection(fresh))):
        raise ValueError('Holdout cached manifest differs from the declared pipeline')
    effective = fresh if manifest is None else manifest
    unchanged(effective)
    if any(name in vars(effective) for name in ('sensor_input_prep', '_prepare_column_shims', '_process_bars')):
        raise ValueError('Holdout refuses replaced manifest methods')
    if effective.decoder_lookback != 1 or effective.bar_formation is not None or effective.split_walk_forward is not None or effective.test_data_source_config is not None or effective.data_dict_extension is not None or effective.split_dates is None or any(type(item) is not date for item in effective.split_dates):
        raise ValueError('Holdout requires ordinary dated native binary inference')
    backtest = effective.backtest_config
    if backtest is not None and (backtest.product is not None or backtest.execution_data_source is not None or backtest.funding is not None or backtest.prediction_mode != 'binary'):
        raise ValueError('Unsupported holdout execution mode')
    resolved = effective.prediction_calibration_config
    if normalized(config) != normalized(resolved.resolve(sensor._round_params) if resolved is not None else None):
        raise ValueError('Holdout fitted calibration differs from declared configuration')
    proof = provenance(sensor)
    interval = proof.get('interval_seconds')
    if type(interval) is not int:
        raise ValueError('Holdout requires recorded native interval')
    source = source_spec(effective, history.ProjectBinding(Path.cwd(), None, 0, None, interval))
    return {'yaml': normalized(sensor._yaml_reference), 'params': normalized(sensor._round_params), 'pipeline': manifest_projection(effective),
            'source': normalized(source), 'manifest_id': sensor.manifest_id, 'permutation_id': sensor.permutation_id,
            'provenance': normalized(proof), 'test_end': effective.split_dates[-1].isoformat()}


class FrozenHoldoutSelection:
    __slots__ = ('_lock', '_snapshot', '_spec', '_spec_digest', '_state_digest')
    _snapshot: tuple[Sensor | Cohort, NextBarUpTarget | NextBarDownTarget | None]
    _spec: dict[str, object]
    _state_digest: str
    _spec_digest: str
    _lock: Lock

    def __new__(cls, token: object, /) -> FrozenHoldoutSelection:
        if token is not _TOKEN:
            raise TypeError('Use freeze_holdout_selection')
        return super().__new__(cls)


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
    return pickle.dumps(tuple((item._yaml_reference, item._round_params, item._fitted_params, _model_witness(item._model), item._trade_contract) for item in members(selection)), protocol=5)


def owned_copy(selection: Sensor | Cohort) -> Sensor | Cohort:
    before = _copy_witness(selection)
    result = copy.deepcopy(selection)
    if _copy_witness(result) != before:
        raise ValueError('Holdout fitted-state copy changed selected state')
    for original, member in zip(members(selection), members(result), strict=True):
        _ = validate_member(member)
        if original._manifest is None:
            member._manifest = CompiledSFD(member._yaml_reference).manifest()
    return result


def freeze_holdout_selection(selection: Sensor | Cohort, *, context_start: str,
                             scenarios: list[Mapping[str, object]],
                             evaluation_target: NextBarUpTarget | NextBarDownTarget | None = None) -> FrozenHoldoutSelection:
    """Own the fitted selection and preregister its final-assessment specification."""
    _ = history.utc_day(context_start)
    declarations = scenario_declarations(scenarios)
    specs = [validate_member(member) for member in members(selection)]
    if evaluation_target is not None:
        if type(evaluation_target) not in (NextBarUpTarget, NextBarDownTarget) or set(vars(evaluation_target)) != {'target_name'} or not isinstance(evaluation_target.target_name, str):
            raise ValueError('Holdout target must be an initialized exact next-bar binary target')
        unchanged(evaluation_target)
    versions: dict[str, object] = {'python': platform.python_version()}
    for package in ('limen', 'numpy', 'polars', 'scikit-learn', 'lightgbm'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    result = FrozenHoldoutSelection(_TOKEN)
    result._snapshot = (owned_copy(selection), copy.deepcopy(evaluation_target))
    result._spec = {'schema_version': 1, 'members': specs, 'context_start': context_start, 'scenarios': declarations,
                    'aggregation': 'sensor' if type(selection) is Sensor else selection.aggregation_mode,
                    'order': [spec['permutation_id'] for spec in specs], 'threshold': 0.5,
                    'target': None if evaluation_target is None else {'class': normalized(type(evaluation_target)), 'params': vars(evaluation_target)}, 'versions': versions}
    result._state_digest, result._spec_digest = state_digest(result._snapshot[0]), digest(result._spec)
    result._lock = Lock()
    return result
