from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from functools import lru_cache, partial
from importlib import import_module
from typing import Literal, cast

from limen.experiment._factorized_contract import ReuseContract, declared


def native_threads() -> AbstractContextManager[object]:
    limit = cast(Callable[..., AbstractContextManager[object]], vars(import_module('threadpoolctl'))['threadpool_limits'])
    return limit(limits=1)


def native_rule(component: object) -> bool:
    while isinstance(component, partial):
        component = component.func
    return component is vars(import_module('limen.sfd.reference_architecture.rule_based'))['rule_based']


def _pure(_params: Mapping[str, object]) -> None:
    return None


def _seed(params: Mapping[str, object]) -> None:
    value = params.get('random_state')
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError('an explicit integer random_state is required')


def _ridge(params: Mapping[str, object]) -> None:
    if params.get('solver') != 'cholesky' or params.get('positive') is not False or params.get('copy_X') is not True:
        raise ValueError('Ridge requires solver=cholesky, positive=False and copy_X=True')


def _lightgbm(params: Mapping[str, object]) -> None:
    _seed(params)
    expected = {'deterministic': True, 'force_row_wise': True, 'boosting_type': 'gbdt', 'n_jobs': 1,
                'subsample_freq': 0, 'subsample': 1.0, 'colsample_bytree': 1.0}
    if any((params.get(key) is not value if isinstance(value, bool) else isinstance(params.get(key), bool) or params.get(key) != value) for key, value in expected.items()):
        raise ValueError('unsafe LightGBM settings')


def _logreg(params: Mapping[str, object]) -> None:
    solver = params.get('solver')
    if solver not in ('lbfgs', 'newton-cg', 'newton-cholesky', 'liblinear', 'sag', 'saga') or type(params.get('n_jobs')) is not int or params.get('n_jobs') != 1:
        raise ValueError('LogReg requires a supported solver and n_jobs=1')
    if solver in ('liblinear', 'sag', 'saga'):
        _seed(params)


def _xgboost(params: Mapping[str, object]) -> None:
    _seed(params)
    required = {'tree_method': 'hist', 'grow_policy': 'depthwise', 'sampling_method': 'uniform', 'booster': 'gbtree',
                'num_parallel_tree': 1, 'n_jobs': 1, 'subsample': 1.0, 'colsample_bytree': 1.0,
                'colsample_bylevel': 1.0, 'colsample_bynode': 1.0}
    if params.get('device') not in (None, 'cpu') or any(params.get(key) != value for key, value in required.items()):
        raise ValueError('XGBoost requires deterministic full-sample CPU histogram settings')


@lru_cache(maxsize=1)
def _contracts() -> dict[object, ReuseContract]:
    contracts: dict[object, ReuseContract] = {}
    # These exported numerical transforms have no random generator or mutable
    # invocation state. Stateful random targets receive a separate predicate.
    for namespace in ('features', 'indicators', 'transforms', 'scalers', 'targets', 'data.bars', 'metrics'):
        module = import_module('limen.' + namespace)
        for name in cast(list[str], getattr(module, '__all__', [])):
            component = getattr(module, name)
            if callable(component):
                contracts[component] = ReuseContract(_seed if name == 'RandomBinaryTarget' else _pure, validation_keys=('random_state',) if name == 'RandomBinaryTarget' else ())
    for name, validator, kind in (
        ('ridge_regressor', _ridge, 'continuous'), ('dlinear_regressor', _pure, 'continuous'),
        ('lightgbm_binary', _lightgbm, 'binary'), ('logreg_binary', _logreg, 'binary'),
        ('xgboost_regressor', _xgboost, 'continuous'), ('rule_based', _pure, 'binary'),
    ):
        module = import_module('limen.sfd.reference_architecture.' + name)
        keys = {'ridge_regressor': ('solver', 'positive', 'copy_X'), 'dlinear_regressor': (), 'rule_based': (),
                'lightgbm_binary': ('random_state', 'deterministic', 'force_row_wise', 'boosting_type', 'n_jobs', 'subsample_freq', 'subsample', 'colsample_bytree'),
                'logreg_binary': ('solver', 'n_jobs', 'random_state'),
                'xgboost_regressor': ('random_state', 'tree_method', 'grow_policy', 'sampling_method', 'booster', 'num_parallel_tree', 'n_jobs', 'subsample', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode', 'device')}
        contracts[getattr(module, name)] = ReuseContract(validator, validation_keys=keys[name], prediction_kind=cast(Literal['binary', 'continuous'], kind))
    calibration = import_module('limen.calibration')
    for name in ('sklearn_probability_calibrator', 'grid_threshold_optimizer'):
        contracts[getattr(calibration, name)] = ReuseContract(_pure, validation_keys=())
    return contracts


def contract_for(component: object) -> ReuseContract | None:
    explicit = declared(component)
    if explicit is not None:
        return explicit
    if isinstance(component, partial):
        return contract_for(component.func)
    try:
        return _contracts().get(component)
    except TypeError as exc:
        raise ValueError('Stateful callable instances require an explicit reuse declaration') from exc
