from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from importlib import import_module
from typing import Literal, cast

from limen.experiment._factorized_contract import ReuseContract, declared


def _pure(_params: Mapping[str, object]) -> None:
    return None


def _seed(params: Mapping[str, object]) -> None:
    if type(params.get('random_state')) is not int:
        raise ValueError('an explicit integer random_state is required')


def _ridge(params: Mapping[str, object]) -> None:
    if params.get('solver') != 'cholesky' or params.get('positive') is not False or params.get('copy_X') is not True:
        raise ValueError('Ridge requires solver=cholesky, positive=False and copy_X=True')


def _lightgbm(params: Mapping[str, object]) -> None:
    _seed(params)
    expected = {'deterministic': True, 'force_row_wise': True, 'boosting_type': 'gbdt', 'n_jobs': 1,
                'subsample_freq': 0, 'subsample': 1.0, 'colsample_bytree': 1.0}
    if any(type(params.get(key)) is not type(value) or params.get(key) != value for key, value in expected.items()):
        raise ValueError('unsafe LightGBM settings')


def _logreg(params: Mapping[str, object]) -> None:
    solver = params.get('solver')
    if solver not in ('lbfgs', 'newton-cg', 'newton-cholesky', 'liblinear', 'sag', 'saga') or params.get('n_jobs') != 1:
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
                contracts[component] = ReuseContract(_seed if name == 'RandomBinaryTarget' else _pure)
    for name, validator, kind in (
        ('ridge_regressor', _ridge, 'continuous'), ('dlinear_regressor', _pure, 'continuous'),
        ('lightgbm_binary', _lightgbm, 'binary'), ('logreg_binary', _logreg, 'binary'),
        ('xgboost_regressor', _xgboost, 'continuous'), ('rule_based', _pure, 'binary'),
    ):
        module = import_module('limen.sfd.reference_architecture.' + name)
        contracts[getattr(module, name)] = ReuseContract(validator, prediction_kind=cast(Literal['binary', 'continuous'], kind))
    return contracts


def contract_for(component: object) -> ReuseContract | None:
    explicit = declared(component)
    if explicit is not None:
        return explicit
    try:
        return _contracts().get(component)
    except TypeError as exc:
        raise ValueError('Stateful callable instances require an explicit reuse declaration') from exc
