from __future__ import annotations

from importlib import import_module
from typing import Any

from limen.sfd.reference_architecture.base import ReferenceModel
from limen.sfd.reference_architecture.logreg_binary import LogRegBinary
from limen.sfd.reference_architecture.logreg_binary import logreg_binary
from limen.sfd.reference_architecture.random_binary import RandomBinary
from limen.sfd.reference_architecture.random_binary import random_binary
from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy
from limen.sfd.reference_architecture.rule_based import rule_based


_LAZY_EXPORTS = {
    'LightGBMBinary': ('limen.sfd.reference_architecture.lightgbm_binary', 'LightGBMBinary'),
    'TabPFNBinary': ('limen.sfd.reference_architecture.tabpfn_binary', 'TabPFNBinary'),
    'XGBoostRegressor': ('limen.sfd.reference_architecture.xgboost_regressor', 'XGBoostRegressor'),
    'lightgbm_binary': ('limen.sfd.reference_architecture.lightgbm_binary', 'lightgbm_binary'),
    'tabpfn_binary': ('limen.sfd.reference_architecture.tabpfn_binary', 'tabpfn_binary'),
    'xgboost_regressor': ('limen.sfd.reference_architecture.xgboost_regressor', 'xgboost_regressor'),
}


__all__ = [
    'LightGBMBinary',
    'LogRegBinary',
    'RandomBinary',
    'ReferenceModel',
    'RuleBasedStrategy',
    'TabPFNBinary',
    'XGBoostRegressor',
    'lightgbm_binary',
    'logreg_binary',
    'random_binary',
    'rule_based',
    'tabpfn_binary',
    'xgboost_regressor',
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'limen.sfd.reference_architecture' has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
