from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING
from typing import Any


_LAZY_EXPORTS = {
    'dlinear_regressor': ('limen.sfd.foundational_sfd.dlinear_regressor', None),
    'dollar_bar_crash_reversal': (
        'limen.sfd.foundational_sfd.dollar_bar_crash_reversal',
        None,
    ),
    'lightgbm_binary': ('limen.sfd.foundational_sfd.lightgbm_binary', None),
    'logreg_binary': ('limen.sfd.foundational_sfd.logreg_binary', None),
    'random_binary': ('limen.sfd.foundational_sfd.random_binary', None),
    'rule_based': ('limen.sfd.foundational_sfd.rule_based', None),
    'tabpfn_binary': ('limen.sfd.foundational_sfd.tabpfn_binary', None),
    'xgboost_regressor': ('limen.sfd.foundational_sfd.xgboost_regressor', None),
}


if TYPE_CHECKING:
    from limen.sfd.foundational_sfd import dlinear_regressor
    from limen.sfd.foundational_sfd import dollar_bar_crash_reversal
    from limen.sfd.foundational_sfd import lightgbm_binary
    from limen.sfd.foundational_sfd import logreg_binary
    from limen.sfd.foundational_sfd import random_binary
    from limen.sfd.foundational_sfd import rule_based
    from limen.sfd.foundational_sfd import tabpfn_binary
    from limen.sfd.foundational_sfd import xgboost_regressor


__all__ = [
    'dlinear_regressor',
    'dollar_bar_crash_reversal',
    'lightgbm_binary',
    'logreg_binary',
    'random_binary',
    'rule_based',
    'tabpfn_binary',
    'xgboost_regressor',
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'limen.sfd.foundational_sfd' has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value
