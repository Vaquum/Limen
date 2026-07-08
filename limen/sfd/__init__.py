from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING
from typing import Any


_LAZY_EXPORTS = {
    'foundational_sfd': ('limen.sfd.foundational_sfd', None),
    'foundational_rule_based': ('limen.sfd.foundational_sfd.rule_based', None),
    'logreg_binary': ('limen.sfd.foundational_sfd.logreg_binary', None),
    'random_binary': ('limen.sfd.foundational_sfd.random_binary', None),
    'xgboost_regressor': ('limen.sfd.foundational_sfd.xgboost_regressor', None),
}


if TYPE_CHECKING:
    from limen.sfd import foundational_sfd
    from limen.sfd.foundational_sfd import logreg_binary
    from limen.sfd.foundational_sfd import random_binary
    from limen.sfd.foundational_sfd import rule_based as foundational_rule_based
    from limen.sfd.foundational_sfd import xgboost_regressor


__all__ = [
    'foundational_rule_based',
    'foundational_sfd',
    'logreg_binary',
    'random_binary',
    'xgboost_regressor',
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'limen.sfd' has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value
