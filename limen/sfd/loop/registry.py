'''Name → callable registries for the Loop payload compiler.

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands.
'''

from collections.abc import Callable

import limen.features as _feat
import limen.indicators as _ind
import limen.sfd.reference_architecture as _arch
from limen.features.forward_breakout_target import forward_breakout_target as _forward_breakout_target
from limen.scalers.registry import SCALER_REGISTRY as _SCALER_REG


# Components that exist in Limen but are not exported via __all__ in their
# parent package. We import them directly so the Loop payload can reference
# them by short name. Keep this list minimal — adding entries here is the
# only place the quarantine policy expects manual maintenance.
_FEATURE_MANUAL_ADDITIONS: dict[str, Callable] = {
    'forward_breakout_target': _forward_breakout_target,
}


INDICATOR_REGISTRY: dict[str, Callable] = {
    name: getattr(_ind, name) for name in _ind.__all__
}

FEATURE_REGISTRY: dict[str, Callable] = {
    **{name: getattr(_feat, name) for name in _feat.__all__},
    **_FEATURE_MANUAL_ADDITIONS,
}


def _is_model_function(name: str) -> bool:

    obj = getattr(_arch, name, None)
    if obj is None:
        return False
    if isinstance(obj, type):
        return False
    return callable(obj)


MODEL_REGISTRY: dict[str, Callable] = {
    name: getattr(_arch, name) for name in _arch.__all__ if _is_model_function(name)
}

SCALER_REGISTRY: dict[str, type] = dict(_SCALER_REG)


__all__ = [
    'FEATURE_REGISTRY',
    'INDICATOR_REGISTRY',
    'MODEL_REGISTRY',
    'SCALER_REGISTRY',
]
