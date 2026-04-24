'''Name → callable registries for the Loop payload compiler.

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands.
'''

from collections.abc import Callable

import limen.features as _feat
import limen.indicators as _ind
import limen.sfd.reference_architecture as _arch
from limen.scalers.registry import SCALER_REGISTRY as _SCALER_REG


INDICATOR_REGISTRY: dict[str, Callable] = {
    name: getattr(_ind, name) for name in _ind.__all__
}

FEATURE_REGISTRY: dict[str, Callable] = {
    name: getattr(_feat, name) for name in _feat.__all__
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
