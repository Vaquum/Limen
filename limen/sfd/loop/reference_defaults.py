'''Default model hyperparam search spaces for reference architectures.

The Loop payload is not trusted to provide reference architecture parameters.
This module supplies them by reusing the foundational SFDs' params() dicts,
filtered to only those keys that match the reference architecture function's
signature.

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands.
'''

import inspect
from collections.abc import Callable
from typing import Any

from limen.sfd.foundational_sfd import logreg_binary as _fsfd_logreg
from limen.sfd.foundational_sfd import random_binary as _fsfd_random
from limen.sfd.foundational_sfd import xgboost_regressor as _fsfd_xgb


_FOUNDATIONAL_MAP: dict[str, Any] = {
    'logreg_binary': _fsfd_logreg,
    'random_binary': _fsfd_random,
    'xgboost_regressor': _fsfd_xgb,
}


def get_reference_architecture_params(arch_name: str,
                                      arch_func: Callable) -> dict[str, list]:

    '''
    Compute the default model hyperparam search space for a reference architecture.

    Filters the matching foundational SFD's params() to only keys that match the
    reference architecture function signature, excluding data-prep params (e.g.
    'frac_diff_d', 'roc_period', 'feature_groups') that are wired to the
    foundational SFD's specific manifest and have no meaning in Loop-driven
    manifests.

    Fallback chain:
        1. Foundational SFD's params() filtered by signature
        2. Signature defaults as singleton lists (for unknown arch or
           when the filter produces an empty dict, e.g. random_binary
           whose foundational params do not intersect its function signature)

    Args:
        arch_name (str): Reference architecture short name (e.g. 'logreg_binary')
        arch_func (Callable): The reference architecture function whose signature
            defines the canonical parameter set

    Returns:
        dict[str, list]: Mapping from parameter name to candidate values list

    '''

    sig_param_names = {
        p.name for p in inspect.signature(arch_func).parameters.values()
        if p.name != 'data'
    }

    sfd_module = _FOUNDATIONAL_MAP.get(arch_name)
    if sfd_module is not None:
        all_params = sfd_module.params()
        filtered = {k: v for k, v in all_params.items() if k in sig_param_names}
        if filtered:
            return filtered

    return _signature_defaults(arch_func)


def _signature_defaults(func: Callable) -> dict[str, list]:

    '''
    Compute singleton search space from a function's signature defaults.

    Args:
        func (Callable): Function whose default values define the search space

    Returns:
        dict[str, list]: Each parameter with a default becomes a one-element list

    '''

    sig = inspect.signature(func)
    return {
        name: [p.default]
        for name, p in sig.parameters.items()
        if name != 'data' and p.default is not inspect.Parameter.empty
    }


__all__ = [
    'get_reference_architecture_params',
]
