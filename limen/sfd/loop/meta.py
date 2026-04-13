'''Hand-maintained metadata for the Loop payload compiler.

Mappings that cannot be derived by introspection:

- LABEL_TARGET_COLUMNS: label function name → target column name produced
- SCALER_NAME_MAP: PascalCase class name (Loop's format) → SCALER_REGISTRY key
- FITTED_LABELS: label name → fitted-transform wiring (for labels whose
  threshold/cutoff is computed from training data rather than user-specified)

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands.
'''

from collections.abc import Callable
from typing import Any

from limen.features.quantile_flag import compute_quantile_cutoff


LABEL_TARGET_COLUMNS: dict[str, str] = {
    'forward_breakout_target': 'forward_breakout',
}


def get_target_column(label_name: str) -> str:

    '''
    Compute the target column name produced by a label function.

    Args:
        label_name (str): The label function short name as used in the payload

    Returns:
        str: The target column name. Falls back to label_name itself when no
            explicit override is registered (works for labels like quantile_flag
            whose function name matches the column name they produce)

    '''

    return LABEL_TARGET_COLUMNS.get(label_name, label_name)


SCALER_NAME_MAP: dict[str, str] = {
    'LinearScaler': 'linear',
    'LogRegScaler': 'logreg',
    'RobustScaler': 'robust',
    'RankGaussScaler': 'rank_gauss',
}


class FittedLabelConfig:

    '''
    Wiring specification for a label that requires a fitted transform.

    Fitted labels compute a per-run parameter from the training split
    (e.g. a quantile cutoff) before being applied. The manifest builder
    expresses this via `.add_fitted_transform().fit_param().with_params()`.

    Attributes:
        compute_func: Callable that takes (data, **compute_params) and
            returns a scalar. Called once on the training split
        fitted_param_name: Name of the computed value in round_params.
            Must start with underscore to mark it as internal
        compute_param_keys: Names of user params (from the payload label
            entry) that must be forwarded to compute_func
        apply_param_keys: Names of user params that are forwarded to the
            label function itself at transform time. The fitted param is
            always injected as `cutoff` (keyed on fitted_param_name)
        apply_fitted_as: The keyword argument name under which the fitted
            param is passed to the label function
    '''

    def __init__(self,
                 compute_func: Callable[..., Any],
                 fitted_param_name: str,
                 compute_param_keys: tuple[str, ...],
                 apply_param_keys: tuple[str, ...],
                 apply_fitted_as: str) -> None:

        self.compute_func = compute_func
        self.fitted_param_name = fitted_param_name
        self.compute_param_keys = compute_param_keys
        self.apply_param_keys = apply_param_keys
        self.apply_fitted_as = apply_fitted_as


FITTED_LABELS: dict[str, FittedLabelConfig] = {
    'quantile_flag': FittedLabelConfig(
        compute_func=compute_quantile_cutoff,
        fitted_param_name='_quantile_cutoff',
        compute_param_keys=('col', 'q'),
        apply_param_keys=('col',),
        apply_fitted_as='cutoff',
    ),
}


__all__ = [
    'FITTED_LABELS',
    'LABEL_TARGET_COLUMNS',
    'SCALER_NAME_MAP',
    'FittedLabelConfig',
    'get_target_column',
]
