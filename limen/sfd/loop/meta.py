'''Hand-maintained metadata for the Loop payload compiler.

Mappings that cannot be derived by introspection:

- LABEL_TARGET_COLUMNS: label function name → target column name produced
- DATA_SOURCE_REGISTRY: method name string → HistoricalData bound method
- TARGET_LABEL_REGISTRY: label name → target class wiring (maps payload param
  names to target class __init__ and transform() param names)

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands.
'''

from collections.abc import Callable
from dataclasses import dataclass, field

from limen.data import HistoricalData
from limen.targets import ForwardBreakoutTarget
from limen.targets import QuantileBinaryTarget


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


DATA_SOURCE_REGISTRY: dict[str, Callable] = {
    'get_spot_klines': HistoricalData.get_spot_klines,
}


@dataclass
class TargetLabelEntry:

    '''
    Registry entry mapping a Loop payload label name to a target class.

    fit_param_aliases maps payload param names to the corresponding keyword
    arguments in the target class __init__ (after train_data and target_name).
    transform_param_aliases maps payload param names to transform() arguments.
    Params not listed in either alias dict are not forwarded.
    '''

    target_class: type
    fit_param_aliases: dict[str, str] = field(default_factory=dict)
    transform_param_aliases: dict[str, str] = field(default_factory=dict)


TARGET_LABEL_REGISTRY: dict[str, TargetLabelEntry] = {
    'quantile_flag': TargetLabelEntry(
        target_class=QuantileBinaryTarget,
        fit_param_aliases={'col': 'source_column', 'q': 'quantile'},
    ),
    'forward_breakout_target': TargetLabelEntry(
        target_class=ForwardBreakoutTarget,
        transform_param_aliases={
            'forward_periods': 'forward_periods',
            'threshold': 'threshold',
            'shift': 'shift',
        },
    ),
}


__all__ = [
    'DATA_SOURCE_REGISTRY',
    'LABEL_TARGET_COLUMNS',
    'TARGET_LABEL_REGISTRY',
    'TargetLabelEntry',
    'get_target_column',
]
