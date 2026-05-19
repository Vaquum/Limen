import importlib
from typing import Any

from limen.yaml.errors import ResolutionError


ALLOWED_NAMESPACES = [
    'limen.calibration',
    'limen.data',
    'limen.experiment',
    'limen.features',
    'limen.indicators',
    'limen.metrics',
    'limen.scalers',
    'limen.sfd.reference_architecture',
    'limen.sfd.rule_based',
    'limen.targets',
    'limen.transforms',
]


def resolve(path: str) -> Any:

    '''
    Resolve a dotted path string to a Python object.

    Args:
        path (str): Fully qualified dotted path, e.g. 'limen.indicators.roc'

    Returns:
        Any: The resolved Python object (callable, class, or module)

    Raises:
        ResolutionError: If the path is not in an allowed namespace or cannot be imported

    '''

    if not any(path == ns or path.startswith(ns + '.') for ns in ALLOWED_NAMESPACES):
        raise ResolutionError(path, ALLOWED_NAMESPACES)

    # Try progressively shorter module paths to support class method paths
    # e.g. limen.data.HistoricalData.get_spot_klines →
    #      import limen.data, get HistoricalData, get get_spot_klines
    parts = path.split('.')
    for split_at in range(len(parts) - 1, 0, -1):
        module_path = '.'.join(parts[:split_at])
        attrs = parts[split_at:]
        try:
            obj = importlib.import_module(module_path)
            for attr in attrs:
                obj = getattr(obj, attr)
            return obj
        except (ImportError, AttributeError):
            continue

    raise ResolutionError(path, ALLOWED_NAMESPACES)


def is_resolvable(path: str) -> bool:

    '''Return True if path can be resolved without raising.'''

    try:
        resolve(path)
        return True
    except ResolutionError:
        return False
