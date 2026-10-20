import importlib
from types import ModuleType
from typing import Any

from limen.yaml.errors import ResolutionError


ALLOWED_NAMESPACES = [
    'limen.backtest',
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


def _allowed_namespace(name: str) -> bool:
    return any(name == ns or name.startswith(ns + '.') for ns in ALLOWED_NAMESPACES)


def resolve(path: str) -> Any:

    '''Resolve a dotted path without leaving the allowed namespaces.'''

    parts = path.split('.')
    if not _allowed_namespace(path) or any(part.startswith('__') for part in parts):
        raise ResolutionError(path, ALLOWED_NAMESPACES)

    # Try progressively shorter module paths to support class method paths
    # e.g. limen.data.HistoricalData.get_spot_klines →
    #      import limen.data, get HistoricalData, get get_spot_klines
    for split_at in range(len(parts) - 1, 0, -1):
        module_path = '.'.join(parts[:split_at])
        attrs = parts[split_at:]
        try:
            obj = importlib.import_module(module_path)
            if not (_allowed_namespace(obj.__name__) or any(ns.startswith(obj.__name__ + '.') for ns in ALLOWED_NAMESPACES)):
                raise ResolutionError(path, ALLOWED_NAMESPACES)
            for attr in attrs:
                obj = getattr(obj, attr)
                namespace = obj.__name__ if isinstance(obj, ModuleType) else (getattr(obj, '__module__', '') or '')
                if not _allowed_namespace(namespace):
                    raise ResolutionError(path, ALLOWED_NAMESPACES)
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
