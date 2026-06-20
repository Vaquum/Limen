from __future__ import annotations

from importlib import import_module
from importlib import metadata
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def _resolve_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    if pyproject.exists():
        project = tomllib.loads(pyproject.read_text(encoding='utf-8'))['project']
        return str(project['version'])
    try:
        return metadata.version('vaquum-limen')
    except metadata.PackageNotFoundError:  # pragma: no cover - broken install metadata
        return '0+unknown'


__version__ = _resolve_version()


_LAZY_EXPORTS = {
    'Cohort': ('limen.cohort', 'Cohort'),
    'HistoricalData': ('limen.data', 'HistoricalData'),
    'Log': ('limen.log.log', 'Log'),
    'MLManifest': ('limen.experiment', 'MLManifest'),
    'Manifest': ('limen.experiment', 'Manifest'),
    'ReconstructionError': ('limen.inference', 'ReconstructionError'),
    'RuleBasedManifest': ('limen.experiment', 'RuleBasedManifest'),
    'Sensor': ('limen.inference', 'Sensor'),
    'Trainer': ('limen.inference', 'Trainer'),
    'UniversalExperimentLoop': ('limen.experiment', 'UniversalExperimentLoop'),
    'features': ('limen.features', None),
    'indicators': ('limen.indicators', None),
    'log': ('limen.log', None),
    'metrics': ('limen.metrics', None),
    'scalers': ('limen.scalers', None),
    'sfd': ('limen.sfd', None),
    'targets': ('limen.targets', None),
    'transforms': ('limen.transforms', None),
    'utils': ('limen.utils', None),
}


__all__ = [
    'Cohort',
    'HistoricalData',
    'Log',
    'MLManifest',
    'Manifest',
    'ReconstructionError',
    'RuleBasedManifest',
    'Sensor',
    'Trainer',
    'UniversalExperimentLoop',
    '__version__',
    'features',
    'indicators',
    'log',
    'metrics',
    'scalers',
    'sfd',
    'targets',
    'transforms',
    'utils',
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'limen' has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
