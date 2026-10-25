from collections.abc import Mapping
from importlib import import_module
from typing import cast

__all__: list[str] = []
CALIBRATION_CONTROLS = ('use_calibration', 'use_threshold')


def boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f'{name} must be a bool; got {value!r}')
    return value


def calibration_controls(params: Mapping[str, object]) -> tuple[bool, bool]:
    return (boolean(params.get('use_calibration', True), 'use_calibration'),
            boolean(params.get('use_threshold', True), 'use_threshold'))


def manifest_controls(manifest: object, params: Mapping[str, object]) -> tuple[bool, bool]:
    kind = cast(type[object], vars(import_module('limen.experiment.manifest_core'))['MLManifest'])
    if isinstance(manifest, kind):
        _ = boolean(vars(manifest)['strict_mode'], 'strict_mode')
        return calibration_controls(params)
    return True, True


def control_domain(manifest: object, domain: Mapping[str, list[object]], context: Mapping[str, object] | None) -> None:
    kind = cast(type[object], vars(import_module('limen.experiment.manifest_core'))['MLManifest'])
    if isinstance(manifest, kind):
        _ = manifest_controls(manifest, context or {})
        for name in CALIBRATION_CONTROLS:
            for value in domain.get(name, []):
                _ = boolean(value, name)
