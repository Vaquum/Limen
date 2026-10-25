from collections.abc import Mapping

__all__: list[str] = []
_CALIBRATION_CONTROLS = ('use_calibration', 'use_threshold')


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f'{name} must be a bool; got {value!r}')
    return value


def _calibration_controls(params: Mapping[str, object]) -> tuple[bool, bool]:
    return (_boolean(params.get('use_calibration', True), 'use_calibration'),
            _boolean(params.get('use_threshold', True), 'use_threshold'))


def _manifest_controls(manifest: object, params: Mapping[str, object]) -> tuple[bool, bool]:
    from limen.experiment.manifest_core import MLManifest
    if isinstance(manifest, MLManifest):
        _ = _boolean(manifest.strict_mode, 'strict_mode')
        return _calibration_controls(params)
    return True, True


def _control_domain(manifest: object, domain: Mapping[str, list[object]], context: Mapping[str, object] | None) -> None:
    from limen.experiment.manifest_core import MLManifest
    if isinstance(manifest, MLManifest):
        _ = _manifest_controls(manifest, context or {})
        for name in _CALIBRATION_CONTROLS:
            for value in domain.get(name, []):
                _ = _boolean(value, name)
