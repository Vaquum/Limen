from collections.abc import Mapping
from typing import cast

from ruamel.yaml.scalarbool import ScalarBoolean

from limen.experiment._boolean_controls import _boolean, _CALIBRATION_CONTROLS
from limen.yaml.errors import YAMLError

__all__: list[str] = []


def _mapping(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _normalize_controls(config: Mapping[str, object]) -> None:
    sfd = _mapping(config.get('sfd'))
    manifest = _mapping(sfd.get('manifest'))
    if manifest.get('type') != 'ml':
        return
    if isinstance(manifest.get('strict_mode'), ScalarBoolean):
        manifest['strict_mode'] = bool(manifest['strict_mode'])
    params = _mapping(sfd.get('params'))
    for name in _CALIBRATION_CONTROLS:
        values = params.get(name)
        if isinstance(values, list):
            params[name] = [bool(value) if isinstance(value, ScalarBoolean) else value for value in cast(list[object], values)]


class BooleanControlsSpec:
    def check(self, config: dict[str, object], errors: list[YAMLError], _warnings: list[YAMLError]) -> None:
        _normalize_controls(config)
        sfd = _mapping(config.get('sfd'))
        manifest = _mapping(sfd.get('manifest'))
        if manifest.get('type') != 'ml':
            return
        controls = [('sfd.manifest.strict_mode', manifest['strict_mode'])] if 'strict_mode' in manifest else []
        params = _mapping(sfd.get('params'))
        for name in _CALIBRATION_CONTROLS:
            values = params.get(name)
            if isinstance(values, list):
                controls.extend((f'sfd.params.{name}[{index}]', value) for index, value in enumerate(cast(list[object], values)))
        for path, value in controls:
            try:
                _ = _boolean(value, path)
            except ValueError as exc:
                errors.append(YAMLError(message=str(exc), path=path))
