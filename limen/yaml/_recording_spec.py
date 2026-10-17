from collections.abc import Mapping
from typing import cast

from limen.experiment._recording import parse_recording
from limen.yaml.errors import YAMLError

__all__ = ['RecordingSpec']


class RecordingSpec:
    def check(self, yaml_dict: dict[str, object], errors: list[YAMLError], _warnings: list[YAMLError]) -> None:
        sfd = yaml_dict.get('sfd')
        manifest = cast(Mapping[str, object], sfd).get('manifest') if isinstance(sfd, Mapping) else None
        for path, container in (('recording', yaml_dict), ('sfd.recording', sfd), ('sfd.manifest.recording', manifest)):
            if isinstance(container, Mapping) and 'recording' in container:
                errors.append(YAMLError(message='Declare recording only under uel', path=path))
        uel = yaml_dict.get('uel')
        if not isinstance(uel, Mapping) or 'recording' not in uel:
            return
        try:
            _ = parse_recording(cast(Mapping[str, object], uel)['recording'])
        except (TypeError, ValueError) as exc:
            errors.append(YAMLError(message=str(exc), path='uel.recording'))
