from collections.abc import Mapping
from typing import cast

from limen.experiment._recording import parse_recording, validate_pbo_recording
from limen.yaml.errors import YAMLError

__all__ = ['RecordingSpec']


class RecordingSpec:
    def check(self, yaml_dict: dict[str, object], errors: list[YAMLError], _warnings: list[YAMLError]) -> None:
        sfd = yaml_dict.get('sfd')
        manifest = cast(Mapping[str, object], sfd).get('manifest') if isinstance(sfd, Mapping) else None
        containers: tuple[tuple[str, object], ...] = (('recording', yaml_dict), ('sfd.recording', sfd), ('sfd.manifest.recording', manifest))
        for path, container in containers:
            if isinstance(container, Mapping) and 'recording' in container:
                errors.append(YAMLError(message='Declare recording only under uel', path=path))
        uel = yaml_dict.get('uel')
        if not isinstance(uel, Mapping) or 'recording' not in uel:
            return
        try:
            policy = parse_recording(cast(Mapping[str, object], uel)['recording'])
            raw_acceptance = cast(Mapping[str, object], manifest).get('acceptance') if isinstance(manifest, Mapping) else None
            acceptance = cast(Mapping[str, object], raw_acceptance) if isinstance(raw_acceptance, Mapping) else None
            validate_pbo_recording(policy, acceptance)
        except (TypeError, ValueError) as exc:
            errors.append(YAMLError(message=str(exc), path='uel.recording'))
