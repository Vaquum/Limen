import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from limen.experiment._acceptance_settings import prepare_pbo_settings
from limen.experiment._parallel_evidence import project_record

__all__ = ['project_record']

_DEFAULTS: dict[str, bool] = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), False)


def parse_recording(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise ValueError('recording must be a mapping of payload names to booleans')
    policy = dict(_DEFAULTS)
    for key, enabled in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or key not in policy:
            raise ValueError(f'Unknown recording payload {key!r}')
        if type(enabled) is not bool:
            raise ValueError(f'recording.{key} must be a bool, not a parameter reference or coerced value')
        policy[key] = enabled
    return policy


def _configured(reference: Mapping[str, object] | None) -> dict[str, bool] | None:
    if reference is None:
        return None
    configuration = reference.get('uel', {})
    if not isinstance(configuration, Mapping):
        raise ValueError('uel must be a mapping')
    settings = cast(Mapping[str, object], configuration)
    return parse_recording(settings['recording']) if 'recording' in settings else None


def validate_pbo_recording(policy: Mapping[str, bool] | None,
                           acceptance: Mapping[str, object] | None) -> None:
    if acceptance is not None and bool({'max_pbo', 'pbo_blocks'} & acceptance.keys()) and policy is not None and (not policy['alignment'] or not policy['fold_details']):
        raise ValueError('PBO requires recording.alignment and recording.fold_details; omit explicit PBO settings or retain their evidence')


def validate_recorded_pbo(directory: Path, acceptance: Mapping[str, object]) -> None:
    metadata_path = directory / 'metadata.json'
    if not {'max_pbo', 'pbo_blocks'} & acceptance.keys() or not metadata_path.exists():
        return
    metadata: object = json.loads(metadata_path.read_text())
    if not isinstance(metadata, Mapping):
        raise ValueError('PBO requires valid recording metadata')
    metadata = cast(Mapping[str, object], metadata)
    if 'recording' in metadata:
        validate_pbo_recording(parse_recording(metadata['recording']), acceptance)


def prepare_recording(value: object, *, reference: Mapping[str, object] | None,
                      compiled_reference: Mapping[str, object] | None,
                      directory: Path | None, has_search: bool,
                      resume: bool, acceptance: Mapping[str, object] | None = None) -> dict[str, bool] | None:
    acceptance = prepare_pbo_settings(acceptance, reference=reference, compiled_reference=compiled_reference, directory=directory, has_search=has_search, resume=resume)
    if compiled_reference is not None and reference is not None and (_configured(compiled_reference) or _DEFAULTS) != (_configured(reference) or _DEFAULTS):
        raise ValueError('Conflicting compiled YAML and yaml_reference recording declarations')
    reference = compiled_reference if compiled_reference is not None else reference
    configured = _configured(reference)
    declared = value if value is not None else configured

    saved: dict[str, bool] | None = None
    metadata_path = None if directory is None else directory / 'metadata.json'
    if resume and metadata_path is not None and metadata_path.exists():
        metadata: object = json.loads(metadata_path.read_text())
        if not isinstance(metadata, Mapping):
            raise ValueError('Cannot resume: metadata must be a mapping')
        metadata = cast(Mapping[str, object], metadata)
        saved = parse_recording(metadata['recording']) if 'recording' in metadata else dict.fromkeys(_DEFAULTS, True)
        original = metadata.get('yaml_reference')
        if reference is not None and isinstance(original, Mapping):
            if (_configured(cast(Mapping[str, object], original)) or _DEFAULTS) != (configured or _DEFAULTS):
                raise ValueError('Cannot resume with a different YAML recording declaration')
        elif configured is not None and configured != saved:
            raise ValueError('Cannot resume with a different recording policy')
        if value is None and (configured is None or 'recording' in metadata):
            declared = saved

    policy = parse_recording(declared) if declared is not None else dict(_DEFAULTS)
    validate_pbo_recording(policy if has_search and directory is not None else None, acceptance)
    if (value is not None or configured is not None) and (not has_search or directory is None):
        raise ValueError('recording requires search_strategy and experiment_dir')
    if resume and has_search and directory is not None:
        if saved is not None and saved != policy:
            raise ValueError('Cannot resume with a different recording policy')
        if not all(policy.values()):
            raise ValueError('Cannot resume a lighter recording policy: complete recorded recovery evidence is required')
    return policy if has_search and directory is not None else None


