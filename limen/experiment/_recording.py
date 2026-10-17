import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

__all__: list[str] = []

_DEFAULTS = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)


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


def prepare_recording(value: object, *, reference: Mapping[str, object] | None,
                      directory: Path | None, has_search: bool,
                      resume: bool) -> dict[str, bool] | None:
    declared = value
    explicit = value is not None
    if not explicit and reference is not None:
        configuration = reference.get('uel', {})
        if not isinstance(configuration, Mapping):
            raise ValueError('uel must be a mapping')
        settings = cast(Mapping[str, object], configuration)
        if 'recording' in settings:
            declared, explicit = settings['recording'], True

    saved: dict[str, bool] | None = None
    metadata_path = None if directory is None else directory / 'metadata.json'
    if resume and metadata_path is not None and metadata_path.exists():
        metadata: object = json.loads(metadata_path.read_text())
        if not isinstance(metadata, Mapping):
            raise ValueError('Cannot resume: metadata must be a mapping')
        metadata = cast(Mapping[str, object], metadata)
        saved = parse_recording(metadata['recording']) if 'recording' in metadata else dict(_DEFAULTS)
        if value is None and 'recording' in metadata:
            declared, explicit = metadata['recording'], True

    policy = parse_recording(declared) if explicit else dict(_DEFAULTS)
    if explicit and (not has_search or directory is None):
        raise ValueError('recording requires search_strategy and experiment_dir')
    if resume:
        if saved is not None and saved != policy:
            raise ValueError('Cannot resume with a different recording policy')
        if not all(policy.values()):
            raise ValueError('Cannot resume a lighter recording policy: complete recorded recovery evidence is required')
    return policy if explicit else None


def project_record(record: Mapping[str, object], policy: Mapping[str, bool] | None) -> dict[str, object]:
    if policy is None or all(policy.values()):
        return dict(record)
    excluded = {field for setting, field in (
        ('predictions', 'preds'), ('trade_ledger', 'trade_ledger'), ('fold_details', 'folds'),
    ) if not policy[setting]}
    projected = {key: value for key, value in record.items() if key not in excluded}
    if not policy['alignment'] and 'alignment' in projected:
        alignment = cast(Mapping[str, object], projected['alignment'])
        projected['alignment'] = {key: value for key, value in alignment.items()
                                  if key not in ('missing_datetimes', 'test_datetimes')}
    if 'folds' in projected:
        projected['folds'] = [project_record(fold, policy)
                              for fold in cast(list[Mapping[str, object]], projected['folds'])]
    return projected
