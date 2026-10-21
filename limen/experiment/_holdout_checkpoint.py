from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
from typing import cast

import polars as pl

from limen.experiment import _holdout_history as history
from limen.experiment import _round_storage as storage

__all__: list[str] = []
_HEADER = {'schema_version', 'project_id', 'reservation_revision', 'cutoff_ns', 'interval_seconds'}
_BINDING_SINCE = (5, 45, 27)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _binding(manifest: object) -> history.ProjectBinding | None:
    return cast(history.ProjectBinding | None, getattr(manifest, '_holdout_binding', None))


def _legacy(metadata: Mapping[str, object]) -> bool:
    version = metadata.get('limen_version')
    parts = version.split('.') if isinstance(version, str) else []
    return len(parts) == 3 and all(part.isdigit() for part in parts) and tuple(map(int, parts)) < _BINDING_SINCE


def header(manifest: object) -> dict[str, object] | None:
    binding = _binding(manifest)
    if binding is None:
        return None
    history.check_binding(binding)
    return {'schema_version': 1, 'project_id': binding.project_id, 'reservation_revision': binding.revision,
            'cutoff_ns': binding.cutoff_ns, 'interval_seconds': binding.interval_seconds}


def trainer_metadata(manifest: object, metadata: Mapping[str, object]) -> object:
    if 'holdout_binding' not in metadata and header(manifest) is not None and not _legacy(metadata):
        raise ValueError('Holdout Trainer metadata lacks its saved checkpoint binding')
    if 'holdout_binding' in metadata and _json(metadata['holdout_binding']) != _json(header(manifest)):
        raise ValueError('Holdout Trainer checkpoint binding changed')
    return metadata.get('holdout')


def _proof(binding: history.ProjectBinding, row: Mapping[str, object], *, completed_parent: bool) -> dict[str, object]:
    value = row.get('holdout')
    if not isinstance(value, dict):
        raise ValueError('Holdout checkpoint/Trainer lacks a protected round witness')
    proof = cast(dict[str, object], value)
    expected = {'schema_version': 1, 'project_id': binding.project_id, 'reservation_revision': binding.revision,
                'cutoff_ns': binding.cutoff_ns, 'interval_seconds': binding.interval_seconds}
    if _json({key: proof.get(key) for key in _HEADER}) != _json(expected) or proof.get('round_id') != row.get('round_id'):
        raise ValueError('Holdout round project/revision/population association changed')
    record = history.research_record(binding, cast(str, proof['research_record_id']))
    if (record['state'] != 'completed' or record['role'] != 'experiment' or record['revision'] != binding.revision
            or record['round_id'] != row['round_id'] or _json(json.loads(cast(str, record['provenance_json']))) != _json(proof)):
        raise ValueError('Holdout requires the exact completed source experiment round')
    parent = history.research_record(binding, cast(str, proof['experiment_record_id']))
    if parent['role'] != 'experiment' or parent['round_id'] is not None or parent['revision'] != binding.revision or (completed_parent and parent['state'] != 'completed'):
        raise ValueError('Holdout source experiment parent is not completed/current')
    for key in ('source', 'selected', 'splits', 'input_schema', 'model_schema', 'schema_reason'):
        if key not in proof:
            raise ValueError('Holdout round population/schema witness is missing')
    return proof


def _rows(directory: Path, frontier: int) -> list[dict[str, object]]:
    path = storage.round_path(directory)
    expected = storage.csv_prefix(path, frontier + 1) if frontier >= 0 else []
    storage.validate_prefix(path, expected)
    if not expected:
        return []
    with storage.open_text(path, limit=len(expected)) as stream:
        return [cast(dict[str, object], json.loads(line)) for line in stream]


def _protected(row: Mapping[str, object]) -> bool:
    return 'holdout' in row or any('holdout' in fold for fold in cast(list[dict[str, object]], row.get('folds', [])))


def capture(host: object, frontier: int, *, completed_parents: bool = False) -> dict[str, object] | None:
    attrs = cast(dict[str, object], vars(host))
    manifest = attrs.get('manifest')
    current = header(manifest)
    if current is None:
        return None
    binding = _binding(manifest)
    assert binding is not None
    if type(frontier) is not int or frontier < -1 or not isinstance(data := attrs.get('data'), pl.DataFrame):
        raise ValueError('Holdout checkpoint requires a valid frontier and current admitted data')
    directory = cast(Path, attrs['_experiment_dir'])
    witnesses: list[dict[str, object]] = []
    links: list[list[object]] = []
    for row in _rows(directory, frontier):
        if binding.cutoff_ns is None:
            if _protected(row):
                raise ValueError('Protected checkpoint cannot resume after reservation release')
            continue
        entries = cast(list[dict[str, object]], row['folds']) if 'folds' in row else [row]
        for entry in entries:
            proof = _proof(binding, entry | {'round_id': row['round_id']}, completed_parent=completed_parents)
            if entry is not row:
                workflow = proof.get('workflow')
                if (not isinstance(workflow, dict) or type(entry.get('fold')) is not int
                        or _json({key: cast(dict[str, object], workflow).get(key) for key in ('schema_version', 'fold')}) != _json({'schema_version': 1, 'fold': entry['fold']})):
                    raise ValueError('Holdout checkpoint fold association/version changed')
            witnesses.append(proof)
            links.append([row['round_id'], proof['research_record_id']])
    buf = data.rechunk().write_ipc(None, compression='uncompressed')
    reference = cast(Mapping[str, object] | None, attrs.get('_compiled_yaml') or attrs.get('_yaml_reference'))
    selector = getattr(manifest, 'pre_split_data_selector', None)
    declaration = None if selector is None else {'method': f'{selector[0].__module__}.{selector[0].__qualname__}', 'params': selector[1]}
    return current | {'population': {'project_root': str(binding.root.resolve()), 'input_sha256': sha256(buf.getvalue()).hexdigest(), 'source': getattr(manifest, '_holdout_source', None), 'rounds': witnesses,
                                      'sfd': None if reference is None else reference.get('sfd'), 'selector': declaration},
                      'completed_rounds': links}


def validate(host: object, checkpoint: Mapping[str, object], metadata: Mapping[str, object]) -> None:
    attrs = cast(dict[str, object], vars(host))
    manifest = attrs.get('manifest')
    current = header(manifest)
    frontier = cast(Mapping[str, object], checkpoint['metadata'])['experiment_round']
    if type(frontier) is not int or frontier < -1:
        raise ValueError('Holdout checkpoint frontier is invalid')
    marked = 'holdout_binding' in checkpoint or 'holdout_binding' in metadata
    if not marked and 'holdout' not in metadata:
        if current is None or (current['cutoff_ns'] is None and _legacy(metadata)):
            if any(_protected(row) for row in _rows(cast(Path, attrs['_experiment_dir']), frontier)):
                raise ValueError('Protected checkpoint lost its reservation/metadata binding')
            return
    saved = checkpoint.get('holdout_binding')
    if (current is None or _json(metadata.get('holdout_binding')) != _json(current)
            or not isinstance(saved, dict) or set(cast(dict[str, object], saved)) != _HEADER | {'population', 'completed_rounds'}):
        raise ValueError('Holdout checkpoint binding/version is missing or changed')
    payload = cast(dict[str, object], saved)
    if _json({key: payload[key] for key in _HEADER}) != _json(current):
        raise ValueError('Holdout checkpoint and metadata headers disagree')
    if _json(payload) != _json(capture(host, frontier, completed_parents=True)):
        raise ValueError('Holdout checkpoint source/population/history changed')


def trainer_round(manifest: object, expected: Mapping[str, object]) -> None:
    binding = _binding(manifest)
    if binding is None or binding.cutoff_ns is None:
        if 'holdout' in expected:
            raise ValueError('Protected Trainer selection lost its reservation binding')
        return
    proof = _proof(binding, expected, completed_parent=True)
    actual = getattr(manifest, '_holdout_source', None)
    source = cast(Mapping[str, object], proof['source'])
    if not isinstance(actual, Mapping) or _json({key: cast(Mapping[str, object], actual).get(key) for key in ('kind', 'method', 'identity', 'interval_seconds', 'admitted_sha256')}) != _json({key: source.get(key) for key in ('kind', 'method', 'identity', 'interval_seconds', 'admitted_sha256')}):
        raise ValueError('Holdout Trainer source differs from its completed experiment')
