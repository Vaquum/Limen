from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from typing import cast

import limen.experiment._round_storage as _storage
from limen.experiment._holdout_history import ProjectBinding, check_binding, research_record

__all__: list[str] = []

_FIELDS = ('schema_version', 'project_id', 'reservation_revision', 'cutoff_ns', 'interval_seconds')


def binding_header(binding: ProjectBinding | None) -> dict[str, object] | None:
    if binding is None or binding.project_id is None:
        return None
    return {'schema_version': 1, 'project_id': binding.project_id,
            'reservation_revision': binding.revision, 'cutoff_ns': binding.cutoff_ns,
            'interval_seconds': binding.interval_seconds}


def add_metadata(metadata: dict[str, object], binding: ProjectBinding | None) -> None:
    if (header := binding_header(binding)) is not None:
        metadata['holdout_binding'] = header


def _record(binding: ProjectBinding, identity: str, value: object) -> str:
    if not isinstance(value, Mapping):
        raise ValueError('Protected round is missing its saved holdout evidence')
    proof = cast(Mapping[str, object], value)
    header = binding_header(binding)
    if header is None or any(proof.get(key) != header[key] for key in _FIELDS):
        raise ValueError('Protected round binding/revision differs from its research population')
    if proof.get('schema_version') != 1 or proof.get('round_id') != identity:
        raise ValueError('Protected round identity differs from its recorded occurrence')
    if not {'source', 'selected', 'splits', 'input_schema', 'model_schema'} <= proof.keys():
        raise ValueError('Protected round lacks source, membership or model-input evidence')
    record_id = proof.get('research_record_id')
    if not isinstance(record_id, str):
        raise ValueError('Protected round lacks an actual research history identity')
    saved = research_record(binding, record_id)
    if (saved['role'] != 'experiment' or saved['state'] != 'completed'
            or saved['revision'] != binding.revision or saved['round_id'] != identity
            or json.loads(str(saved['provenance_json'])) != dict(proof)):
        raise ValueError('Protected round and completed research history disagree')
    return hashlib.sha256(json.dumps(proof, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _rows(directory: Path, count: int) -> list[dict[str, str | None]]:
    csv_path = directory / 'results.csv'
    if count == 0 and not csv_path.exists():
        return []
    if not csv_path.is_file():
        raise ValueError('Protected checkpoint lacks its committed CSV population')
    with csv_path.open(newline='') as stream:
        rows = list(islice(csv.DictReader(stream), count))
    if len(rows) != count or any(not row.get('id') or row.get('_round_index') != str(index)
                                 or 'strict_mode_error' not in row for index, row in enumerate(rows)):
        raise ValueError('Protected checkpoint has an incomplete terminal CSV population')
    return rows


def _population(directory: Path, frontier: int, binding: ProjectBinding) -> dict[str, object]:
    rows = _rows(directory, frontier + 1)
    expected = [(index, str(row['id'])) for index, row in enumerate(rows) if not row['strict_mode_error']]
    path = _storage.round_path(directory)
    _storage.validate_prefix(path, expected)
    completed: list[dict[str, object]] = []
    if expected:
        with _storage.open_text(path, limit=len(expected)) as stream:
            for (index, identity), line in zip(expected, stream, strict=True):
                entry = json.loads(line)
                if not isinstance(entry, dict) or entry.get('_round_index') != index or entry.get('round_id') != identity:
                    raise ValueError('Protected checkpoint round occurrence differs from CSV')
                proof = entry.get('holdout')
                completed.append({'index': index, 'round_id': identity,
                                  'research_record_id': proof.get('research_record_id') if isinstance(proof, dict) else None,
                                  'holdout_sha256': _record(binding, identity, proof) if binding.cutoff_ns is not None else None})
    return {'prefix': [{'index': index, 'id': row['id'], 'strict_mode_error': row['strict_mode_error']}
                       for index, row in enumerate(rows)], 'completed': completed}


def checkpoint_state(directory: Path, frontier: int, binding: ProjectBinding | None) -> dict[str, object] | None:
    if (header := binding_header(binding)) is None or binding is None:
        return None
    check_binding(binding)
    return {'binding': header, 'population': _population(directory, frontier, binding)}


def validate_checkpoint(directory: Path, checkpoint: Mapping[str, object],
                        metadata: Mapping[str, object], binding: ProjectBinding | None) -> None:
    header = binding_header(binding)
    claimed = 'holdout_binding' in checkpoint or 'holdout_binding' in metadata
    if not claimed:
        if header is not None and header['cutoff_ns'] is not None or 'holdout' in metadata:
            raise ValueError('Protected resume requires the original checkpoint population binding')
        return
    if binding is None or header is None or checkpoint.get('holdout_binding') != header or metadata.get('holdout_binding') != header:
        raise ValueError('Protected checkpoint binding/revision changed or its envelope is missing')
    check_binding(binding)
    parent = metadata.get('holdout')
    if binding.cutoff_ns is not None:
        if not isinstance(parent, Mapping) or any(parent.get(key) != header[key] for key in _FIELDS):
            raise ValueError('Protected checkpoint lacks its original experiment parent')
        parent_id = parent.get('research_record_id')
        if not isinstance(parent_id, str) or research_record(binding, parent_id)['role'] != 'experiment':
            raise ValueError('Protected checkpoint has no genuine experiment history parent')
    state = checkpoint.get('metadata')
    if not isinstance(state, Mapping) or type(state.get('experiment_round')) is not int:
        raise ValueError('Protected checkpoint lacks a terminal round frontier')
    if checkpoint.get('holdout_population') != _population(directory, cast(int, state['experiment_round']), binding):
        raise ValueError('Protected checkpoint research population differs from its saved history')


def validate_trainer(binding: ProjectBinding | None, metadata: Mapping[str, object],
                     rounds: Mapping[str, object]) -> None:
    saved = metadata.get('holdout')
    if binding is None or binding.cutoff_ns is None:
        if saved is not None or 'holdout_binding' in metadata:
            if binding_header(binding) != metadata.get('holdout_binding') or saved is not None:
                raise ValueError('Protected Trainer metadata binding/revision changed')
        return
    header = binding_header(binding)
    if not isinstance(saved, Mapping) or header is None or any(saved.get(key) != header[key] for key in _FIELDS):
        raise ValueError('Protected Trainer requires original saved project/population evidence')
    if 'holdout_binding' in metadata and metadata['holdout_binding'] != header:
        raise ValueError('Protected Trainer metadata binding/revision changed')
    parent_id = saved.get('research_record_id')
    if not isinstance(parent_id, str) or research_record(binding, parent_id)['role'] != 'experiment':
        raise ValueError('Protected Trainer requires genuine experiment parent history')
    for identity, entry in rounds.items():
        if not isinstance(entry, Mapping):
            raise ValueError('Protected Trainer round lacks recorded evidence')
        _ = _record(binding, identity, entry.get('holdout'))


def validate_trainer_round(binding: ProjectBinding, identity: object, proof: object) -> str:
    if not isinstance(identity, str) or not isinstance(proof, Mapping):
        raise ValueError('Protected Trainer round lacks completed experiment evidence')
    _ = _record(binding, identity, proof)
    record_id = proof.get('research_record_id')
    if not isinstance(record_id, str):
        raise ValueError('Protected Trainer requires genuine round lineage')
    return record_id
