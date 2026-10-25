from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from itertools import islice
from pathlib import Path
from typing import cast

import polars as pl

from limen.data.historical_data import HistoricalData
import limen.experiment._round_storage as _storage
from limen.experiment._holdout_history import HoldoutEvidenceError, ProjectBinding, check_binding, research_record

__all__: list[str] = []

_FIELDS = ('schema_version', 'project_id', 'reservation_revision', 'cutoff_ns', 'interval_seconds')
_SOURCE_FIELDS = ('admitted_sha256', 'admitted', 'interval_seconds')


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HoldoutEvidenceError('Protected research population lacks recorded evidence')
    return cast(Mapping[str, object], value)


def binding_header(binding: ProjectBinding | None) -> dict[str, object] | None:
    if binding is None:
        return None
    return {'schema_version': 1, 'project_id': binding.project_id,
            'reservation_revision': binding.revision, 'cutoff_ns': binding.cutoff_ns,
            'interval_seconds': binding.interval_seconds}


def metadata_binding(binding: ProjectBinding | None) -> dict[str, object]:
    return {} if binding is None else {'holdout_binding': binding_header(binding)}


def _same_header(proof: Mapping[str, object], binding: ProjectBinding) -> bool:
    return set(_FIELDS) <= proof.keys() and json.dumps({key: proof[key] for key in _FIELDS}, sort_keys=True) == json.dumps(binding_header(binding), sort_keys=True)


def _parent(binding: ProjectBinding, identity: object, round_id: str | None = None) -> str:
    if not isinstance(identity, str):
        raise HoldoutEvidenceError('Protected population requires genuine experiment parent history')
    parent = research_record(binding, identity)
    if parent['role'] != 'experiment' or parent['revision'] != binding.revision:
        raise HoldoutEvidenceError('Protected population requires original experiment parent history')
    if parent['round_id'] is not None:
        if parent['round_id'] != round_id or parent['state'] != 'completed':
            raise HoldoutEvidenceError('Protected worker parent occurrence or completion changed')
        return _parent(binding, _mapping(json.loads(str(parent['provenance_json']))).get('experiment_record_id'))
    return str(parent['id'])


def completed_history(binding: ProjectBinding, proof: Mapping[str, object]) -> None:
    if not _same_header(proof, binding):
        raise HoldoutEvidenceError('Protected round binding/revision or occurrence changed')
    record_id = proof.get('research_record_id')
    if not isinstance(record_id, str):
        raise HoldoutEvidenceError('Protected round lacks an actual research history identity')
    saved = research_record(binding, record_id)
    if (saved['role'] != 'experiment' or saved['state'] != 'completed' or saved['revision'] != binding.revision
            or saved['round_id'] != proof.get('round_id') or json.loads(str(saved['provenance_json'])) != dict(proof)):
        raise HoldoutEvidenceError('Protected round and completed research history disagree')


def _record(binding: ProjectBinding, identity: str, value: object,
            source: Mapping[str, object] | None = None) -> Mapping[str, object]:
    proof = _mapping(value)
    if proof.get('round_id') != identity:
        raise HoldoutEvidenceError('Protected round binding/revision or occurrence changed')
    if not {'source', 'selected', 'splits', 'input_schema', 'model_schema'} <= proof.keys():
        raise HoldoutEvidenceError('Protected round lacks source, membership or model-input evidence')
    recorded_source = _mapping(proof['source'])
    if source is not None and any(recorded_source.get(key) != source.get(key) for key in _SOURCE_FIELDS):
        raise HoldoutEvidenceError('Protected checkpoint admitted source changed')
    completed_history(binding, proof)
    _ = _parent(binding, proof.get('experiment_record_id'), identity)
    return proof


def _rows(directory: Path, count: int) -> list[dict[str, str | None]]:
    csv_path = directory / 'results.csv'
    if count == 0 and not csv_path.exists():
        return []
    with csv_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames) or not {'id', '_round_index', 'strict_mode_error'} <= set(reader.fieldnames):
            raise HoldoutEvidenceError('Protected checkpoint has an incomplete CSV schema')
        rows = list(islice(reader, count))
    if len(rows) != count or any(not row.get('id') or row.get('_round_index') != str(index)
                                 or None in row or any(value is None for value in row.values()) for index, row in enumerate(rows)):
        raise HoldoutEvidenceError('Protected checkpoint has an incomplete terminal CSV population')
    return rows


def validate_live(manifest: object, raw: pl.DataFrame, binding: ProjectBinding,
                  entry: Mapping[str, object]) -> None:
    from limen.experiment._holdout_folds import live
    live(manifest, raw, binding, entry)


def _population(directory: Path, frontier: int, binding: ProjectBinding, source: Mapping[str, object] | None = None,
                manifest: object = None, raw: pl.DataFrame | None = None) -> dict[str, object]:
    rows = _rows(directory, frontier + 1)
    expected = [(index, str(row['id'])) for index, row in enumerate(rows) if not row['strict_mode_error']]
    path = _storage.round_path(directory)
    _storage.validate_prefix(path, expected)
    completed: list[dict[str, object]] = []
    if expected:
        with _storage.open_text(path, limit=len(expected)) as stream:
            for (index, identity), line in zip(expected, stream, strict=True):
                entry = _mapping(json.loads(line))
                if type(entry.get('_round_index')) is not int or entry.get('_round_index') != index or entry.get('round_id') != identity:
                    raise HoldoutEvidenceError('Protected checkpoint round occurrence differs from CSV')
                proof: Mapping[str, object] = _record(binding, identity, entry.get('holdout'), source) if binding.cutoff_ns is not None else {}
                if binding.cutoff_ns is None and 'holdout' in entry:
                    raise HoldoutEvidenceError('Protected round cannot acquire an unprotected binding/revision')
                if proof and raw is not None:
                    validate_live(manifest, raw, binding, entry)
                completed.append({'index': index, 'round_id': identity, 'research_record_id': proof.get('research_record_id'), 'holdout': dict(proof)})
    terminal = [{'index': index, 'round_id': row['id'], 'failed': bool(row['strict_mode_error'])} for index, row in enumerate(rows)]
    return {'source': source, 'prefix': terminal, 'completed': completed}


def checkpoint_state(directory: Path, frontier: int, binding: ProjectBinding | None,
                     source: Mapping[str, object] | None = None) -> dict[str, object] | None:
    if binding is None:
        return None
    check_binding(binding)
    return {'binding': binding_header(binding), 'population': _population(directory, frontier, binding, source)}


def validate_checkpoint(directory: Path, checkpoint: Mapping[str, object], metadata: Mapping[str, object],
                        binding: ProjectBinding | None, source: Mapping[str, object] | None = None,
                        manifest: object = None, raw: pl.DataFrame | None = None) -> None:
    if 'holdout_binding' not in checkpoint and 'holdout_binding' not in metadata:
        if (binding is not None and binding.cutoff_ns is not None) or 'holdout' in metadata or 'holdout_population' in checkpoint:
            raise HoldoutEvidenceError('Protected resume requires the original checkpoint population binding')
        path = _storage.round_path(directory)
        if path.exists():
            with _storage.open_text(path, limit=len(_storage.csv_prefix(path, cast(int, _mapping(checkpoint['metadata'])['experiment_round']) + 1))) as stream:
                if any('holdout' in _mapping(json.loads(line)) for line in stream):
                    raise HoldoutEvidenceError('Protected resume requires the original checkpoint population binding')
        return
    if binding is None or any(not isinstance(item, Mapping) or dict(cast(Mapping[str, object], item)) != binding_header(binding)
                              or not _same_header(cast(Mapping[str, object], item), binding) for item in (checkpoint.get('holdout_binding'), metadata.get('holdout_binding'))):
        raise HoldoutEvidenceError('Protected checkpoint binding/revision changed or its envelope is missing')
    check_binding(binding)
    validate_trainer(binding, metadata, {})
    frontier = _mapping(checkpoint.get('metadata')).get('experiment_round')
    if type(frontier) is not int or frontier < -1:
        raise HoldoutEvidenceError('Protected checkpoint lacks a terminal round frontier')
    if checkpoint.get('holdout_population') != _population(directory, frontier, binding, source, manifest, raw):
        raise HoldoutEvidenceError('Protected checkpoint research population differs from its saved history')


def validate_trainer(binding: ProjectBinding | None, metadata: Mapping[str, object], rounds: Mapping[str, object],
                     source: Mapping[str, object] | None = None, manifest: object = None, raw: pl.DataFrame | None = None) -> None:
    saved = metadata.get('holdout')
    if binding is None or binding.cutoff_ns is None:
        if (saved is not None or any('holdout' in _mapping(entry) for entry in rounds.values())
                or ('holdout_binding' in metadata and metadata['holdout_binding'] != binding_header(binding))):
            raise HoldoutEvidenceError('Protected Trainer metadata binding/revision changed')
        return
    if not isinstance(saved, Mapping) or not _same_header(cast(Mapping[str, object], saved), binding):
        raise HoldoutEvidenceError('Protected Trainer requires original saved project/population evidence')
    if 'holdout_binding' in metadata and metadata['holdout_binding'] != binding_header(binding):
        raise HoldoutEvidenceError('Protected Trainer metadata binding/revision changed')
    _ = _parent(binding, cast(Mapping[str, object], saved).get('research_record_id'))
    for identity, entry in rounds.items():
        record = _mapping(entry)
        _ = _record(binding, identity, record.get('holdout'), source)
        if raw is not None:
            validate_live(manifest, raw, binding, record)


def validate_trainer_round(binding: ProjectBinding, identity: object, proof: object) -> str:
    if not isinstance(identity, str):
        raise HoldoutEvidenceError('Protected Trainer round lacks completed experiment evidence')
    return cast(str, _record(binding, identity, proof)['research_record_id'])


def trainer_source(manifest: object, rounds: Mapping[str, object]) -> dict[str, object] | None:
    """Reload the actual local source of SQL-validated legacy Python rounds."""
    if not rounds:
        return None
    source = _mapping(_mapping(_mapping(next(iter(rounds.values()))).get('holdout')).get('source'))
    if source.get('kind') == 'local_file':
        configure = cast(Callable[..., object], getattr(manifest, 'set_data_source', None))
        _ = configure(HistoricalData.get_any_file, dict(_mapping(source['identity'])))
        return {'method': 'limen.data.HistoricalData.get_any_file', 'params': dict(_mapping(source['identity']))}
    return None
