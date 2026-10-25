import json
from collections.abc import Collection, Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import cast

import limen.experiment._round_storage as _storage

from limen.experiment._objective_run import run_objective
from limen.experiment._boolean_controls import control_domain

__all__: list[str] = []

_BOOKKEEPING = frozenset({
    'id', 'execution_time', 'strict_mode_error', '_warnings',
    '_convergence_warning', '_dropped_features', '_param_hash', '_id',
    '_round_index', '_injected', '_generation_index', '_search_strategy',
})
_ARTIFACTS = frozenset({'extras', 'models', '_model', '_preds'})


def _reject(collisions: Mapping[str, Collection[str]]) -> None:
    details = [f'{source}: {", ".join(sorted(names))}'
               for source, names in collisions.items() if names]
    if details:
        raise ValueError('UEL field collisions — ' + '; '.join(details))


def validate_parameters(domain: Mapping[str, list[object]], context: Mapping[str, object] | None,
                        manifest: object, *, legacy_details: bool) -> None:
    control_domain(manifest, domain, context)
    reserved = set(_BOOKKEEPING)
    objective = run_objective(manifest)
    if objective is not None:
        reserved.add(objective.column)
    if legacy_details:
        reserved.add('_experiment_details')
    _reject({'domain': reserved.intersection(domain),
             'context': reserved.intersection(context or {})})


def bookkeeping_names(result: Mapping[str, object]) -> frozenset[str]:
    return _BOOKKEEPING.intersection(result)


def validate_result(parameters: Mapping[str, object], result: Mapping[str, object],
                    *, succeeded: bool) -> None:
    if succeeded:
        _reject({'result bookkeeping': bookkeeping_names(result),
                 'round parameters/result': parameters.keys() & (result.keys() - _ARTIFACTS)})


def validate_saved_successes(path: Path, expected: Sequence[tuple[int, str]], start_round: int) -> None:
    _storage.validate_prefix(path, expected)
    if not path.exists():
        return
    successful = set(expected)
    with path.open('rb') as stream:
        for raw, end in islice(_storage.frames(stream, path, tail=True), len(expected), None):
            try:
                value = cast(object, json.loads(raw))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if path.suffix != '.gz' and not raw.endswith(b'\n'):
                    return
                raise _storage.RoundStorageError(f'Malformed JSON in {path} ending at byte {end}') from exc
            if not isinstance(value, dict):
                raise _storage.RoundStorageError(f'Round must be an object in {path} ending at byte {end}')
            record = cast(dict[str, object], value)
            index, identity = record.get('_round_index'), record.get('round_id')
            if type(index) is int and 0 <= index < start_round and (not isinstance(identity, str) or (index, identity) not in successful):
                raise _storage.RoundStorageError(f'Round evidence has no successful CSV occurrence in {path}: {(index, identity)!r}')
