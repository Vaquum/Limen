"""Restore declared filters and refuse unattended continuation after filter loss."""

import copy
import csv
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.experiment.param_search.search_strategy import SearchStrategy
from limen.experiment.reducer.filter_types import FILTER_BUILDERS, FILTER_SAMPLE, SAMPLE_HASH_RULE

Filter = Callable[[dict[str, object]], bool]
Descriptor = tuple[str, dict[str, object]]


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f'Cannot resume with invalid {name}')
    return cast(dict[str, object], value)


def restore_filters(state: Mapping[str, object]) -> tuple[dict[str, Filter], dict[str, Descriptor], int, dict[str, str]]:
    """Build supported descriptors once for standalone restore and UEL admission."""
    custom = state.get('custom_filters_count', 0)
    keys = state.get('named_filter_keys', [])
    if type(custom) is not int or custom < 0:
        raise ValueError('Cannot resume with invalid custom filter count')
    if not isinstance(keys, list) or any(not isinstance(key, str) for key in cast(list[object], keys)):
        raise ValueError('Cannot resume with invalid named filter keys')
    descriptors = _mapping(state.get('named_filter_descriptors', {}), 'named filter descriptors')
    restored: dict[str, Filter] = {}
    definitions: dict[str, Descriptor] = {}
    lost = dict.fromkeys(cast(list[str], keys), 'no descriptor')
    for key, descriptor in descriptors.items():
        if not isinstance(key, str) or not isinstance(descriptor, (list, tuple)) or len(cast(list[object], descriptor)) != 2:
            raise ValueError('Cannot resume with invalid named filter descriptor')
        kind, parameters = cast(tuple[object, object], descriptor)
        if not isinstance(kind, str):
            raise ValueError(f'Cannot resume with invalid filter type for {key!r}')
        params = _mapping(parameters, f'filter parameters for {key!r}')
        if kind == FILTER_SAMPLE and params.get('hash_rule') != SAMPLE_HASH_RULE:
            raise ValueError('Cannot resume an unversioned or unsupported sampling hash rule; use the original package version or start a new experiment')
        builder = FILTER_BUILDERS.get(kind)
        if builder is None:
            lost[key] = f'unsupported filter type {kind!r}'
        else:
            try:
                restored[key] = builder(params)
            except (KeyError, TypeError, ValueError) as exc:
                lost[key] = f'invalid descriptor: {exc}'
            else:
                definitions[key] = (kind, dict(params))
                _ = lost.pop(key, None)
    return restored, definitions, custom, lost


def validate_filter_resume(checkpoint: Mapping[str, object], strategy: SearchStrategy,
                           csv_path: Path, *, parallel: bool) -> None:
    """Refuse lost constraints before recovery can rewrite any run artifact."""
    state = _mapping(checkpoint['msq_state'], 'MSQ state')
    restored, _, custom, lost = restore_filters(state)
    history = state.get('intervention_log', [])
    if not isinstance(history, list):
        raise ValueError('Cannot resume with invalid filter intervention history')
    active: set[str] = set()
    historical_custom = 0
    for value in cast(list[object], history):
        entry = _mapping(value, 'filter intervention')
        operation = entry.get('operation')
        if operation == 'remove_custom':
            historical_custom += 1
        elif operation in ('set_filter', 'clear_filter'):
            key = entry.get('key')
            if not isinstance(key, str):
                raise ValueError('Cannot resume with invalid filter intervention key')
            if operation == 'set_filter':
                active.add(key)
            else:
                active.discard(key)
    custom = max(custom, historical_custom)
    for key in active - restored.keys():
        lost.setdefault(key, 'missing checkpoint descriptor')
    if not custom and not lost:
        return

    cap, trim, yielded = state.get('n_permutations'), state.get('trim_budget'), state['yielded_count']
    if type(yielded) is not int or yielded < 0 or (cap is not None and type(cap) is not int) or (trim is not None and type(trim) is not int):
        raise ValueError('Cannot resume with invalid search completion counters')
    if (type(trim) is int and trim <= 0) or (type(cap) is int and yielded >= cap):
        return
    if not state['priority_queue'] and type(strategy) is GridStrategy:
        probe = copy.deepcopy(strategy)
        probe.domain.set_state(_mapping(checkpoint['domain_state'], 'domain state'))
        probe.set_state(_mapping(state['strategy_state'], 'strategy state'))
        frontier = _mapping(checkpoint['metadata'], 'checkpoint metadata')['experiment_round']
        if type(frontier) is not int or frontier < -1:
            raise ValueError('Cannot resume with invalid checkpoint frontier')
        with csv_path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            column = 'id' if parallel else next((key for key in ('_param_hash', '_id') if key in (reader.fieldnames or [])), None)
            hashes = [row[column] for _, row in zip(range(frontier + 1), reader, strict=False) if column is not None and row.get(column)]
        probe.rebuild_seen_from_log(hashes)
        if next(probe, None) is None:
            return

    details = f'{custom} custom filter(s), named filters: {sorted(lost)}'
    if lost:
        details += '; ' + '; '.join(f'{key}: {lost[key]}' for key in sorted(lost))
    raise ValueError(
        f'Cannot resume: Checkpoint had non-restorable filters ({details}). '
        'Start a new experiment directory; use supported declarative set_filter descriptors for resumable constraints.'
    )
