"""Invocation-local protection checks for the two factorization caches."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Protocol
import warnings

from limen.experiment.convergence_report import convergence_warning
from typing import TypeVar, cast

import polars as pl

import limen.experiment._holdout_research as _holdout
from limen.experiment._holdout_history import ProjectBinding, check_binding, research_record
from limen.experiment._holdout_source import frame_evidence

__all__: list[str] = []
_T = TypeVar('_T')
_HEADER = ('project_id', 'reservation_revision', 'cutoff_ns', 'interval_seconds')


class _Prepared(Protocol):
    cache: dict[str, dict[str, object]]
    proofs: dict[str, dict[str, object]]
    manifest: object


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError('Factorized holdout reuse lacks genuine recorded protection evidence')
    return cast(Mapping[str, object], value)


def _binding(manifest: object) -> ProjectBinding | None:
    binding = _holdout.binding_of(manifest)
    if binding is None:
        return None
    check_binding(binding)
    if binding.cutoff_ns is not None:
        op = _holdout._CURRENT.get()
        if op is None or op.manifest is not manifest or op.binding != binding:
            raise ValueError('Protected factorized reuse requires a held research operation')
    return binding


def _check(manifest: object, raw: pl.DataFrame, proof: Mapping[str, object]) -> None:
    binding = _binding(manifest)
    if binding is None or binding.cutoff_ns is None:
        if proof:
            raise ValueError('Protected factorized population cannot become unprotected')
        return
    if any(proof.get(key) != getattr(binding, key if key != 'reservation_revision' else 'revision')
           for key in _HEADER):
        raise ValueError('Factorized holdout binding or revision changed')
    admitted, actual = frame_evidence(raw, binding)
    if admitted.height != raw.height:
        raise ValueError('Factorized holdout source introduced reserved observations')
    source = _mapping(proof.get('source'))
    original = _mapping(getattr(manifest, '_holdout_source', None))
    for key in ('admitted_sha256', 'admitted', 'interval_seconds'):
        if source.get(key) != actual[key] or original.get(key) != source.get(key):
            raise ValueError('Factorized holdout admitted source population changed')
    if any(source.get(key) != original.get(key) for key in ('kind', 'identity', 'method')):
        raise ValueError('Factorized holdout source declaration changed')
    identity = proof.get('research_record_id')
    if not isinstance(identity, str):
        raise ValueError('Factorized holdout reuse lacks original research record')
    saved = research_record(binding, identity)
    original_proof = dict(proof)
    original_proof.pop('reuse', None)
    if (saved['role'] != 'experiment' or saved['state'] != 'completed'
            or saved['revision'] != binding.revision
            or json.loads(str(saved['provenance_json'])) != original_proof):
        raise ValueError('Factorized holdout reuse conflicts with completed history')


def prepared_cache(cache: dict[str, dict[str, object]], proofs: dict[str, dict[str, object]],
                   manifest: object, raw: pl.DataFrame, key: str) -> dict[str, object] | None:
    binding = _binding(manifest)
    saved = cache.get(key)
    if saved is None:
        proofs.pop(key, None)
        return None
    if binding is not None and binding.cutoff_ns is not None:
        _check(manifest, raw, _mapping(proofs.get(key)))
    cache.pop(key)
    cache[key] = saved
    return saved


def _carrier(data: dict[str, object], proof: Mapping[str, object]) -> None:
    copied = copy.deepcopy(dict(proof))
    alignment = cast(dict[str, object], data['_alignment'])
    alignment['holdout'] = copied
    data['holdout'] = copied
    fitted = data.get('_fitted_params')
    if isinstance(fitted, dict):
        cast(dict[str, object], fitted)['_holdout'] = copied


def fitted_artifacts(manifest: object, data: dict[str, object],
                     artifacts: dict[str, object], original: object) -> dict[str, object]:
    binding = _binding(manifest)
    if binding is None or binding.cutoff_ns is None:
        if '_holdout' in artifacts:
            raise ValueError('Unprotected factorization cannot carry protected fitted metadata')
        return artifacts
    proof = _mapping(data.get('holdout'))
    if (_mapping(artifacts.get('_holdout')) != proof
            or _mapping(_mapping(data.get('_alignment')).get('holdout')) != proof):
        raise ValueError('Factorized protection carrier differs across prepared data')
    if original is not None:
        _carrier(data, dict(proof, reuse={'kind': 'fitted',
                                          'source_preparation_record_id': _mapping(original)['research_record_id']}))
    return {key: value for key, value in artifacts.items() if key != '_holdout'}


def remember_fitted(owner: _Prepared, key: str, fitted: dict[str, object],
                    data: dict[str, object], raw: pl.DataFrame) -> None:
    binding = _binding(owner.manifest)
    if binding is not None and binding.cutoff_ns is not None and key not in owner.proofs:
        proof = _mapping(_mapping(data['_alignment']).get('holdout'))
        original = dict(proof)
        original.pop('reuse', None)
        _check(owner.manifest, raw, original)
        owner.proofs[key] = copy.deepcopy(original)
    owner.cache[key] = fitted


def evict_fitted(cache: dict[str, dict[str, object]], proofs: dict[str, dict[str, object]]) -> None:
    key = next(iter(cache))
    del cache[key]
    proofs.pop(key, None)


def snapshot(manifest: object, raw: pl.DataFrame
             ) -> tuple[pl.DataFrame, tuple[ProjectBinding | None, Mapping[str, object] | None]]:
    source = getattr(manifest, '_holdout_source', None)
    return raw.clone(), (_holdout.binding_of(manifest),
                         copy.deepcopy(_mapping(source)) if source is not None else None)


def signal_cache(manifest: object, raw: pl.DataFrame,
                 original: tuple[ProjectBinding | None, Mapping[str, object] | None],
                 cache: dict[str, _T], key: str) -> _T | None:
    binding = _binding(manifest)
    if binding != original[0]:
        raise ValueError('Factorized holdout binding changed since preparation')
    if binding is not None and binding.cutoff_ns is not None:
        source = _mapping(original[1])
        current = _mapping(getattr(manifest, '_holdout_source', None))
        admitted, evidence = frame_evidence(raw, binding)
        if admitted.height != raw.height or any(
                source.get(name) != current.get(name) for name in ('kind', 'identity', 'method', 'admitted_sha256', 'admitted')
        ) or any(source.get(name) != evidence[name] for name in ('admitted_sha256', 'admitted')):
            raise ValueError('Factorized holdout source population changed since cache construction')
    entry = cache.get(key)
    if entry is not None and binding is not None and binding.cutoff_ns is not None:
        data = _mapping(getattr(entry, 'data'))
        alignment = _mapping(data.get('_alignment'))
        _check(manifest, raw, _mapping(alignment.get('holdout')))
    return cache.pop(key, None)


def signal_result(manifest: object, raw: pl.DataFrame, data: dict[str, object],
                  result: dict[str, object], recorded: list[str], previous: bool,
                  caught: list[warnings.WarningMessage]
                  ) -> tuple[dict[str, object], dict[str, object], list[str], bool]:
    converged = previous or bool(convergence_warning(caught, True))
    binding = _binding(manifest)
    if binding is None or binding.cutoff_ns is None:
        return data, result, recorded, converged
    original = _mapping(_mapping(data['_alignment']).get('holdout'))
    _check(manifest, raw, original)
    with _holdout.operation(manifest, 'experiment') as op:
        if op is None:
            raise ValueError('Protected factorized signal reuse lacks research history')
        op.provenance.update({key: copy.deepcopy(value) for key, value in original.items()
                              if key not in ('research_record_id', 'role', 'experiment_record_id', 'round_id', 'reuse')})
        op.provenance['reuse'] = {'kind': 'signal', 'source_preparation_record_id': original['research_record_id']}
        _carrier(data, op.provenance)
    return data, result, recorded, converged
