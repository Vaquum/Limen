from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from functools import wraps
from importlib import import_module
from typing import Concatenate, ParamSpec, Protocol, TypeVar, cast

import polars as pl

from limen.experiment._holdout_history import ProjectBinding, research_record
from limen.experiment._holdout_source import bounds, frame_evidence, record_geometry
from limen.experiment._walk_forward_split import WalkForwardConfig
__all__: list[str] = []
_T = TypeVar('_T')
_P = ParamSpec('_P')


class _Manifest(Protocol):
    @property
    def split_walk_forward(self) -> WalkForwardConfig | None: ...
    @property
    def split_config(self) -> tuple[int, int, int]: ...


class _Operation(Protocol):
    binding: ProjectBinding
    provenance: dict[str, object]
    replay_witness: Mapping[str, object] | None


class _Research(Protocol):
    def operation(self, manifest: object, role: str) -> AbstractContextManager[_Operation | None]: ...
    def admit(self, manifest: object, raw: pl.DataFrame) -> pl.DataFrame: ...
    def selected_input(self, manifest: object, raw: pl.DataFrame) -> pl.DataFrame: ...
    def binding_of(self, manifest: object) -> ProjectBinding | None: ...
    def metadata(self, manifest: object) -> dict[str, object] | None: ...
    def bind_round(self, manifest: object, provenance: Mapping[str, object], round_id: str) -> dict[str, object]: ...


class _Checkpoint(Protocol):
    def completed_history(self, binding: ProjectBinding, proof: Mapping[str, object]) -> None: ...
    def validate_trainer_round(self, binding: ProjectBinding, identity: object, proof: object) -> str: ...


class _Native(Protocol):
    def add_metadata(self, metadata: dict[str, object], host: object, workers: int) -> None: ...
    def source(self, raw: pl.DataFrame, declared_seconds: object = None) -> dict[str, object]: ...
    def read_fold_context(self, metadata: Mapping[str, object], entry: Mapping[str, object], fold: int) -> dict[str, object]: ...


_M = TypeVar('_M', bound=_Manifest)
research = cast(_Research, import_module('limen.experiment._holdout_research'))
checkpoint = cast(_Checkpoint, import_module('limen.experiment._holdout_checkpoint'))


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError('Protected fold lacks recorded workflow evidence')
    return cast(Mapping[str, object], value)


def construction(func: Callable[Concatenate[_T, _M, pl.DataFrame, _P], None]) -> Callable[Concatenate[_T, _M, pl.DataFrame, _P], None]:
    @wraps(func)
    def guarded(host: _T, manifest: _M, raw: pl.DataFrame, *args: _P.args, **kwargs: _P.kwargs) -> None:
        if not isinstance(manifest.split_walk_forward, WalkForwardConfig):
            raise ValueError('split_walk_forward requires its manifest configuration')
        with research.operation(manifest, 'experiment'):
            func(host, manifest, research.admit(manifest, raw), *args, **kwargs)
    return guarded


def prepare(host: object, raw: pl.DataFrame, round_params: Mapping[str, object] | None,
            resolve: Callable[[Mapping[str, object], Mapping[str, object]], dict[str, object]],
            requires_validation: Callable[..., bool]) -> dict[str, object]:
    manifest = getattr(host, 'manifest', None)
    params = dict(round_params or {})
    with research.operation(manifest, 'experiment') as op:
        raw = research.admit(manifest, raw)
        selector = cast(tuple[Callable[..., pl.DataFrame], Mapping[str, object]] | None, getattr(manifest, 'pre_split_data_selector', None))
        if selector is not None:
            func, base = selector
            resolved = resolve(base, params)
            raw = func(raw, **resolved)
            if op is not None:
                op.provenance['selector'] = {'method': f'{func.__module__}.{func.__qualname__}', 'params': resolved, 'seed': params.get('seed', base.get('seed'))}
        raw = research.selected_input(manifest, raw)
        cast(Callable[..., None], getattr(host, '_preflight', None))(raw, require_validation=requires_validation(manifest, params))
        data: dict[str, object] = {'_walk_forward_params': params, '_walk_forward_raw': raw}
        if op is not None:
            op.provenance.update(selected=bounds(raw, cast(int, op.binding.interval_seconds)), schema_reason='walk-forward-trial')
            data['_fold_holdout'] = copy.deepcopy(op.provenance)
    return data


def trial_record(host: object, record: dict[str, object]) -> dict[str, object]:
    native = cast(_Native, import_module('limen.experiment._fold_evidence'))
    from limen.yaml.store import canonical_manifest_id
    state = vars(host)
    manifest = state['manifest']
    binding = research.binding_of(manifest)
    if binding is None or binding.cutoff_ns is None:
        return record
    record = copy.deepcopy(record)
    proof = _mapping(record.get('holdout'))
    checkpoint.completed_history(binding, proof)
    parent = research.metadata(manifest)
    if parent is None or proof['experiment_record_id'] != parent['research_record_id']:
        raise ValueError('Protected fold trial lacks its current coordinator')
    records = cast(list[dict[str, object]], record['folds'])
    for fold, item in enumerate(records):
        prepared = _mapping(item.get('holdout'))
        checkpoint.completed_history(binding, prepared)
        if prepared['experiment_record_id'] != parent['research_record_id']:
            raise ValueError('Protected fold preparation belongs to another trial operation')
        item['holdout'] = research.bind_round(manifest, dict(prepared, workflow={'version': 1, 'kind': 'fold', 'fold': fold,
            'preparation_record_id': prepared['research_record_id'], 'native': copy.deepcopy(item['fold_promotion'])}), str(record['round_id']))
    metadata: dict[str, object] = {}
    native.add_metadata(metadata, state['_walk_forward'], cast(int, state['_workers']))
    reference = cast(dict[str, object] | None, state['_yaml_reference'])
    workflow = {'version': 1, 'kind': 'walk_forward', 'preparation_record_id': proof['research_record_id'],
        'manifest_id': None if reference is None else canonical_manifest_id(reference), 'campaign': metadata['fold_promotion'],
        'recording': state['_recording'], 'record_model_outputs': state['_record_model_outputs'],
        'trial': record['fold_promotion'], 'folds': [cast(Mapping[str, object], item['holdout'])['research_record_id'] for item in records]}
    record['holdout'] = research.bind_round(manifest, dict(proof, workflow=workflow), str(record['round_id']))
    return record


def live(manifest: object, raw: pl.DataFrame, binding: ProjectBinding, entry: Mapping[str, object]) -> None:
    native = cast(_Native, import_module('limen.experiment._fold_evidence'))
    raw, source = frame_evidence(raw, binding)
    read = cast(Callable[[ProjectBinding, str, object, Mapping[str, object] | None], Mapping[str, object]], vars(checkpoint)['_record'])
    proof = read(binding, str(entry['round_id']), entry.get('holdout'), source)
    module = sys.modules['limen.experiment.manifest_core']
    resolve = cast(Callable[[Mapping[str, object], Mapping[str, object]], dict[str, object]], vars(module)['_resolve_params'])
    split = cast(Callable[[object, pl.DataFrame], list[pl.DataFrame]], vars(module)['_resolve_split'])
    selector = cast(tuple[Callable[..., pl.DataFrame], Mapping[str, object]] | None, getattr(manifest, 'pre_split_data_selector', None))
    selected = raw if selector is None else selector[0](raw, **resolve(selector[1], _mapping(entry['round_params'])))
    selected = research.selected_input(manifest, selected)
    if 'workflow' not in proof or _mapping(proof['workflow']).get('kind') == 'sources':
        record_geometry({'splits': {}}, proof, selected, dict(zip(('train', 'validation', 'test'), split(manifest, selected), strict=True)), cast(int, binding.interval_seconds))
        return
    flow = _mapping(proof['workflow'])
    if flow.get('version') != 1 or flow.get('kind') != 'walk_forward':
        raise ValueError('Unsupported protected fold workflow')
    campaign, trial = _mapping(flow['campaign']), _mapping(flow['trial'])
    interval = cast(int, _mapping(campaign['source'])['interval_ns']) / 1_000_000_000
    if native.source(raw, interval) != campaign['source'] or native.source(selected, interval) != trial['selected_source']:
        raise ValueError('Protected fold source or selected population changed')
    config = cast(_Manifest, manifest).split_walk_forward
    if config is None:
        raise ValueError('Protected fold configuration is missing')
    metadata = {'manifest_id': flow['manifest_id'], 'recording': flow['recording'], 'record_model_outputs': flow['record_model_outputs'],
        'split_walk_forward': config.as_dict(), 'walk_forward_validation_ratio': list(cast(_Manifest, manifest).split_config[:2]), 'fold_promotion': campaign}
    if not isinstance(entry.get('folds'), list) or any(not isinstance(item, Mapping) for item in cast(list[object], entry['folds'])):
        raise ValueError('Protected fold records are missing or malformed')
    records = cast(list[Mapping[str, object]], entry['folds'])
    if flow['trial'] != entry['fold_promotion'] or flow['folds'] != [_mapping(item.get('holdout'))['research_record_id'] for item in records]:
        raise ValueError('Protected fold trial or occurrence history changed')
    for fold, item in enumerate(records):
        _ = native.read_fold_context(metadata, entry, fold)
        saved = _mapping(item.get('holdout'))
        checkpoint.completed_history(binding, saved)
        folded = _mapping(saved.get('workflow'))
        if saved['round_id'] != entry['round_id'] or folded != {'version': 1, 'kind': 'fold', 'fold': fold,
                'preparation_record_id': folded.get('preparation_record_id'), 'native': item['fold_promotion']}:
            raise ValueError('Protected fold membership or identity changed')
        original = _mapping(json.loads(str(research_record(binding, cast(str, folded['preparation_record_id']))['provenance_json'])))
        checkpoint.completed_history(binding, original)
        excluded = ('research_record_id', 'round_id', 'workflow')
        if {key: value for key, value in original.items() if key not in excluded} != {key: value for key, value in saved.items() if key not in excluded}:
            raise ValueError('Protected fold publication differs from its actual preparation')
        copied = copy.deepcopy(manifest)
        vars(copied)['_walk_forward_fold'] = fold
        record_geometry({'splits': {}}, saved, selected, dict(zip(('train', 'validation', 'test'), split(copied, selected), strict=True)), cast(int, binding.interval_seconds))


@contextmanager
def trainer(manifest: _Manifest, entry: Mapping[str, object], fold: int) -> Iterator[None]:
    binding = research.binding_of(manifest)
    if binding is None or binding.cutoff_ns is None:
        yield
        return
    record = _mapping(cast(list[object], entry['folds'])[fold])
    proof = _mapping(record.get('holdout'))
    _ = checkpoint.validate_trainer_round(binding, entry['round_id'], proof)
    workflow = _mapping(proof.get('workflow'))
    if workflow.get('version') != 1 or workflow.get('kind') != 'fold' or workflow.get('fold') != fold or workflow.get('native') != record['fold_promotion']:
        raise ValueError('Protected Trainer selected fold history changed')
    vars(manifest)['_holdout_round_id'], vars(manifest)['_holdout_experiment_record'] = entry['round_id'], proof['research_record_id']
    with research.operation(manifest, 'trainer') as op:
        if op is None:
            raise ValueError('Protected fold Trainer lacks its research operation')
        op.replay_witness = proof
        yield


def restore_entry(host: object, entry: dict[str, object]) -> dict[str, object]:
    state = vars(host)
    manifest = state['manifest']
    binding = research.binding_of(manifest)
    if binding is not None and binding.cutoff_ns is not None:
        if research.metadata(manifest) is None:
            raise ValueError('Protected fold restore requires its held research operation')
        live(manifest, cast(pl.DataFrame, state['raw']), binding, entry)
    return entry
