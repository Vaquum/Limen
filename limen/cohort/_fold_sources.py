from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from numbers import Real
from pathlib import Path
from typing import Protocol, TypeVar, cast

import numpy as np

from limen.cohort._round_sources import RoundSource, RoundSources, architecture_identity, same_architecture
from limen.experiment import manifest_core
from limen.experiment import _round_storage
from limen.experiment._fold_evidence import model_witness, read_fold_context
from limen.experiment.manifest_core import Manifest
from limen.inference.errors import ReconstructionError
from limen.yaml.compiler import CompiledSFD
from limen.yaml.config import is_mapping

__all__ = ['FoldSources', 'bind_native_output', 'select_sources']
_PAIR = 2
_TRIPLE = 3
Key = tuple[str, str] | tuple[str, str, int]
MemberId = tuple[str, str] | tuple[str, str, int, str]
Selection = tuple[Path, str, int | None]
Architecture = Callable[[Mapping[str, object], Mapping[str, object]], str]
Source = TypeVar('Source', bound=RoundSource)
Bar = TypeVar('Bar')
Load = Callable[[Path], Mapping[str, Mapping[str, object]]]
resolve_params = cast(Callable[[Mapping[str, object], Mapping[str, object]], dict[str, object]], vars(manifest_core)['_resolve_params'])


class Owner(Protocol):
    supports_probabilities: bool | None
    aggregation_mode: str | None

    @property
    def architecture_id(self) -> str: ...

    @property
    def member_ids(self) -> Sequence[tuple[object, ...]]: ...


@dataclass
class _Source:
    experiment_dir: Path
    metadata: dict[str, object]
    manifest_id: str
    architecture_id: str
    available_permutation_ids: list[str]
    permutation_ids: list[str] | None = None
    experiment_id: None = None
    supports_probabilities: bool | None = None
    aggregation_mode: str | None = None


def _metadata(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads((path / 'metadata.json').read_text())
    except (ValueError, UnicodeError) as exc:
        raise ReconstructionError(f'Cohort source {path}: invalid metadata: {exc}') from exc
    if not isinstance(value, dict):
        raise ReconstructionError(f'Cohort source {path}: metadata must be an object')
    return cast(dict[str, object], value)


def _walk_forward(metadata: Mapping[str, object]) -> bool:
    reference = metadata.get('yaml_reference')
    sfd = cast(Mapping[str, object], reference).get('sfd') if isinstance(reference, Mapping) else None
    manifest = cast(Mapping[str, object], sfd).get('manifest') if isinstance(sfd, Mapping) else None
    declaration = cast(Mapping[str, object], manifest).get('split_walk_forward') if isinstance(manifest, Mapping) else None
    return metadata.get('split_walk_forward') is not None or declaration is not None


def _parse(rounds: object, normalize: Callable[[int | str], str]) -> list[Selection]:
    if not isinstance(rounds, list):
        raise TypeError('Cohort rounds must be a list')
    if not rounds:
        raise ValueError('Cohort rounds must be a non-empty list')
    selected: list[Selection] = []
    for raw in cast(list[object], rounds):
        if not isinstance(raw, (tuple, list)):
            raise ValueError(f'Cohort invalid member selection {raw!r}')
        parts = cast(Sequence[object], raw)
        if len(parts) not in (_PAIR, _TRIPLE):
            raise ValueError(f'Cohort invalid member selection {raw!r}')
        directory = parts[0]
        if not isinstance(directory, (str, Path)):
            raise TypeError('Cohort member directory must be str or Path')
        pid = normalize(cast(str, parts[1]))
        if not pid:
            raise ValueError('Cohort member permutation_id must not be blank')
        fold = parts[_PAIR] if len(parts) == _TRIPLE else None
        if len(parts) == _TRIPLE and (type(fold) is not int or fold < 0):
            raise ValueError(f'Cohort member {pid}: fold must be a nonnegative literal integer')
        selected.append((directory.expanduser().resolve() if isinstance(directory, Path) else Path(directory).expanduser().resolve(), pid, cast(int | None, fold)))
    return selected


def select_sources(rounds: object, owner: Owner,
                   legacy_factory: Callable[[list[tuple[str | Path, str]]], RoundSources[Source]],
                   normalize: Callable[[int | str], str], architecture: Architecture, load: Load) -> RoundSources[Source] | FoldSources:
    selected = _parse(rounds, normalize)
    if any(fold is not None for _, _, fold in selected):
        return FoldSources(selected, owner, architecture, load)
    for path in dict.fromkeys(path for path, _, _ in selected):
        if _walk_forward(_metadata(path)):
            raise ValueError(f'Cohort source {path}: walk-forward rounds cannot be promoted; supply fold')
    return legacy_factory([(path, pid) for path, pid, _ in selected])


def _binary(value: object) -> bool:
    native: Mapping[str, object] = cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}
    output = native.get('output_contract')
    if native.get('prediction_mode') != 'binary' or not isinstance(output, Mapping) or cast(Mapping[str, object], output).get('kind') != 'binary' or type(cast(Mapping[str, object], output).get('probabilities')) is not bool:
        raise ValueError('Cohort requires declared binary output contract')
    return cast(bool, output['probabilities'])


def bind_native_output(owner: Owner, members: Sequence[object], default: bool, architectures: Sequence[str] = ()) -> None:
    probabilities = default
    if owner.architecture_id.startswith('mixed-binary-v1:'):
        promises = [_binary(model_witness(getattr(member, '_model', None))) for member in members]
        policy = tuple(sorted(zip(owner.member_ids, architectures, promises, strict=True), key=lambda row: json.dumps(row[0])))
        vars(owner)['_mixed_policy'] = policy
        owner.supports_probabilities = all(promises)
        owner.aggregation_mode = 'probability_weighted' if all(promises) else 'majority_vote'
        return
    declarations = [cast(object, getattr(type(cast(object, getattr(member, '_model', None))), 'inference_output', None)) for member in members]
    for value in (False, True):
        if declarations and all(isinstance(declaration, tuple) and len(cast(tuple[object, ...], declaration)) == _PAIR and cast(tuple[object, ...], declaration)[0] == 'binary' and cast(tuple[object, ...], declaration)[1] is value for declaration in declarations):
            probabilities = value
    owner.supports_probabilities = probabilities
    owner.aggregation_mode = 'probability_weighted' if probabilities else 'majority_vote'


def mixed_outputs(owner: Owner, bars: Sequence[Bar]) -> list[Bar]:
    policy = cast(tuple[tuple[tuple[object, ...], str, bool], ...], vars(owner)['_mixed_policy'])
    promises = {row[0]: row[2] for row in policy}
    result: list[Bar] = []
    for member_id, bar in sorted(zip(owner.member_ids, bars, strict=True), key=lambda item: json.dumps(item[0])):
        if getattr(bar, 'reason', None) == 'sensor-error':
            continue
        decision, probability = getattr(bar, 'prediction', None), getattr(bar, 'probability', None)
        if isinstance(decision, bool) or not isinstance(decision, Real) or not np.isfinite(decision) or decision not in (0, 1):
            raise ValueError(f'Cohort member {member_id}: mixed binary decisions must be finite numeric 0 or 1')
        if promises[member_id] and (isinstance(probability, bool) or not isinstance(probability, Real) or not np.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError(f'Cohort member {member_id}: promised probability must be finite and in [0, 1]')
        result.append(bar)
    return result


def _compile(path: Path, metadata: Mapping[str, object]) -> Manifest:
    try:
        reference = metadata.get('yaml_reference')
        if not is_mapping(reference):
            raise ValueError('yaml_reference is required')
        manifest = CompiledSFD(reference).manifest()
        config = manifest.split_walk_forward
        stored = metadata.get('split_walk_forward')
        if isinstance(stored, Mapping) and type(cast(Mapping[str, object], stored).get('n_folds')) is not int:
            raise ValueError('fold count must be a literal integer')
        if metadata.get('split_walk_forward') != (None if config is None else config.as_dict()):
            raise ValueError('Cohort source declarations disagree')
        return manifest
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionError(f'Cohort source {path}: {exc}') from exc


class FoldSources:
    qualified_folds = True

    def __init__(self, selections: list[Selection], owner: Owner, architecture: Architecture, load: Load) -> None:
        super().__init__()
        self._owner = owner
        self._architecture = architecture
        self.sources: dict[Path, _Source] = {}
        self.source_metadata: dict[Path, Mapping[str, object]] = {}
        self.member_ids: list[MemberId] = []
        self._selected: dict[Key, dict[str, object] | None] = {}
        self._member_sources: dict[Key, _Source] = {}
        self._architectures: dict[Key, str] = {}
        signatures: list[object] = []
        campaigns: list[object] = []
        architectures: set[str] = set()
        grouped: dict[Path, list[Selection]] = {}
        for selection in selections:
            grouped.setdefault(selection[0], []).append(selection)
        for path, selected in grouped.items():
            metadata = _metadata(path)
            if _walk_forward(metadata) and any(fold is None for _, _, fold in selected):
                raise ValueError(f'Cohort source {path}: walk-forward rounds cannot be promoted; supply fold')
            manifest = _compile(path, metadata)
            config = manifest.split_walk_forward
            if config is None and any(fold is not None for _, _, fold in selected):
                raise ValueError(f'Cohort source {path}: fold on ordinary source')
            for _, pid, fold in selected:
                if fold is not None and config is not None and fold >= config.n_folds:
                    raise ValueError(f'Cohort member {pid}: fold {fold} out of range')
            mid = metadata.get('manifest_id')
            if not isinstance(mid, str) or not mid.strip():
                raise ValueError(f'Cohort source {path}: a known manifest_id is required')
            try:
                entries = load(_round_storage.round_path(path))
            except (KeyError, TypeError, ValueError) as exc:
                raise ReconstructionError(f'Cohort source {path}: invalid round records: {exc}') from exc
            source = _Source(path, metadata, mid, '', sorted(entries))
            for _, pid, fold in selected:
                if pid not in entries:
                    raise ValueError(f'Cohort source {path}: unknown member {pid}')
                entry = entries[pid]
                try:
                    context = None if fold is None else read_fold_context(metadata, entry, fold)
                    params = entry['round_params']
                    if not is_mapping(params):
                        raise ValueError('round_params must be an object')
                except (KeyError, TypeError, ValueError, AttributeError) as exc:
                    raise ReconstructionError(f'Cohort source {path}: {exc}') from exc
                key = (mid, pid) if fold is None else (mid, pid, fold)
                self._add(key, context, source)
                architecture_id = self._architecture(entry, metadata)
                self._architectures[key] = architecture_id
                architectures.add(architecture_id)
                source.architecture_id = architecture_id
                signature = self._signature(manifest, params)
                if signatures and signature != signatures[0]:
                    raise ValueError(f'Cohort member {key}: incompatible data source, architecture or effective bar formation')
                signatures.append(signature)
                if context is not None:
                    campaign = cast(Mapping[str, object], context['campaign'])
                    original = cast(Mapping[str, object], campaign['source'])
                    signature = (tuple(original[name] for name in ('digest', 'rows', 'first_ns', 'last_ns', 'interval_ns')), metadata['split_walk_forward'])
                    if campaigns and signature != campaigns[0]:
                        raise ValueError(f'Cohort member {key}: incompatible fold campaign source or split declaration')
                    campaigns.append(signature)
            self.sources[path], self.source_metadata[path] = source, metadata
        self.architecture_id = architecture_identity(architectures)
        for path, pid, fold in selections:
            mid = self.sources[path].manifest_id
            context = self._selected[(mid, pid) if fold is None else (mid, pid, fold)]
            self.member_ids.append((mid, pid) if context is None else (mid, pid, cast(int, fold), cast(str, context['provenance_digest'])))
        mids = {member[0] for member in self.member_ids}
        self.manifest_id = next(iter(mids)) if len(mids) == 1 else None

    def _add(self, key: Key, context: dict[str, object] | None, source: _Source) -> None:
        if key in self._selected:
            if self._selected[key] != context:
                raise ValueError(f'Cohort evidence conflict for member {key}')
            raise ValueError(f'Cohort duplicate member {key}')
        if context is not None:
            witness = cast(Mapping[str, object], context['fold_witness'])
            native = cast(Mapping[str, object], witness['inference_model'])
            if set(native) != {'class', 'prediction_mode', 'output_contract'} or not isinstance(native.get('class'), str) or not cast(str, native['class']).strip():
                raise ReconstructionError(f'Cohort source {source.experiment_dir}: invalid inference model witness')
            _ = _binary(native)
        self._selected[key], self._member_sources[key] = context, source

    @staticmethod
    def _signature(manifest: Manifest, params: Mapping[str, object]) -> object:
        source = manifest.data_source_config
        if source is None:
            raise ValueError('Cohort source requires a data source')
        formation = manifest.bar_formation
        bar_type = params.get('bar_type', 'base')
        if formation is not None and bar_type != 'base':
            formation = (formation[0], resolve_params(formation[1], params))
        else:
            formation = None
        return (source.method, {key: value for key, value in source.params.items() if key not in ('start_date_limit', 'end_date_limit')}, bar_type, formation)

    def bind(self, members: list[object]) -> list[object]:
        if len(members) != len(self.member_ids):
            raise ValueError('Cohort member count must match selected rounds count')
        bound: dict[Key, object] = {}
        for member in members:
            mid, pid, fold = getattr(member, 'manifest_id', None), getattr(member, 'permutation_id', None), getattr(member, 'fold', None)
            if not isinstance(mid, str) or not isinstance(pid, str) or (fold is not None and type(fold) is not int):
                raise ValueError('Cohort unknown or wrong-fold member identity')
            key = (mid, pid.strip()) if fold is None else (mid, pid.strip(), fold)
            if key not in self._selected:
                raise ValueError(f'Cohort unknown or wrong-fold member {key}')
            if key in bound:
                raise ValueError(f'Cohort duplicate bound member {key}')
            bound[key] = member
        probabilities: list[bool] = []
        for key, member in bound.items():
            source, context = self._member_sources[key], self._selected[key]
            params = getattr(member, 'round_params', None)
            if not is_mapping(params) or not same_architecture(self._architecture({'round_params': params}, source.metadata), self._architectures[key]):
                raise ValueError(f'Cohort member {key}: bound architecture must match its source')
            try:
                native = model_witness(getattr(member, '_model', None))
                probabilities.append(_binary(native))
            except ValueError as exc:
                raise ValueError(f'Cohort requires declared binary output contract for member {key}: {exc}') from exc
            if context is not None:
                witness = cast(Mapping[str, object], context['fold_witness'])
                if type(getattr(member, 'n_folds', None)) is not int or getattr(member, 'n_folds', None) != witness['n_folds'] or getattr(member, 'provenance_digest', None) != context['provenance_digest'] or native != witness['inference_model']:
                    raise ValueError(f'Cohort selected evidence mismatch for member {key}')
        keys = [cast(Key, member[:_TRIPLE] if len(member) > _PAIR else member) for member in self.member_ids]
        ordered = [bound[key] for key in keys]
        bind_native_output(self._owner, ordered, all(probabilities), [self._architectures[key] for key in keys])
        return ordered

    def identity(self, aggregation_mode: str) -> dict[str, object]:
        rows = [['ordinary', *member] if len(member) == _PAIR else ['fold', *member] for member in self.member_ids]
        rows.sort(key=lambda row: (cast(str, row[1]), cast(str, row[2]), cast(str, row[0]), cast(int, row[_TRIPLE]) if row[0] == 'fold' else -1, cast(str, row[_TRIPLE + 1]) if row[0] == 'fold' else ''))
        return {'architecture_id': self.architecture_id, 'aggregation_mode': aggregation_mode, 'members': rows}
