from __future__ import annotations

import limen.experiment._round_storage as _storage
import json

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Generic, Protocol, TypeVar, cast

from limen.experiment import manifest_core
from limen.yaml.compiler import CompiledSFD
from limen.yaml.config import is_mapping
from limen.yaml.resolver import resolve


def architecture_identity(architectures: set[str]) -> str:
    return next(iter(architectures)) if len(architectures) == 1 else 'mixed-binary-v1:' + json.dumps(sorted(architectures), separators=(',', ':'))


def same_architecture(actual: str, expected: str) -> bool:
    return actual.strip().lower() == expected.strip().lower() or resolve(actual) is resolve(expected)

__all__ = ['RoundSources']

_resolve_params = cast(Callable[[Mapping[str, object], Mapping[str, object]], dict[str, object]], vars(manifest_core)['_resolve_params'])


class RoundSource(Protocol):
    @property
    def metadata(self) -> Mapping[str, object] | None: ...

    @property
    def manifest_id(self) -> str | None: ...

    @property
    def architecture_id(self) -> str: ...


Source = TypeVar('Source', bound=RoundSource)


class RoundSources(Generic[Source]):
    def __init__(self, rounds: list[tuple[str | Path, str]],
                 source_factory: Callable[[Path, list[str]], Source],
                 load_entries: Callable[[Path], Mapping[str, Mapping[str, object]]],
                 architecture: Callable[[Mapping[str, object], Mapping[str, object]], str],
                 normalize_id: Callable[[int | str], str]) -> None:
        super().__init__()
        if not rounds:
            raise ValueError('Cohort rounds must be a non-empty list.')
        grouped: dict[Path, list[str]] = {}
        normalized: list[tuple[Path, str]] = []
        for directory, raw_pid in rounds:
            path = Path(directory).expanduser().resolve()
            pid = normalize_id(raw_pid)
            grouped.setdefault(path, []).append(pid)
            normalized.append((path, pid))

        self.sources: dict[Path, Source] = {}
        self.member_sources: dict[tuple[str, str], Source] = {}
        self.source_metadata: dict[Path, Mapping[str, object]] = {}
        self._architecture = architecture
        self._architectures: dict[tuple[str, str], str] = {}
        signatures: list[object] = []
        architectures: set[str] = set()
        for path, ids in grouped.items():
            try:
                source = source_factory(path, ids)
            except ValueError as exc:
                raise ValueError(f'Cohort source {path}: {exc}') from exc
            metadata = source.metadata
            if metadata is None or not isinstance(source.manifest_id, str) or not source.manifest_id.strip():
                raise ValueError(f'Cohort source {path}: a known manifest_id is required.')
            yaml_reference = metadata.get('yaml_reference')
            if not is_mapping(yaml_reference):
                raise ValueError(f'Cohort source {path}: yaml_reference is required for compatibility checks.')
            manifest = CompiledSFD(yaml_reference).manifest()
            if manifest.split_walk_forward is not None:
                raise ValueError(f'Cohort source {path}: walk-forward rounds cannot be promoted.')
            data_source = manifest.data_source_config
            if data_source is None:
                raise ValueError(f'Cohort source {path}: data source is required.')
            source_signature = (data_source.method, {k: v for k, v in data_source.params.items()
                                                    if k not in ('start_date_limit', 'end_date_limit')})
            entries = load_entries(_storage.round_path(path))
            for pid in ids:
                key = (source.manifest_id, pid)
                if key in self.member_sources:
                    raise ValueError(f'Cohort duplicate member {key} in source {path}.')
                params = entries[pid]['round_params']
                if not is_mapping(params):
                    raise ValueError(f'Cohort member {key} in source {path}: round_params are required.')
                bar_type = params.get('bar_type', 'base')
                formation = None
                if manifest.bar_formation and bar_type != 'base':
                    func, base_params = manifest.bar_formation
                    formation = (func, _resolve_params(base_params, params))
                signature = (source_signature, bar_type, formation)
                if signatures and signature != signatures[0]:
                    raise ValueError(f'Cohort member {key} in source {path}: incompatible data source or effective bar formation.')
                signatures.append(signature)
                self.member_sources[key] = source
                self._architectures[key] = architecture(entries[pid], metadata)
                architectures.add(self._architectures[key])
            self.sources[path] = source
            self.source_metadata[path] = metadata

        self.member_ids = [(cast(str, self.sources[path].manifest_id), pid) for path, pid in normalized]
        self.architectures = [self._architectures[key] for key in self.member_ids]
        self.architecture_id = architecture_identity(architectures)
        manifest_ids = {mid for mid, _ in self.member_ids}
        self.manifest_id = next(iter(manifest_ids)) if len(manifest_ids) == 1 else None

    def bind(self, members: list[object]) -> list[object]:
        if len(members) != len(self.member_ids):
            raise ValueError('Cohort member count must match selected rounds count.')
        bound: dict[tuple[str, str], object] = {}
        for member in members:
            mid = getattr(member, 'manifest_id', None)
            pid = getattr(member, 'permutation_id', None)
            if not isinstance(mid, str) or not isinstance(pid, str):
                raise ValueError(f'Cohort member {(mid, pid)}: manifest_id and permutation_id are required.')
            pid = pid.strip()
            key = (mid, pid)
            if key not in self.member_sources:
                raise ValueError(f'Cohort unknown or wrong-source member {key}.')
            if key in bound:
                raise ValueError(f'Cohort duplicate bound member {key}.')
            if getattr(member, 'prediction_mode', 'binary') == 'target_exposure':
                raise ValueError(f'Binary Cohort cannot aggregate target-exposure member {key}.')
            source = self.member_sources[key]
            assert source.metadata is not None
            params = getattr(member, 'round_params', None)
            if not is_mapping(params):
                raise ValueError(f'Cohort member {key}: round_params are required for architecture binding.')
            if not same_architecture(self._architecture({'round_params': params}, source.metadata), self._architectures[key]):
                raise ValueError(f'Cohort member {key}: bound architecture must match its source.')
            bound[key] = member
        missing = [key for key in self.member_ids if key not in bound]
        if missing:
            raise ValueError(f'Cohort missing bound members: {missing}.')
        return [bound[key] for key in self.member_ids]

    def identity(self, aggregation_mode: str) -> dict[str, object]:
        payload: dict[str, object] = {'architecture_id': self.architecture_id, 'aggregation_mode': aggregation_mode}
        if self.manifest_id is None:
            payload['rounds'] = sorted(self.member_ids)
        else:
            payload.update(manifest_id=self.manifest_id, permutation_ids=sorted(pid for _, pid in self.member_ids))
        return payload
