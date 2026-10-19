from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import copy
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import wraps
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar, cast

import polars as pl

from limen.experiment._holdout_history import ProjectBinding, admission, bind, check_binding, finish_research, start_research
from limen.experiment._holdout_source import SourceEvidence, admit_frame, bounds, frame_evidence, load_admitted_source, source_spec

if TYPE_CHECKING:
    from limen.experiment.manifest_core import Manifest

_P = ParamSpec('_P')
_R = TypeVar('_R')
_T = TypeVar('_T')


@dataclass
class _Operation:
    manifest: object
    binding: ProjectBinding
    record_id: str
    role: str
    parent_id: str | None
    provenance: dict[str, object] = field(default_factory=dict)


_CURRENT: ContextVar[_Operation | None] = ContextVar('holdout_operation', default=None)


def binding_of(manifest: object) -> ProjectBinding | None:
    return cast(ProjectBinding | None, getattr(manifest, '_holdout_binding', None))


def bind_manifest(manifest: object, *hints: str | Path | None) -> ProjectBinding | None:
    binding = bind(*hints)
    previous = binding_of(manifest)
    if previous is not None and previous != binding:
        raise ValueError('Holdout manifest binding/revision changed; construct a new manifest')
    vars(manifest)['_holdout_binding'] = binding
    return binding


def reject_modes(manifest: object, *, workers: object = 1, factorize: object = False,
                 resume: object = False, callbacks: tuple[object, ...] = (), test_mode: bool = False,
                 profile: bool = False) -> None:
    binding = binding_of(manifest)
    if binding is None or binding.cutoff_ns is None:
        return
    from limen.experiment.manifest_core import MLManifest, RuleBasedManifest
    if type(manifest) not in (MLManifest, RuleBasedManifest):
        raise ValueError('Holdout research requires a built-in manifest')
    model = cast(Manifest, manifest)
    if (workers != 1 or bool(factorize) or bool(resume) or profile or any(value is not None for value in callbacks)
            or model.split_walk_forward is not None or model.bar_formation is not None
            or (test_mode and model.test_data_source_config is not None)
            or getattr(model, 'data_dict_extension', None) is not None):
        raise ValueError('Holdout research refuses profile, resume, parallel/factorized/walk-forward, alternate test source and callbacks')
    config = model.backtest_config
    if config is not None and (getattr(config, 'trade_policy', None) is not None or getattr(config, 'execution_data_source', None) is not None or getattr(config, 'funding', None) is not None):
        raise ValueError('Holdout research does not support event/funding/alternate execution sources')
    if model.split_dates is not None:
        for value in model.split_dates:
            if not isinstance(value, (date, datetime)):
                raise ValueError('Holdout split_dates must be resolved UTC dates')
            dt = value if isinstance(value, datetime) else datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if int(dt.timestamp()) * 1_000_000_000 > binding.cutoff_ns:
                raise ValueError('Holdout split_dates extends beyond reserved research cutoff')
    source_spec(model, binding)


def _base(op: _Operation) -> dict[str, object]:
    empty = {'first_open_ns': None, 'last_close_ns': None, 'rows': 0, 'gaps': []}
    return {'schema_version': 1, 'project_id': op.binding.project_id, 'reservation_revision': op.binding.revision,
            'cutoff_ns': op.binding.cutoff_ns, 'interval_seconds': op.binding.interval_seconds,
            'research_record_id': op.record_id, 'role': op.role, 'experiment_record_id': op.parent_id,
            'round_id': None, 'source': None, 'selector': None, 'selected': empty,
            'split_mode': 'dates' if getattr(op.manifest, 'split_dates', None) is not None else 'ratio',
            'splits': {key: empty.copy() for key in ('train', 'validation', 'test')},
            'input_schema': None, 'model_schema': None, 'schema_reason': 'admission-without-fitted-model', 'earlier_exposure': 'unknown'}


@contextmanager
def operation(manifest: object, role: str) -> Iterator[_Operation | None]:
    binding = binding_of(manifest)
    with admission(binding.root) if binding is not None else nullcontext():
        if binding is not None:
            check_binding(binding)
        if binding is None or binding.cutoff_ns is None:
            yield None
            return
        reject_modes(manifest)
        parent = _CURRENT.get()
        parent_id = parent.record_id if parent is not None else None
        if role == 'trainer':
            parent_id = cast(str | None, getattr(manifest, '_holdout_experiment_record', None))
        op = _Operation(manifest, binding, start_research(binding, role), role, parent_id)
        op.provenance = _base(op)
        token = _CURRENT.set(op)
        try:
            yield op
        except Exception as exc:
            finish_research(binding, op.record_id, op.provenance, error=f'{type(exc).__name__}: {exc}')
            raise
        else:
            finish_research(binding, op.record_id, op.provenance)
        finally:
            _CURRENT.reset(token)


def fetch(manifest: Manifest) -> pl.DataFrame:
    binding = binding_of(manifest)
    if binding is None:
        binding = bind_manifest(manifest)
    if binding is None or binding.cutoff_ns is None:
        from limen.experiment.manifest_core import DataSourceResolver
        if manifest.data_source_config is None:
            raise ValueError('Manifest No data source configured')
        with admission(binding.root) if binding is not None else nullcontext():
            if binding is not None:
                check_binding(binding)
            return DataSourceResolver.resolve(manifest.data_source_config)
    with operation(manifest, 'experiment') as op:
        frame, source = load_admitted_source(manifest, binding, lower_ns=None, upper_ns=binding.cutoff_ns, complete=False)
        vars(manifest)['_holdout_source'] = source
        if op is not None:
            op.provenance['source'] = source
        return frame


def admit(manifest: object, raw: pl.DataFrame) -> pl.DataFrame:
    binding = binding_of(manifest)
    if binding is None or binding.cutoff_ns is None:
        return raw
    check_binding(binding)
    frame, evidence = frame_evidence(raw, binding)
    previous = cast(SourceEvidence | None, getattr(manifest, '_holdout_source', None))
    if previous is not None and previous['admitted_sha256'] == evidence['admitted_sha256']:
        evidence = previous
    op = _CURRENT.get()
    if op is not None and op.manifest is manifest:
        op.provenance['source'] = evidence
    vars(manifest)['_holdout_source'] = evidence
    return frame


def selected(manifest: object, raw: pl.DataFrame, splits: list[pl.DataFrame], params: Mapping[str, object]) -> None:
    op = _CURRENT.get()
    if op is None or op.manifest is not manifest or op.binding.interval_seconds is None:
        return
    selector = getattr(manifest, 'pre_split_data_selector', None)
    declaration = None
    if selector is not None:
        func, config = cast(tuple[Callable[..., object], Mapping[str, object]], selector)
        from limen.experiment.manifest_core import _resolve_params
        declaration = {'method': f'{func.__module__}.{func.__qualname__}', 'params': _resolve_params(dict(config), dict(params)), 'seed': params.get('seed', config.get('seed'))}
    op.provenance.update(selector=declaration, selected=bounds(raw, op.binding.interval_seconds),
                         splits={name: bounds(frame, op.binding.interval_seconds) for name, frame in zip(('train', 'validation', 'test'), splits, strict=True)})


def capture_schema(manifest: object, frame: pl.DataFrame, stage: str) -> None:
    op = _CURRENT.get()
    if op is None or op.manifest is not manifest:
        return
    excluded = {'datetime', getattr(manifest, 'target_column', None), '__trade_available_at_ns__', '__source_row__'}
    op.provenance[f'{stage}_schema'] = [{'name': name, 'dtype': str(dtype)} for name, dtype in frame.schema.items() if name not in excluded]
    op.provenance['schema_reason'] = None


def preparation(func: Callable[Concatenate[_T, _P], _R]) -> Callable[Concatenate[_T, _P], _R]:
    @wraps(func)
    def guarded(manifest: _T, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        if binding_of(manifest) is None:
            bind_manifest(manifest)
        parent = _CURRENT.get()
        with operation(manifest, parent.role if parent is not None else 'experiment') as op:
            result = func(manifest, *args, **kwargs)
            if op is not None and isinstance(result, dict):
                data = cast(dict[str, object], result)
                if getattr(manifest, 'strategy', None) is not None:
                    op.provenance['schema_reason'] = 'rule-based-no-fitted-model'
                provenance = copy.deepcopy(op.provenance)
                alignment = cast(dict[str, object], data['_alignment'])
                alignment['holdout'] = provenance
                data['holdout'] = provenance
                fitted = data.get('_fitted_params')
                if isinstance(fitted, dict):
                    cast(dict[str, object], fitted)['_holdout'] = provenance
            return result
    return guarded


def initialize_loop(sfd: object, data: pl.DataFrame | None, *, test_mode: bool,
                    project_root: str | Path | None, experiment_dir: str | Path | None) -> tuple[Manifest | None, pl.DataFrame, Callable[..., dict[str, object]] | None, Callable[..., dict[str, object]] | None]:
    factory = getattr(sfd, 'manifest', None)
    manifest = cast(Callable[[], Manifest], factory)() if callable(factory) else None
    hint = cast(str | Path | None, getattr(sfd, '_holdout_project_root', None))
    binding = bind_manifest(manifest, project_root, experiment_dir, hint) if manifest is not None else bind(project_root, experiment_dir, hint)
    if manifest is None:
        if binding is not None and binding.cutoff_ns is not None:
            raise ValueError('Holdout research refuses custom-function SFDs')
        if data is None:
            raise ValueError('UniversalExperimentLoop data parameter required for custom SFDs using custom functions approach')
        return None, data, cast(Callable[..., dict[str, object]] | None, getattr(sfd, 'prep', None)), cast(Callable[..., dict[str, object]] | None, getattr(sfd, 'model', None))
    reference = cast(Mapping[str, object], getattr(sfd, '_yaml', {}))
    config = cast(Mapping[str, object], reference.get('uel', {}))
    reject_modes(manifest, workers=config.get('workers', 1), factorize=config.get('factorize', False), test_mode=test_mode)
    if manifest.architecture_function is None:
        raise ValueError('UniversalExperimentLoop Manifest without architecture_function is not supported. Use .with_reference_architecture(func) in your manifest.')
    with operation(manifest, 'experiment'):
        frame = data if data is not None else manifest.fetch_test_data() if test_mode and manifest.test_data_source_config is not None else manifest.fetch_data()
        frame = admit(manifest, frame)
    def prep(raw: pl.DataFrame, round_params: dict[str, object] | None = None) -> dict[str, object]:
        return manifest.prepare_data(raw, round_params or {})
    def model(raw: dict[str, object], round_params: dict[str, object] | None) -> dict[str, object]:
        return manifest.run_model(raw, round_params or {})
    return manifest, frame, prep, model


def run_guard(func: Callable[Concatenate[_T, _P], _R]) -> Callable[Concatenate[_T, _P], _R]:
    @wraps(func)
    def guarded(host: _T, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        manifest = getattr(host, 'manifest', None)
        if manifest is None:
            return func(host, *args, **kwargs)
        values = inspect.signature(func).bind(host, *args, **kwargs)
        values.apply_defaults()
        call = cast(Mapping[str, object], values.arguments)
        with operation(manifest, 'experiment'):
            reject_modes(manifest, workers=call.get('workers') or getattr(host, '_workers', 1), factorize=call.get('factorize'), resume=call.get('resume'),
                         callbacks=(*[call.get(name) for name in ('params', 'prep', 'model')], getattr(host, '_intra_callback', None)))
            binding = binding_of(manifest)
            if binding is not None and binding.cutoff_ns is not None:
                attrs = cast(dict[str, object], vars(host))
                if (attrs.get('prep'), attrs.get('model')) != attrs.get('_manifest_functions'):
                    raise ValueError('Holdout research refuses replaced preparation/model functions')
                attrs['data'] = admit(manifest, cast(pl.DataFrame, attrs['data']))
            return func(host, *args, **kwargs)
    return guarded


def metadata(manifest: object) -> dict[str, object] | None:
    op = _CURRENT.get()
    if op is None or op.manifest is not manifest:
        return None
    return {key: op.provenance[key] for key in ('schema_version', 'project_id', 'reservation_revision', 'cutoff_ns', 'interval_seconds', 'research_record_id')}
