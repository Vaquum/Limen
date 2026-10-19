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
from typing import Concatenate, ParamSpec, Protocol, TypeVar, cast
import sys

import polars as pl

from limen.experiment._holdout_history import ProjectBinding, admission, bind, check_binding, finish_research, start_research
from limen.experiment._holdout_source import Bounds, SourceEvidence, SourceManifest, admit_frame, bounds, frame_evidence, load_admitted_source, source_spec

class ResearchManifest(SourceManifest, Protocol):
    @property
    def backtest_config(self) -> object | None: ...
    @property
    def split_dates(self) -> tuple[object, ...] | None: ...
    @property
    def split_walk_forward(self) -> object | None: ...
    @property
    def bar_formation(self) -> object | None: ...
    @property
    def test_data_source_config(self) -> object | None: ...
    @property
    def architecture_function(self) -> object | None: ...
    def fetch_data(self) -> pl.DataFrame: ...
    def fetch_test_data(self) -> pl.DataFrame: ...
    def prepare_data(self, raw_data: pl.DataFrame, round_params: dict[str, object]) -> dict[str, object]: ...
    def run_model(self, data: dict[str, object], round_params: dict[str, object]) -> dict[str, object]: ...

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
    provenance: dict[str, object] = field(default_factory=dict[str, object])


_CURRENT: ContextVar[_Operation | None] = ContextVar('holdout_operation', default=None)


def binding_of(manifest: object) -> ProjectBinding | None:
    return cast(ProjectBinding | None, getattr(manifest, '_holdout_binding', None))


def bind_manifest(manifest: object, *hints: str | Path | None) -> ProjectBinding | None:
    binding = bind(*hints)
    previous = binding_of(manifest)
    if previous is not None and previous != binding:
        raise ValueError('Holdout manifest binding/revision changed; construct a new manifest')
    if binding is not None:
        vars(manifest)['_holdout_binding'] = binding
    return binding


def reject_modes(manifest: object, *, workers: object = 1, factorize: object = False,
                 resume: object = False, callbacks: tuple[object, ...] = (), test_mode: bool = False,
                 profile: bool = False) -> None:
    binding = binding_of(manifest)
    if binding is None or binding.cutoff_ns is None:
        return
    core = vars(sys.modules['limen.experiment.manifest_core'])
    if type(manifest) not in (core['MLManifest'], core['RuleBasedManifest']):
        raise ValueError('Holdout research requires a built-in manifest')
    model = cast(ResearchManifest, manifest)
    if (workers != 1 or bool(factorize) or bool(resume) or profile or any(value is not None for value in callbacks)
            or model.split_walk_forward is not None or model.bar_formation is not None
            or (test_mode and model.test_data_source_config is not None)
            or getattr(model, 'data_dict_extension', None) is not None):
        raise ValueError('Holdout research refuses profile, resume, parallel/factorized/walk-forward, alternate test source and callbacks')
    if any(name in vars(model) for name in ('prepare_data', '_prepare_data_with_fitted', 'run_model', 'compute_test_bars', 'fetch_data')):
        raise ValueError('Holdout research refuses replaced manifest methods')
    config = model.backtest_config
    if config is not None and (getattr(config, 'product', None) is not None or getattr(config, 'prediction_mode', 'binary') != 'binary' or getattr(config, 'execution_data_source', None) is not None or getattr(config, 'funding', None) is not None):
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
    _ = source_spec(model, binding)


def _base(op: _Operation) -> dict[str, object]:
    empty: Bounds = {'first_open_ns': None, 'last_close_ns': None, 'rows': 0, 'gaps': []}
    return {'schema_version': 1, 'project_id': op.binding.project_id, 'reservation_revision': op.binding.revision,
            'cutoff_ns': op.binding.cutoff_ns, 'interval_seconds': op.binding.interval_seconds,
            'research_record_id': op.record_id, 'role': op.role, 'experiment_record_id': op.parent_id,
            'round_id': getattr(op.manifest, '_holdout_round_id', None), 'source': None, 'selector': None, 'selected': empty,
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


def fetch(manifest: ResearchManifest, resolve: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    binding = binding_of(manifest)
    if binding is None:
        binding = bind_manifest(manifest)
    if binding is None or binding.cutoff_ns is None:
        if manifest.data_source_config is None:
            raise ValueError('Manifest No data source configured')
        with admission(binding.root) if binding is not None else nullcontext():
            if binding is not None:
                check_binding(binding)
            return resolve()
    with operation(manifest, 'experiment') as op:
        frame, source = load_admitted_source(manifest, binding, lower_ns=None, upper_ns=binding.cutoff_ns, complete=False)
        vars(manifest)['_holdout_source'] = source
        if op is not None:
            op.provenance['source'] = source
        return frame


def admit(manifest: object, raw: pl.DataFrame, *, force_caller: bool = False) -> pl.DataFrame:
    binding = binding_of(manifest)
    if binding is None or binding.cutoff_ns is None:
        return raw
    check_binding(binding)
    frame, evidence = frame_evidence(raw, binding)
    previous = cast(SourceEvidence | None, getattr(manifest, '_holdout_source', None))
    if not force_caller and previous is not None and previous['admitted_sha256'] == evidence['admitted_sha256']:
        evidence = previous
    op = _CURRENT.get()
    if op is not None and op.manifest is manifest:
        op.provenance['source'] = evidence
    vars(manifest)['_holdout_source'] = evidence
    return frame


def selected(manifest: object, raw: pl.DataFrame, splits: list[pl.DataFrame], params: Mapping[str, object], resolved: Mapping[str, object] | None = None) -> None:
    op = _CURRENT.get()
    if op is None or op.manifest is not manifest or op.binding.interval_seconds is None:
        return
    selector = getattr(manifest, 'pre_split_data_selector', None)
    declaration = None
    if selector is not None:
        func, config = cast(tuple[Callable[..., object], Mapping[str, object]], selector)
        declaration = {'method': f'{func.__module__}.{func.__qualname__}', 'params': dict(resolved or {}), 'seed': params.get('seed', config.get('seed'))}
    op.provenance.update(selector=declaration, selected=bounds(raw, op.binding.interval_seconds),
                         splits={name: bounds(frame, op.binding.interval_seconds) for name, frame in zip(('train', 'validation', 'test'), splits, strict=True)})


def capture_schema(manifest: object, frame: pl.DataFrame, stage: str) -> None:
    op = _CURRENT.get()
    if op is None or op.manifest is not manifest:
        return
    excluded = {'datetime', getattr(manifest, 'target_column', None), '__trade_available_at_ns__', '__backtest_source_row__'}
    op.provenance[f'{stage}_schema'] = [{'name': name, 'dtype': str(dtype)} for name, dtype in frame.schema.items() if name not in excluded]
    op.provenance['schema_reason'] = None


def preparation(func: Callable[Concatenate[_T, _P], _R]) -> Callable[Concatenate[_T, _P], _R]:
    @wraps(func)
    def guarded(manifest: _T, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        if binding_of(manifest) is None:
            _ = bind_manifest(manifest)
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
            return cast(_R, result)
    return guarded


def initialize_loop(host: object, sfd: object, data: pl.DataFrame | None, *, test_mode: bool,
                    project_root: str | Path | None, experiment_dir: str | Path | None) -> tuple[ResearchManifest | None, pl.DataFrame, Callable[..., dict[str, object]] | None, Callable[..., dict[str, object]] | None]:
    factory = getattr(sfd, 'manifest', None)
    manifest = cast(Callable[[], ResearchManifest], factory)() if callable(factory) else None
    hint = cast(str | Path | None, getattr(sfd, '_holdout_project_root', None))
    binding = bind_manifest(manifest, project_root, experiment_dir, hint) if manifest is not None else bind(project_root, experiment_dir, hint)
    if manifest is None:
        if binding is not None:
            vars(host)['_holdout_binding'] = binding
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
        frame = admit(manifest, frame, force_caller=data is not None)
        if binding is not None and binding.cutoff_ns is not None:
            vars(host)['_holdout_frame'] = frame
    def prep(data: pl.DataFrame, round_params: dict[str, object] | None = None) -> dict[str, object]:
        return manifest.prepare_data(data, round_params or {})
    def model(data: dict[str, object], round_params: dict[str, object] | None) -> dict[str, object]:
        return manifest.run_model(data, round_params or {})
    return manifest, frame, prep, model


def run_guard(func: Callable[Concatenate[_T, _P], _R]) -> Callable[Concatenate[_T, _P], _R]:
    @wraps(func)
    def guarded(host: _T, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        manifest = getattr(host, 'manifest', None)
        if manifest is None:
            binding = binding_of(host)
            with admission(binding.root) if binding is not None else nullcontext():
                if binding is not None:
                    check_binding(binding)
                    if binding.cutoff_ns is not None:
                        raise ValueError('Holdout research refuses custom-function SFDs')
                return func(host, *args, **kwargs)
        if binding_of(manifest) is None:
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
                attrs['data'] = admit(manifest, cast(pl.DataFrame, attrs['data']), force_caller=attrs['data'] is not attrs.get('_holdout_frame'))
                attrs['_holdout_frame'] = attrs['data']
            return func(host, *args, **kwargs)
    return guarded


def metadata(manifest: object) -> dict[str, object] | None:
    op = _CURRENT.get()
    if op is None or op.manifest is not manifest:
        return None
    return {key: op.provenance[key] for key in ('schema_version', 'project_id', 'reservation_revision', 'cutoff_ns', 'interval_seconds', 'research_record_id')}


def initialize_trainer(host: object, data: pl.DataFrame | None, *, project_root: str | Path | None) -> pl.DataFrame:
    attrs = cast(dict[str, object], vars(host))
    manifest = cast(ResearchManifest, attrs['_manifest'])
    binding = bind_manifest(manifest, project_root, cast(Path, attrs['_experiment_dir']))
    saved = cast(Mapping[str, object], attrs['_metadata']).get('holdout')
    if isinstance(saved, Mapping):
        saved = cast(Mapping[str, object], saved)
        if binding is None or any(saved.get(key) != value for key, value in (
            ('project_id', binding.project_id), ('reservation_revision', binding.revision),
            ('cutoff_ns', binding.cutoff_ns), ('interval_seconds', binding.interval_seconds))):
            raise ValueError('Holdout Trainer metadata binding/revision changed')
    with operation(manifest, 'trainer'):
        frame = admit(manifest, data if data is not None else manifest.fetch_data(), force_caller=data is not None)
        if binding is not None and binding.cutoff_ns is not None:
            attrs['_holdout_training_frame'] = frame
        return frame


def trainer_round(manifest: object, expected: Mapping[str, object]) -> None:
    binding = binding_of(manifest)
    if binding is None or binding.cutoff_ns is None:
        return
    vars(manifest)['_holdout_round_id'] = expected.get('round_id')
    recorded = expected.get('holdout')
    if isinstance(recorded, Mapping):
        recorded = cast(Mapping[str, object], recorded)
        if recorded.get('project_id') != binding.project_id or recorded.get('reservation_revision') != binding.revision:
            raise ValueError('Holdout Trainer round binding/revision changed')
        vars(manifest)['_holdout_experiment_record'] = recorded.get('research_record_id')
    else:
        vars(manifest)['_holdout_experiment_record'] = None


def train_guard(func: Callable[Concatenate[_T, _P], _R]) -> Callable[Concatenate[_T, _P], _R]:
    @wraps(func)
    def guarded(host: _T, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        manifest = cast(dict[str, object], vars(host))['_manifest']
        with operation(manifest, 'trainer'):
            attrs = cast(dict[str, object], vars(host))
            attrs['_data'] = admit(manifest, cast(pl.DataFrame, attrs['_data']), force_caller=attrs['_data'] is not attrs.get('_holdout_training_frame'))
            if binding_of(manifest) is not None:
                attrs['_holdout_training_frame'] = attrs['_data']
            return func(host, *args, **kwargs)
    return guarded


def replay_guard(func: Callable[Concatenate[_T, _P], _R]) -> Callable[Concatenate[_T, _P], _R]:
    @wraps(func)
    def guarded(host: _T, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        manifest = getattr(host, 'manifest', None)
        if manifest is None:
            return func(host, *args, **kwargs)
        with operation(manifest, 'replay'):
            attrs = cast(dict[str, object], vars(host))
            raw = attrs.get('data')
            if isinstance(raw, pl.DataFrame):
                admitted = admit(manifest, raw)
                if not admitted.equals(raw):
                    raise ValueError('Holdout replay data geometry changed; do not shorten retained returns')
            return func(host, *args, **kwargs)
    return guarded


def bind_round(manifest: object, provenance: Mapping[str, object], round_id: str) -> dict[str, object]:
    binding = binding_of(manifest)
    if binding is None or binding.cutoff_ns is None:
        raise ValueError('Round provenance requires active holdout binding')
    value = copy.deepcopy(dict(provenance))
    value['round_id'] = round_id
    value['research_record_id'] = start_research(binding, 'experiment')
    finish_research(binding, str(value['research_record_id']), value)
    return value


_Y = TypeVar('_Y', bound=Mapping[str, object])


def initialize_compiled(reference: _Y, project_root: str | Path | None) -> tuple[_Y, str | Path | None, str]:
    metadata = cast(Mapping[str, object], reference['metadata'])
    return reference, project_root, f"yaml:{metadata['name']}"


def cached_manifest(host: object, builder: Callable[[_Y], _T]) -> _T:
    attrs = cast(dict[str, object], vars(host))
    if attrs['_manifest_cache'] is None:
        manifest = builder(cast(_Y, attrs['_yaml']))
        _ = bind_manifest(manifest, cast(str | Path | None, attrs['_holdout_project_root']))
        attrs['_manifest_cache'] = manifest
    return cast(_T, attrs['_manifest_cache'])


def prepare_walk_forward(host: object, raw: pl.DataFrame, round_params: Mapping[str, object] | None,
                         resolve_params: Callable[[Mapping[str, object], Mapping[str, object]], dict[str, object]],
                         requires_validation: Callable[..., bool]) -> dict[str, object]:
    manifest = getattr(host, 'manifest', None)
    reject_modes(manifest)
    params = dict(round_params or {})
    selector = getattr(manifest, 'pre_split_data_selector', None)
    if selector is not None:
        func, base = cast(tuple[Callable[..., pl.DataFrame], Mapping[str, object]], selector)
        raw = func(raw, **resolve_params(base, params))
    preflight = cast(Callable[..., None], getattr(host, '_preflight', None))
    preflight(raw, require_validation=requires_validation(manifest, params))
    return {'_walk_forward_params': params, '_walk_forward_raw': raw}


def selected_input(manifest: object, raw: pl.DataFrame) -> pl.DataFrame:
    binding = binding_of(manifest)
    if binding is None or binding.cutoff_ns is None or binding.interval_seconds is None:
        return raw
    admitted = admit_frame(raw, interval=binding.interval_seconds, lower_ns=None, upper_ns=binding.cutoff_ns)
    if admitted.height != raw.height:
        raise ValueError('Holdout selector returned observations outside the research prefix')
    return admitted
