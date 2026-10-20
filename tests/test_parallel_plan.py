from __future__ import annotations

import json
import pickle
import multiprocessing
import copy
import os
import gzip
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import dataclass, fields, is_dataclass
from functools import partial

import pytest
import polars as pl

from limen._parallel_contract import ParallelRoundContract, declare_parallel, seeded_stage
from limen.experiment._parallel_encoding import decode, digest, encode, parameter_matches, source_digest
from limen.experiment._parallel_plan import ScalerFactory, freeze_manifest
from limen.experiment._parallel_resources import sklearn_stage, validate_resources, xgboost_stage, validate_domain
from limen.experiment._parallel_worker import startup_limits
from limen.experiment._parallel_evidence import record_context, validate_record, wire_record
from limen.experiment import _round_storage as storage
from limen.scalers.robust_scaler import RobustScaler
from limen.targets.random_binary import RandomBinaryTarget
from limen.yaml import parse
from limen.yaml.compiler import build_manifest
from tests.test_parallel_rounds import CallerRollingBase


def _configuration() -> dict[str, object]:
    config, errors = parse(Path('limen/yaml/templates/logreg_binary.yaml').read_text())
    assert not errors
    return config


def _parameters():
    return {'n_jobs': 1, 'random_state': 42, 'solver': 'liblinear', 'scaler_type': 'robust',
            'roc_period': 1, 'frac_diff_d': 0.0, 'q': 0.5, 'shift': -1,
            'cal_method': 'sigmoid', 'threshold_min': 0.25, 'threshold_max': 0.65,
            'threshold_step': 0.05, 'window': 2, 'range': (0.1, 0.9)}


def _caller_feature(data, window=2):
    return data.with_columns(pl.col('close').rolling_mean(window).alias('caller_mean'))


def _caller_validator(params):
    if type(params.get('window')) is not int or params['window'] < 1:
        raise ValueError('positive integer window required')


@dataclass
class CallerRolling(CallerRollingBase):
    window: int

    @classmethod
    def class_transform(cls, data, window=3):
        return _caller_feature(data, window)

    def __call__(self, data):
        return self.transform(data)


class CallerTarget(RandomBinaryTarget):
    '''Importable caller target using the native seeded noise benchmark.'''


def _target_probe(plan, params):
    from tests.test_parallel_rounds import recorded_source
    validate_resources(plan, params)
    data = copy.deepcopy(plan.manifest).prepare_data(recorded_source(), params)
    return {key: data[key].to_list() for key in ('y_train', 'y_val', 'y_test')}


def _carried_probe(plan, params):
    validate_resources(plan, params)
    from tests.test_parallel_rounds import recorded_source
    data = recorded_source().head(24).select('close')
    component = plan.manifest.feature_transforms[-1].func
    return component(data)['caller_mean'].to_list(), os.getpid()


def _frozen_probe(plan, params):
    from limen.experiment import _parallel_encoding as codec
    from limen.experiment._parallel_plan import plan_digests
    def no_discovery():
        raise AssertionError('worker repeated parent distribution discovery')
    with pytest.MonkeyPatch.context() as patch, codec.distribution_providers(plan.distribution_providers):
        patch.setattr(codec, 'package_distributions', no_discovery)
        assert plan_digests(plan.manifest, plan.stages) == (plan.pipeline_digest, plan.resource_digest)
        version = codec.metadata.version
        patch.setattr(codec.metadata, 'version', lambda name: version(name) + '-changed')
        assert plan_digests(plan.manifest, plan.stages) != (plan.pipeline_digest, plan.resource_digest)
    validate_resources(plan, params)
    from tests.test_parallel_rounds import recorded_source
    data = recorded_source().head(24).select('close')
    computations = plan.manifest.scaler[0]
    from limen.experiment import manifest_core
    resolved = manifest_core.__dict__['_resolve_params'](computations[0][2], params)
    scaler = computations[0][1](data, **resolved)
    transformed = scaler.transform(data)
    caller = next((stage for stage in plan.stages if stage.component is _caller_feature), None)
    actual = None if caller is None else caller.component(data, window=params['window'])['caller_mean'].to_list()
    return (plan.pipeline_digest, isinstance(computations[0][1], ScalerFactory),
            hasattr(_caller_feature, '__parallel_round_contract__'), os.getpid(),
            Path(__file__).resolve().as_posix(), transformed.to_dict(as_series=False), actual)


def test_frozen_manifest_round_plan() -> None:
    config = _configuration()
    original = build_manifest(config)
    plan = freeze_manifest(original, config)
    assert plan.pipeline_digest == freeze_manifest(build_manifest(config), config).pipeline_digest
    assert len(pickle.dumps(plan)) <= 65536
    declare_parallel(_caller_feature, ParallelRoundContract(_caller_validator, ('window',)))
    native = copy.deepcopy(plan.manifest)
    native.add_feature(_caller_feature, window='{window}')
    with startup_limits(), ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        cases = [(native, _parameters()), (native, _parameters() | {'scaler_type': 'rank_gauss'})]
        fixed = copy.deepcopy(native).set_scaler(RobustScaler, extra_params={'quantile_range': (0.1, 0.9)})
        dynamic = copy.deepcopy(native).set_scaler_from_params(extra_params={'quantile_range': '{range}'})
        renamed = copy.deepcopy(native).set_scaler_from_params(param_name='scale_choice')
        cases += [(fixed, _parameters()), (dynamic, _parameters()),
                  (renamed, {key: value for key, value in _parameters().items() if key != 'scaler_type'} | {'scale_choice': 'robust'})]
        for manifest, params in cases:
            frozen = freeze_manifest(manifest, None)
            actual = pool.submit(_frozen_probe, frozen, params).result(timeout=30)
            expected = _frozen_probe(frozen, params)
            assert actual[:3] == (frozen.pipeline_digest, True, False)
            assert actual[3] != os.getpid() and actual[4] == Path(__file__).resolve().as_posix()
            assert actual[5:] == expected[5:]
        for component in (partial(_caller_feature, window=3), CallerRolling(3).transform, CallerRolling(3), CallerRolling.class_transform):
            imports = ('tests.test_parallel_rounds',) if isinstance(component, CallerRolling) else ()
            declare_parallel(component, ParallelRoundContract(_caller_validator, ('window',), imports))
            carried = copy.deepcopy(plan.manifest).add_feature(component)
            frozen = freeze_manifest(carried, None)
            assert digest(decode(encode(frozen))) == digest(frozen)
            if getattr(component, '__name__', None) == 'transform':
                _assert_inherited_identity(frozen)
            actual, pid = pool.submit(_carried_probe, frozen, _parameters()).result(timeout=30)
            expected, _ = _carried_probe(frozen, _parameters())
            assert actual == expected and pid != os.getpid()
        component = CallerRolling(3)
        declare_parallel(component, ParallelRoundContract(_caller_validator, ('window',)))
        vars(component)['cached_result'] = []
        with pytest.raises(ValueError, match='undeclared runtime state'):
            freeze_manifest(copy.deepcopy(plan.manifest).add_feature(component), None)
        from tests.test_parallel_rounds import parallel_config
        declare_parallel(CallerTarget, ParallelRoundContract(seeded_stage, ('random_state',), ('numpy',)))
        target = build_manifest(parallel_config()).with_target_label('noise', CallerTarget, fit_params={'random_state': '{target_seed}'})
        frozen = freeze_manifest(target, None)
        params = _parameters() | {'target_seed': 42}
        assert pool.submit(_target_probe, frozen, params).result(timeout=30) == _target_probe(frozen, params)
    original.split_config = (2, 1, 1)
    assert plan.manifest.split_config != original.split_config
    with pytest.raises(ValueError, match='unchanged compiled YAML'):
        freeze_manifest(original, config)
    values = {'tuple': (1, 0.5, None), 'list': [1, 0.5, None], 'bool': True}
    assert decode(json.loads(json.dumps(encode(values)))) == values
    assert digest(values['tuple']) != digest(values['list'])
    assert digest(values) == digest(dict(reversed(list(values.items()))))
    _assert_plan_identity(plan)
    _assert_selector_and_metric_identity(plan)


def _assert_inherited_identity(plan):
    from limen.experiment import _parallel_encoding as codec
    from limen.experiment._parallel_plan import plan_digests
    original = codec.module_identity
    def identity(name):
        result = original(name)
        return [result[0], 'changed-inherited-source', result[2]] if name == 'tests.test_parallel_rounds' else result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(codec, 'module_identity', identity)
        changed = plan_digests(plan.manifest, plan.stages)
    assert changed[0] != plan.pipeline_digest and changed[1] != plan.resource_digest


def _assert_selector_and_metric_identity(plan):
    from limen.experiment import _parallel_encoding as codec, _parallel_plan as plans
    fixed = freeze_manifest(copy.deepcopy(plan.manifest).set_scaler(RobustScaler), None)
    registry = list(plans.SCALER_REGISTRY.items())
    swapped = dict([(registry[1][0], registry[0][1]), (registry[0][0], registry[1][1]), *registry[2:]])
    assert list(swapped.values()) == list(plans.SCALER_REGISTRY.values())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plans, 'SCALER_REGISTRY', swapped)
        changed = plans.plan_digests(plan.manifest, plan.stages)
        assert changed[0] != plan.pipeline_digest and changed[1] != plan.resource_digest
        assert plans.plan_digests(fixed.manifest, fixed.stages) == (fixed.pipeline_digest, fixed.resource_digest)
    config = _configuration()
    config['sfd']['manifest']['calibration']['threshold_function']['params'] = {}
    default_metric = freeze_manifest(build_manifest(config), config)
    original = codec.module_identity
    def identity(name):
        result = original(name)
        return [result[0], 'changed-default-metric', result[2]] if name == 'limen.metrics.balanced_metric' else result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(codec, 'module_identity', identity)
        patch.setattr(plans, 'module_identity', identity)
        changed = plans.plan_digests(default_metric.manifest, default_metric.stages)
    assert changed[0] != default_metric.pipeline_digest and changed[1] != default_metric.resource_digest


def _assert_plan_identity(plan):
    from tests.test_parallel_rounds import recorded_source
    import limen.experiment._parallel_encoding as codec
    import limen.experiment._parallel_plan as plans
    def check(value):
        assert not isinstance(value, (pl.DataFrame, pl.LazyFrame)) and not hasattr(value, 'read')
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                check(getattr(value, field.name))
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item)
        assert not any(name.endswith('_') and not name.startswith('_') for name in vars(value)) if hasattr(value, '__dict__') else True
    check(plan)
    source = recorded_source()
    expected = source_digest((source, None, None))
    assert source_digest((pl.concat([source.head(10), source.slice(10)], rechunk=False), None, None)) == expected
    for changed in (source.reverse(), source.slice(1), source.select(reversed(source.columns))):
        assert source_digest((changed, None, None)) != expected
    original = codec.module_identity
    for module, changes in (('limen.scalers.robust_scaler', True), ('limen.indicators._atr', True), ('limen.indicators.obv', False)):
        def identity(name, module=module):
            result = original(name)
            return [result[0], 'changed-source', result[2]] if name == module else result
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(codec, 'module_identity', identity)
            patch.setattr(plans, 'module_identity', identity)
            altered = freeze_manifest(build_manifest(_configuration()), _configuration())
            assert (altered.pipeline_digest != plan.pipeline_digest) is changes
            assert (altered.resource_digest != plan.resource_digest) is changes
    cyclic = copy.deepcopy(plan.manifest)
    cyclic.params = {}  # Deliberately undeclared state is rejected rather than transported.
    with pytest.raises(ValueError, match='undeclared runtime state'):
        freeze_manifest(cyclic, None)


def test_effective_resources_and_output_evidence(tmp_path) -> None:
    declare_parallel(_caller_feature, ParallelRoundContract(_caller_validator, ('window',)))
    config = _configuration()
    plan = freeze_manifest(build_manifest(config), config)
    params = _parameters()
    validate_resources(plan, params)
    validate_domain(plan, {key: [value] for key, value in params.items()}, None)
    _assert_resource_projection(plan, params)
    class_weights = {0: 1, 1: 2}
    assert decode(encode(class_weights)) == class_weights
    validate_resources(plan, params | {'class_weight': class_weights})
    assert parameter_matches('(1, 2)', [1, 2], legacy=True)
    assert not parameter_matches('(1, 2)', [1, 2])
    ablation = copy.deepcopy(plan.manifest).set_feature_ablation()
    validate_resources(freeze_manifest(ablation, None), params | {'feature_drop_seed': None})
    disabled = copy.deepcopy(plan.manifest).add_feature(_caller_feature, include_if='optional', window='{unused_window}')
    without_calibration = {key: value for key, value in params.items() if not key.startswith(('cal_', 'threshold_'))}
    validate_resources(freeze_manifest(disabled, None), without_calibration | {'use_calibration': False, 'use_threshold': False})
    with pytest.raises(ValueError, match=r'feature_transforms.*unresolved parameter unused_window'):
        validate_resources(freeze_manifest(disabled, None), params | {'optional': True})
    from limen.indicators.obv import obv
    from limen.features.log_returns import log_returns
    validate_resources(freeze_manifest(copy.deepcopy(plan.manifest).add_feature(obv).add_feature(log_returns), None), params)
    for jobs in (None, True, 0, -1, 2):
        with pytest.raises(ValueError, match='effective n_jobs=1'):
            validate_resources(plan, params | {'n_jobs': jobs})
    for seed in (None, True, '42'):
        with pytest.raises(ValueError, match='fixed integer seed'):
            validate_resources(plan, params | {'random_state': seed})
    with pytest.raises(ValueError, match='fixed integer seed'):
        validate_domain(plan, {'n_jobs': [1], 'solver': ['lbfgs', 'saga'], 'random_state': [None],
                               **{key: [value] for key, value in params.items() if key not in ('n_jobs', 'solver', 'random_state')}}, None)
    sklearn_stage({'solver': 'lbfgs', 'random_state': None})
    for option in ({'device': 'cuda'}, {'tree_method': 'gpu_hist'}, {'sampling_method': 'gradient_based'}):
        with pytest.raises(ValueError, match='CPU backend required'):
            xgboost_stage({'n_jobs': 1, 'random_state': 42} | option)
    for value in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='nonfinite'):
            encode(value)
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match='cycles'):
        encode(cycle)
    target = copy.deepcopy(plan.manifest)
    target.with_target_label('noise', RandomBinaryTarget, fit_params={'random_state': '{target_seed}'})
    seeded = freeze_manifest(target, None)
    validate_resources(seeded, params | {'target_seed': 42})
    for seed in (None, True):
        with pytest.raises(ValueError, match='target: fixed integer seed'):
            validate_resources(seeded, params | {'target_seed': seed})
    with pytest.raises(ValueError, match='target: fixed integer seed'):
        validate_domain(seeded, {key: [value] for key, value in params.items()}, {'target_seed': None})
    omitted = copy.deepcopy(plan.manifest).with_target_label('noise', RandomBinaryTarget)
    with pytest.raises(ValueError, match='target: fixed integer seed'):
        validate_resources(freeze_manifest(omitted, None), params)
    _assert_dispatch_refusal(plan, seeded, params)
    from tests.test_parallel_rounds import assert_runs_equal, parallel_config, records, run_loop
    configuration = parallel_config()
    configuration['uel']['n_permutations'] = 2
    configuration['sfd']['params']['tuple_tag'] = [(1, 2), (3, 4)]
    configuration['sfd']['params']['target_seed'] = [42]
    configuration['sfd']['manifest']['target'] = {'name': 'noise', 'class': 'limen.targets.RandomBinaryTarget',
                                               'fit_params': {'random_state': '{target_seed}'}}
    configuration['uel'].pop('pruning_strategies')
    directory = tmp_path / 'recorded'
    loop = run_loop(configuration, directory, workers=2)
    serial = tmp_path / 'serial-target'
    run_loop(configuration, serial, workers=1)
    assert_runs_equal(directory, serial)
    saved = records(directory)
    assert len(saved) == 2 and all(type(row['round_params']['tuple_tag']) is tuple for row in saved)
    contract = next(stage.contract for stage in vars(loop)['_parallel_plan'].stages if stage.path == 'architecture')
    for row in saved:
        validate_record(row, record_context(row), contract, execution=True, outputs=True, snapshot=True)
        assert wire_record(wire_record(row), restore=True)['round_params'] == row['round_params']
        assert row['parallel_output']['count'] == len(row['alignment']['test_datetimes']) == len(row['preds'])
    for fault in ('count', 'kind', 'digest', 'probabilities', 'prediction', 'occurrence', 'params', 'missing'):
        row = copy.deepcopy(saved[0])
        if fault in ('count', 'kind', 'digest'):
            row['parallel_output'][fault] = 0 if fault == 'count' else 'incorrect'
        elif fault == 'probabilities':
            row['probs'] = None
        elif fault == 'prediction':
            row['preds'][0] = 1 - row['preds'][0]
        elif fault == 'occurrence':
            row['round_id'] = saved[1]['round_id']
        elif fault == 'params':
            row['round_params']['C'] *= 2
        else:
            row.pop('preds')
        with pytest.raises(ValueError, match=r'output|digest|evidence'):
            validate_record(row, record_context(row), contract, execution=True, outputs=True)
    rejected = tmp_path / 'corrupt'
    shutil.copytree(directory, rejected)
    corrupted = records(rejected, raw=True)
    corrupted[0]['execution']['net'][0] += 0.01  # Deliberate saved-ledger corruption.
    storage.round_path(rejected).write_bytes(b''.join(gzip.compress((json.dumps(row) + '\n').encode(), mtime=0) for row in corrupted))
    before = {str(path.relative_to(rejected)): path.read_bytes() for path in rejected.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='digest'):
        run_loop(configuration, rejected, resume=True)
    assert {str(path.relative_to(rejected)): path.read_bytes() for path in rejected.rglob('*') if path.is_file()} == before
    resumed = run_loop(configuration, directory, resume=True)
    assert resumed.experiment_log.height == 2
    assert all(type(row['round_params']['tuple_tag']) is tuple for row in records(directory))
    from limen.experiment.checkpoint_manager import CheckpointManager
    checkpoint = CheckpointManager().load(directory)
    assert checkpoint['domain_state']['tuple_tag'] == [(1, 2), (3, 4)]
    _assert_recovery_faults(configuration, directory, tmp_path)
    _assert_legacy_recovery(configuration, directory, tmp_path)
    _assert_native_output_forms()


def _assert_resource_projection(plan, params):
    from limen.experiment import _parallel_resources as resources
    original = vars(resources)['_validate_stage']
    calls = 0
    def counted(*args):
        nonlocal calls
        calls += 1
        assert calls <= 500, 'unrelated grid axes multiplied resource validation'
        return original(*args)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(resources, '_validate_stage', counted)
        for additions in ({f'unconsumed_{index}': list(range(10)) for index in range(10)},
                          {key: list(range(10)) for key in ('nthread', 'num_threads', 'device', 'device_type', 'seed')}):
            validate_domain(plan, {key: [value] for key, value in params.items()} | additions, None)
    conditional = copy.deepcopy(plan.manifest).add_feature(_caller_feature, include_if='optional', window='{candidate_window}')
    domain = {key: [value] for key, value in params.items()} | {'optional': [False, True], 'candidate_window': [2, 0]}
    with pytest.raises(ValueError, match=r'feature_transforms.*positive integer window'):
        validate_domain(freeze_manifest(conditional, None), domain, None)


def _assert_dispatch_refusal(plan, seeded, params):
    from importlib import import_module
    from tests.test_parallel_rounds import recorded_source
    from limen.experiment import _parallel_worker as worker
    from limen.experiment.manifest_core import MLManifest
    tabpfn_module = import_module('limen.sfd.reference_architecture.tabpfn_binary')
    def no_preparation(*_args, **_kwargs):
        raise AssertionError('unsafe candidate reached preparation or optional backend')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MLManifest, 'prepare_data', no_preparation)
        patch.setattr(worker, '_worker_sources', worker.FrozenRoundSources(recorded_source(), None, None))
        for selected, candidate, error in ((plan, params | {'n_jobs': None}, 'effective n_jobs=1'),
                                          (seeded, params | {'target_seed': None}, 'fixed integer seed')):
            patch.setattr(worker, '_worker_plan', selected)
            with pytest.raises(RuntimeError, match=error):
                worker.evaluate_round(candidate | {'_id': 'refused-control', '_round_index': 0})
        tab_config, errors = parse(Path('limen/yaml/templates/tabpfn_binary.yaml').read_text())
        assert not errors
        tab_plan = freeze_manifest(build_manifest(tab_config), tab_config)
        patch.setattr(tabpfn_module, 'require_optional', no_preparation)
        tab_params = {key: values[0] for key, values in tab_config['sfd']['params'].items()}
        with pytest.raises(ValueError, match='architecture: unfrozen external state'):
            validate_resources(tab_plan, tab_params)


def _assert_recovery_faults(configuration, directory, temporary):
    from tests.test_parallel_rounds import run_loop
    for fault in ('missing_plan', 'version', 'marker', 'worker', 'resource', 'downgraded'):
        rejected = temporary / fault
        shutil.copytree(directory, rejected)
        path = rejected / 'metadata.json'
        metadata = json.loads(path.read_text())
        if fault == 'missing_plan':
            metadata.pop('parallel_plan')
        elif fault == 'version':
            metadata['limen_version'] = 'unknown'
        elif fault == 'downgraded':
            metadata['limen_version'] = '5.45.2'
        elif fault == 'worker':
            metadata['worker_native_threads'] = 2
        elif fault == 'marker':
            metadata['parallel_plan']['version'] = True
        else:
            metadata['parallel_plan']['resource_digest'] = 'changed-control'
        path.write_text(json.dumps(metadata))
        before = {str(item.relative_to(rejected)): item.read_bytes() for item in rejected.rglob('*') if item.is_file()}
        with pytest.raises(ValueError, match=r'identity|version|policy'):
            run_loop(configuration, rejected, resume=True)
        assert {str(item.relative_to(rejected)): item.read_bytes() for item in rejected.rglob('*') if item.is_file()} == before


def _assert_legacy_recovery(configuration, directory, temporary):
    from limen.experiment.checkpoint_manager import CheckpointManager
    from tests.test_parallel_rounds import records, run_loop
    legacy = temporary / 'legacy'
    shutil.copytree(directory, legacy)
    path = legacy / 'metadata.json'
    metadata = json.loads(path.read_text())
    metadata.update(limen_version='5.45.2')
    metadata.pop('parallel_plan')
    path.write_text(json.dumps(metadata))
    checkpoint = CheckpointManager().load(legacy)
    checkpoint.pop('parameter_encoding')
    checkpoint['metadata']['content_hash'] = CheckpointManager.compute_content_hash(configuration['sfd']['params'])
    (legacy / 'checkpoint.json').write_text(json.dumps(checkpoint))
    converted = [wire_record(record, legacy=True) for record in records(directory)]
    storage.round_path(legacy).write_bytes(b''.join(gzip.compress((json.dumps(record)+'\n').encode(),mtime=0) for record in converted))
    resumed = run_loop(configuration, legacy, resume=True)
    assert resumed.experiment_log.height == 2
    assert all('parallel_plan' not in record and record['round_params']['tuple_tag'] in [[1,2],[3,4]] for record in records(legacy))


def _assert_native_output_forms():
    from threadpoolctl import threadpool_limits
    from tests.test_parallel_rounds import recorded_source
    from tests.test_record_model_outputs import _config
    from limen.experiment._parallel_plan import plan_identity
    from limen.experiment._parallel_evidence import round_record
    from limen.experiment.manifest_core import MLManifest, ProductConfig
    from limen.sfd.reference_architecture.direction_sizing import direction_sizing
    from limen.targets.trade_outcome import TradeOutcomeTarget

    source = recorded_source()
    continuous = build_manifest(_config('ridge'))
    rule_config, errors = parse(Path('limen/yaml/templates/rule_based.yaml').read_text())
    assert not errors
    rule = build_manifest(rule_config)
    rule.split_dates = continuous.split_dates
    exposure = MLManifest().set_data_source(recorded_source, params={'kline_size': 3600})
    exposure.set_split_config(6, 2, 2)
    exposure.with_target_label('outcome', TradeOutcomeTarget).with_reference_architecture(direction_sizing)
    exposure.set_backtest_config(prediction_mode='target_exposure', product=ProductConfig('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0), max_holding_seconds=7200)
    cases = [(continuous, {'alpha': 0.1, 'solver': 'svd'}, 'continuous'),
             (rule, {key: values[0] for key, values in rule_config['sfd']['params'].items()}, 'binary'),
             (exposure, {'seed': 42, 'folds': 3, 'min_train_samples': 20, 'sizing_params': {'solver': 'svd'}}, 'target_exposure'),
             (copy.deepcopy(exposure), {'seed': 42, 'folds': 3, 'min_train_samples': 20, 'sizing_params': {'solver': 'svd'}, 'conditional_size': True}, 'target_exposure')]
    for manifest, params, kind in cases:
        plan = freeze_manifest(manifest, None)
        assert digest(decode(encode(plan.manifest))) == digest(plan.manifest)
        validate_resources(plan, params)
        vars(manifest)['_parallel_evidence'] = True
        with threadpool_limits(1):
            data = manifest.prepare_data(source, params)
            data.update(_record_execution=True, _record_model_outputs=True)
            results = manifest.run_model(data, params)
        alignment = {key: value for key, value in data['_alignment'].items() if not key.startswith('_')}
        contract = next(stage.contract for stage in plan.stages if stage.path == 'architecture')
        count = data['x_test'].height if isinstance(manifest, MLManifest) else data['test'].height
        record = round_record({'_id': f'private-{kind}', '_round_index': 0}, params, results, alignment,
                              plan_identity(plan, (source, None, None)), contract, count=count, execution=True, outputs=True)
        assert record['parallel_output']['kind'] == kind and record['parallel_output']['count'] == count
        assert record['probs'] is None and record['parallel_output']['has_probabilities'] is False
        validate_record(record, record_context(record), contract, execution=True, outputs=True,
                        event=kind == 'target_exposure')
