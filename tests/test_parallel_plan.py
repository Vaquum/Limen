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

import pytest
import polars as pl

from limen._parallel_contract import ParallelRoundContract, declare_parallel
from limen.experiment._parallel_encoding import decode, digest, encode
from limen.experiment._parallel_plan import ScalerFactory, freeze_manifest
from limen.experiment._parallel_resources import sklearn_stage, validate_resources, xgboost_stage, validate_domain
from limen.experiment._parallel_worker import startup_limits
from limen.experiment._parallel_evidence import record_context, validate_record, wire_record
from limen.experiment import _round_storage as storage
from limen.scalers.robust_scaler import RobustScaler
from limen.targets.random_binary import RandomBinaryTarget
from limen.yaml import parse
from limen.yaml.compiler import build_manifest


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


def _frozen_probe(plan, params):
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
        cases += [(fixed, _parameters()), (dynamic, _parameters())]
        for manifest, params in cases:
            frozen = freeze_manifest(manifest, None)
            actual = pool.submit(_frozen_probe, frozen, params).result(timeout=30)
            expected = _frozen_probe(frozen, params)
            assert actual[:3] == (frozen.pipeline_digest, True, False)
            assert actual[3] != os.getpid() and actual[4] == Path(__file__).resolve().as_posix()
            assert actual[5:] == expected[5:]
    original.split_config = (2, 1, 1)
    assert plan.manifest.split_config != original.split_config
    with pytest.raises(ValueError, match='unchanged compiled YAML'):
        freeze_manifest(original, config)
    values = {'tuple': (1, 0.5, None), 'list': [1, 0.5, None], 'bool': True}
    assert decode(json.loads(json.dumps(encode(values)))) == values
    assert digest(values['tuple']) != digest(values['list'])
    assert digest(values) == digest(dict(reversed(list(values.items()))))


def test_effective_resources_and_output_evidence(tmp_path) -> None:
    config = _configuration()
    plan = freeze_manifest(build_manifest(config), config)
    params = _parameters()
    validate_resources(plan, params)
    validate_domain(plan, {key: [value] for key, value in params.items()}, None)
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
    from tests.test_parallel_rounds import parallel_config, records, run_loop
    configuration = parallel_config()
    configuration['uel']['n_permutations'] = 2
    configuration['sfd']['params']['tuple_tag'] = [(1, 2), (3, 4)]
    configuration['uel'].pop('pruning_strategies')
    directory = tmp_path / 'recorded'
    loop = run_loop(configuration, directory, workers=2)
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
        with pytest.raises(ValueError, match='output|digest|evidence'):
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
    _assert_native_output_forms()


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
             (exposure, {'seed': 42, 'folds': 3, 'min_train_samples': 20}, 'target_exposure')]
    for manifest, params, kind in cases:
        plan = freeze_manifest(manifest, None)
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
