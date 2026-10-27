import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from limen.experiment import UniversalExperimentLoop
from limen.experiment import _uel_snapshot as snapshot
from tests.test_factorize_equivalence import _loop, _numeric_equal


@pytest.fixture
def completed(tmp_path):
    loop = _loop('binary_costs', tmp_path / 'recorded')
    loop.run('snapshot', n_permutations=1, prep_each_round=True,
             progress_bar=False, post_processing=True)
    return loop


@pytest.mark.parametrize('case', ['binary_costs', 'directional_barriers', 'rule_based'])
def test_recorded_experiment_snapshot_restores_captured_state(tmp_path, case):
    loop = _loop(case, tmp_path / case)
    loop.run(case, n_permutations=1, prep_each_round=True,
             progress_bar=False, post_processing=True)
    path = loop.save(tmp_path / (case + '.uel'))
    restored = UniversalExperimentLoop.load(path)
    assert restored is not loop
    assert restored.experiment_log.equals(loop.experiment_log)
    assert restored.data.equals(loop.data)
    _numeric_equal(restored.round_params, loop.round_params)
    for original, loaded in zip(loop.preds, restored.preds, strict=True):
        np.testing.assert_array_equal(original, loaded)
    assert restored.convergence_report == loop.convergence_report
    assert restored._log.experiment_log.equals(loop._log.experiment_log)
    assert restored._log._captured_log.equals(loop._log._captured_log)
    for original, loaded in zip(loop.models, restored.models, strict=True):
        if hasattr(original, 'coef_'):
            np.testing.assert_array_equal(original.coef_, loaded.coef_)
        if hasattr(original, 'get_params'):
            assert loaded.get_params() == original.get_params()
    assert len(restored.scalers) == len(loop.scalers)
    assert restored._search_strategy._seen == loop._search_strategy._seen
    assert restored._search_strategy.domain is restored._search_strategy._domain


def test_snapshot_restores_local_functions_and_nested_fitted_state(completed, tmp_path):
    offset = float(completed.data['close'][0])

    def local_function(value):
        return value + offset

    completed.extras.append({'callable': local_function, 'frame': completed.data.head(3),
                             'nested': ([completed.models[0]], (None, {'value': np.float32(offset)}))})
    restored = UniversalExperimentLoop.load(completed.save(tmp_path / 'local.uel'))
    assert restored.extras[-1]['callable'](1) == local_function(1)
    assert restored.extras[-1]['frame'].equals(completed.data.head(3))
    assert type(restored.extras[-1]['nested']) is tuple
    assert isinstance(restored.extras[-1]['nested'][1][1]['value'], np.float32)
    assert restored.extras[-1]['nested'][0][0] is restored.models[0]


def test_snapshot_fresh_process_does_not_run_research(completed, tmp_path):
    path = completed.save(tmp_path / 'fresh.uel')
    program = """
import sys
from limen.experiment import UniversalExperimentLoop
def forbidden(*args, **kwargs):
    raise AssertionError('loading executed research')
UniversalExperimentLoop.run = forbidden
UniversalExperimentLoop._finalize = forbidden
loop = UniversalExperimentLoop.load(sys.argv[1])
print(loop.experiment_log.height, len(loop.models), loop.data.height)
"""
    result = subprocess.run([sys.executable, '-c', program, str(path)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == f'{completed.experiment_log.height} {len(completed.models)} {completed.data.height}'


def test_snapshot_default_path_and_failed_serialization_are_atomic(completed, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = completed.save()
    assert path == Path('snapshot.uel')
    original = path.read_bytes()
    completed.extras.append(threading.Lock())
    with pytest.raises(TypeError, match='pickle'):
        completed.save()
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))
    completed._snapshot_ready = False
    with pytest.raises(ValueError, match='successfully returned'):
        completed.save()


@pytest.mark.parametrize('problem', ['format', 'header', 'environment', 'checksum', 'type'])
def test_snapshot_refuses_invalid_or_incompatible_artifacts(completed, tmp_path, problem):
    import cloudpickle

    path = completed.save(tmp_path / 'bad.uel')
    raw = path.read_bytes()
    header_bytes, payload = raw[len(snapshot._MAGIC):].split(b'\n', 1)
    header = json.loads(header_bytes)
    if problem == 'format':
        raw = b'unknown' + raw
    elif problem == 'header':
        raw = snapshot._MAGIC + b'[]\n' + payload
    else:
        if problem == 'environment':
            header['environment']['python'] = [0, 0, 0]
        elif problem == 'checksum':
            payload = payload[:-1]
        else:
            payload = cloudpickle.dumps({'not': 'an experiment'})
            header['sha256'] = hashlib.sha256(payload).hexdigest()
        raw = snapshot._MAGIC + json.dumps(header).encode() + b'\n' + payload
    path.write_bytes(raw)
    with pytest.raises((TypeError, ValueError), match='snapshot'):
        UniversalExperimentLoop.load(path)


def test_snapshot_refuses_dependency_drift_before_deserialization(completed, tmp_path, monkeypatch):
    path = completed.save(tmp_path / 'versions.uel')
    environment = copy.deepcopy(snapshot._environment())
    environment['packages']['numpy'] = 'incompatible'
    monkeypatch.setattr(snapshot, '_environment', lambda: environment)
    with pytest.raises(ValueError, match='dependency versions'):
        UniversalExperimentLoop.load(path)
