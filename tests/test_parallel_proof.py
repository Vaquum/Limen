from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import coverage
import pytest

from tests.run import RuntimeProfilePlugin
from tests.utils.proof_coverage import merge_worker_coverage


ROOT = Path(__file__).resolve().parents[1]
RECORDED = ('tests/test_round_storage.py', 'tests/test_nullable_trade_controls.py',
            'tests/test_param_domain.py', 'tests/test_source_interval.py')
RECORDED_CASES = (RECORDED[0] + '::test_new_runs_write_compressed_by_default',
                  RECORDED[1] + '::test_nullable_round_matches_disabled_execution',
                  RECORDED[2] + '::test_param_domain_init_validation',
                  RECORDED[3] + '::test_historical_interval_reaches_preparation_and_sensor')
PARALLEL = ('-n', '2', '--dist', 'loadfile', '--max-worker-restart=0')
EMPTY_CASE = 'tests/test_runtime_tracking.py::test_execute_test_suite_writes_profile_and_stops_on_first_failure'


def _run(directory, args, *, covered=True, launcher=None, environment=None):
    env = dict(os.environ, LIMEN_RUNTIME_PROFILE_PATH=str(directory / 'runtime.json'))
    command = [sys.executable]
    if covered:
        env.update(LIMEN_COVERAGE_RUN='1', COVERAGE_CORE='sysmon' if sys.version_info >= (3, 12) else 'ctrace',
                   COVERAGE_FILE=str(directory / '.coverage'))
        command.extend(('-m', 'coverage', 'run'))
    if environment:
        for name, value in environment.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
    command.extend(('-m', 'tests.run') if launcher is None else (str(launcher),))
    result = subprocess.run([*command, '--assert=plain', *args], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert (directory / 'runtime.json').is_file(), result.stdout + result.stderr
    return result, json.loads((directory / 'runtime.json').read_text())


@pytest.fixture(scope='module')
def recorded_proof(tmp_path_factory):
    directory = tmp_path_factory.mktemp('recorded-worker-proof')
    (directory / 'workers').mkdir()
    collected = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', *RECORDED_CASES],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
    native_ids = [line for line in collected.stdout.splitlines() if line.startswith('tests/') and '::' in line]
    assert native_ids
    launcher = directory / 'recorded_proof.py'
    launcher.write_text("""
import json
import os
from pathlib import Path
import shutil
import sys
import coverage
sys.path.insert(0, str(Path.cwd()))
import limen.experiment._round_storage
import tests.run as runner
from tests.utils.proof_coverage import DIRECTORY_ENV

destination = Path(CAPTURE_DIRECTORY)
real_assert = runner.RuntimeProfilePlugin.assert_complete
real_merge = runner.merge_worker_coverage

def retain_events(self):
    events = []
    for (worker, nodeid), reports in self._events.items():
        for report in reports:
            events.append(dict(worker_id=worker, nodeid=nodeid, when=report.when,
                               outcome=report.outcome, duration=report.duration,
                               start=report.start, stop=report.stop, longrepr=report.longreprtext))
    (destination / 'events.json').write_text(json.dumps(dict(
        controller_pid=os.getpid(), collections=self.collections, workers=self.workers, events=events)))
    real_assert(self)

def retain_databases(directory, worker_ids):
    for worker_id in worker_ids:
        shutil.copyfile(directory / worker_id, destination / 'workers' / worker_id)
    real_merge(directory, worker_ids)

runner.RuntimeProfilePlugin.assert_complete = retain_events
runner.merge_worker_coverage = retain_databases
outer = coverage.Coverage.current()
before = outer.get_option('run:disable_warnings')
environment = {name: os.getenv(name) for name in (DIRECTORY_ENV, 'COVERAGE_FILE', 'COVERAGE_CORE')}
code = runner.main()
assert outer.get_option('run:disable_warnings') == before
assert environment == {name: os.getenv(name) for name in environment}
raise SystemExit(code)
""".replace('CAPTURE_DIRECTORY', repr(str(directory))))
    result, profile = _run(directory, ('-n', '4', *PARALLEL[2:], *RECORDED_CASES, '-q'), launcher=launcher)
    assert result.returncode == 0, result.stdout + result.stderr
    return directory, native_ids, profile, json.loads((directory / 'events.json').read_text())


@pytest.fixture(scope='module')
def empty_proof(tmp_path_factory):
    directory = tmp_path_factory.mktemp('empty-worker-proof')
    result, profile = _run(directory, (*PARALLEL, EMPTY_CASE))
    assert result.returncode == 0, result.stdout + result.stderr
    return directory, result, profile


def _first_body_line(path, function):
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    body = node.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body[0].lineno


def test_parallel_proof_preserves_recorded_cases_and_worker_coverage(recorded_proof):
    directory, native_ids, profile, retained = recorded_proof
    parallel = profile['parallel']
    assert parallel['worker_count'] == 4
    assert parallel['reconciled'] is True
    assert parallel['distribution'] == 'loadfile'
    assert parallel['collected_count'] == parallel['execution_count'] == len(native_ids)
    assert parallel['collected_ids_sha256'] == hashlib.sha256(json.dumps(native_ids, separators=(',', ':')).encode()).hexdigest()
    assert sorted(record['test_name'] for record in profile['tests']) == sorted(native_ids)
    assert len({worker['pid'] for worker in parallel['workers']}) == 4
    assert all(worker['pid'] != retained['controller_pid'] for worker in parallel['workers'])
    assert sum(worker['test_count'] for worker in parallel['workers']) == len(native_ids)
    assert 0 < parallel['combine_seconds'] < profile['suite']['duration_seconds']
    final_data = coverage.CoverageData(basename=str(directory / '.coverage'))
    final_data.read()
    for worker in parallel['workers']:
        assert worker['backend'] == ('SysMonitor' if sys.version_info >= (3, 12) else 'CTracer')
        assert worker['data_sha256'] == hashlib.sha256((directory / 'workers' / worker['id']).read_bytes()).hexdigest()
        final_data.set_query_context(worker['id'])
        executed = [(name, sorted(final_data.lines(name) or [])) for name in sorted(final_data.measured_files())]
        executed = [(name, lines) for name, lines in executed if lines]
        assert sum(len(lines) for _, lines in executed) == worker['line_count'] > 0
        assert hashlib.sha256(json.dumps(executed, separators=(',', ':')).encode()).hexdigest() == worker['lines_sha256']
    witnesses = (
        (RECORDED[0], 'limen/experiment/_round_storage.py', '_members'),
        (RECORDED[1], 'limen/experiment/_resolve_trade_policy.py', 'resolve_trade_policy'),
    )
    assigned = set()
    for test_file, package_file, function in witnesses:
        workers = {record['worker_id'] for record in profile['tests'] if record['test_name'].startswith(test_file + '::')}
        assert len(workers) == 1
        worker_id = workers.pop()
        assigned.add(worker_id)
        final_data.set_query_context(worker_id)
        assert _first_body_line(package_file, function) in final_data.lines(package_file)
    assert len(assigned) == 2
    package_file = 'limen/experiment/_round_storage.py'
    tree = ast.parse((ROOT / package_file).read_text())
    constant_line = next(node.lineno for node in tree.body if isinstance(node, ast.Assign)
                         and any(isinstance(name, ast.Name) and name.id == 'FORMAT' for target in node.targets for name in ast.walk(target)))
    final_data.set_query_context('')
    assert constant_line in final_data.lines(package_file)


def test_four_worker_proof_preserves_cases_and_merged_coverage(recorded_proof):
    directory, native_ids, profile, _ = recorded_proof
    parallel = profile['parallel']
    assert parallel['worker_count'] == 4 and parallel['reconciled'] is True
    assert parallel['distribution'] == 'loadfile'
    assert parallel['collected_count'] == parallel['execution_count'] == len(native_ids) > 0
    assert parallel['collected_ids_sha256'] == hashlib.sha256(json.dumps(native_ids, separators=(',', ':')).encode()).hexdigest()
    assert sorted(record['test_name'] for record in profile['tests']) == sorted(native_ids)
    assert len({worker['pid'] for worker in parallel['workers']}) == 4
    assert sum(worker['test_count'] for worker in parallel['workers']) == len(native_ids)
    final_data = coverage.CoverageData(basename=str(directory / '.coverage'))
    final_data.read()
    for worker in parallel['workers']:
        assert worker['backend'] == ('SysMonitor' if sys.version_info >= (3, 12) else 'CTracer')
        final_data.set_query_context(worker['id'])
        executed = [(name, sorted(final_data.lines(name) or [])) for name in sorted(final_data.measured_files())]
        executed = [(name, lines) for name, lines in executed if lines]
        assert sum(len(lines) for _, lines in executed) == worker['line_count'] > 0
        assert hashlib.sha256(json.dumps(executed, separators=(',', ':')).encode()).hexdigest() == worker['lines_sha256']


def _replay(retained):
    plugin = RuntimeProfilePlugin()
    plugin.requested_workers = 4
    plugin.distribution = 'loadfile'
    plugin.collections = {worker: tuple(ids) for worker, ids in retained['collections'].items()}
    plugin.workers = copy.deepcopy(retained['workers'])
    for event in retained['events']:
        fields = {name: value for name, value in event.items() if name != 'worker_id'}
        report = pytest.TestReport(location=(event['nodeid'].split('::')[0], 0, event['nodeid']), keywords={}, **fields)
        report.worker_id = event['worker_id']
        plugin.pytest_runtest_logreport(report)
    return plugin


@pytest.mark.parametrize('fault', ['missing_completion', 'duplicate_completion', 'overlap', 'duplicate_phase', 'collection', 'receipt'])
def test_parallel_proof_rejects_incomplete_worker_evidence(recorded_proof, fault):
    retained = recorded_proof[3]
    plugin = _replay(retained)
    worker_ids = list(plugin.workers)
    if fault == 'missing_completion':
        plugin._completions.pop()
    elif fault == 'duplicate_completion':
        plugin._completions.append(plugin._completions[0])
    elif fault == 'overlap':
        worker, nodeid = plugin._completions[0]
        plugin._completions.append((next(other for other in worker_ids if other != worker), nodeid))
    elif fault == 'duplicate_phase':
        plugin.pytest_runtest_logreport(next(iter(plugin._events.values()))[0])
    elif fault == 'collection':
        plugin.collections[worker_ids[0]] = plugin.collections[worker_ids[0]][::-1]
    else:
        plugin.workers[worker_ids[0]] = None
    with pytest.raises(ValueError):
        plugin.assert_complete()
    assert plugin.build_profile()['parallel']['reconciled'] is False


@pytest.mark.parametrize('fault', ['missing', 'empty', 'digest', 'backend', 'pid', 'context'])
def test_parallel_proof_rejects_changed_coverage_databases(recorded_proof, empty_proof, tmp_path, monkeypatch, fault):
    directory, _, _, retained = recorded_proof
    copied = tmp_path / 'workers'
    shutil.copytree(directory / 'workers', copied)
    receipts = copy.deepcopy(retained['workers'])
    first, second = list(receipts)[:2]
    path = copied / first
    if fault == 'missing':
        path.unlink()
    elif fault == 'empty':
        assert receipts[first]['line_count'] > 0
        shutil.copyfile(empty_proof[0] / '.coverage', path)
        receipts[first]['data_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == 'digest':
        receipts[first]['lines_sha256'] = 'changed'
    elif fault == 'backend':
        receipts[first]['backend'] = 'unavailable'
    elif fault == 'pid':
        receipts[second]['pid'] = receipts[first]['pid']
    else:
        shutil.copyfile(copied / second, path)
        for name in ('data_sha256', 'lines_sha256', 'line_count'):
            receipts[first][name] = receipts[second][name]

    def unexpected_outer_access():
        raise AssertionError('Invalid worker evidence touched the outer collector')

    monkeypatch.setattr(coverage.Coverage, 'current', staticmethod(unexpected_outer_access))
    with pytest.raises(ValueError):
        merge_worker_coverage(copied, receipts)


def test_parallel_proof_saves_real_empty_worker_data_under_warning_errors(empty_proof):
    directory, result, profile = empty_proof
    assert profile['suite']['test_count'] == profile['suite']['passed_count'] == 1
    assert profile['parallel']['reconciled'] is True
    workers = profile['parallel']['workers']
    assert len({worker['pid'] for worker in workers}) == 2
    assert sorted(worker['test_count'] for worker in workers) == [0, 1]
    for worker in workers:
        assert worker['backend'] == ('SysMonitor' if sys.version_info >= (3, 12) else 'CTracer')
        assert len(worker['data_sha256']) == 64
        assert worker['line_count'] == 0
        assert worker['lines_sha256'] == hashlib.sha256(b'[]').hexdigest()
        assert worker['warning_options_before'] == worker['warning_options_after']
        assert worker['warning_filters_unchanged'] is True
    final_data = coverage.CoverageData(basename=str(directory / '.coverage'))
    final_data.read()
    assert all(not final_data.lines(name) for name in final_data.measured_files())
    assert 'no-data-collected' not in result.stdout + result.stderr


def test_parallel_proof_rejects_actual_pinned_backend_fallback(tmp_path):
    (tmp_path / 'conftest.py').write_text("""
import coverage
import pytest

@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    if hasattr(session.config, 'workerinput'):
        class TraceOnly(coverage.Coverage):
            def __init__(self, *args, **kwargs):
                kwargs['timid'] = True
                super().__init__(*args, **kwargs)
        coverage.Coverage = TraceOnly
""")
    test_file = tmp_path / 'test_backend.py'
    test_file.write_text('def test_backend():\n    assert True\n')
    result, profile = _run(tmp_path, (*PARALLEL, str(test_file)), environment={'LIMEN_COVERAGE_RUN': '1', 'COVERAGE_CORE': 'sysmon'})
    assert result.returncode != 0
    assert 'requires actual SysMonitor workers' in result.stdout + result.stderr
    assert profile['parallel']['reconciled'] is False


@pytest.mark.parametrize('threads', [None, '2'])
def test_parallel_proof_preserves_thread_override(tmp_path, threads):
    test_file = tmp_path / 'test_threads.py'
    test_file.write_text('import polars as pl\ndef test_threads():\n    assert pl.thread_pool_size() == ' + (threads or '1') + '\n')
    files = (str(test_file), *RECORDED_CASES[1:3]) if threads is None else (str(test_file),)
    parallel = ('-n', '2', '--dist', 'loadgroup') if threads is None else PARALLEL
    result, profile = _run(tmp_path, (*parallel, *files), covered=threads is None,
                           environment={'POLARS_MAX_THREADS': threads})
    assert result.returncode == 0, result.stdout + result.stderr
    assert profile['parallel']['worker_count'] == 2
    assert profile['parallel']['reconciled'] is True
    if threads is None:
        assert profile['parallel']['distribution'] == 'loadgroup'
        assert sum(worker['line_count'] for worker in profile['parallel']['workers']) > 0


@pytest.mark.parametrize('selection', [('-n', '0'), ('-p', 'no:xdist')])
def test_parallel_proof_preserves_explicit_covered_serial_selection(tmp_path, selection):
    result, profile = _run(tmp_path, (*selection, RECORDED[1] + '::test_nullable_search_agrees_with_validation', '-q'))
    assert result.returncode == 0, result.stdout + result.stderr
    assert profile['suite']['passed_count'] == 3
    assert 'parallel' not in profile
    assert all('worker_id' not in record for record in profile['tests'])
    data = coverage.CoverageData(basename=str(tmp_path / '.coverage'))
    data.read()
    assert data.measured_contexts() == {''}
    assert any(data.lines(name) for name in data.measured_files())


def test_parallel_proof_keeps_required_limits():
    budget = json.loads((ROOT / 'tests/runtime_budget.json').read_text())
    assert budget['max_total_seconds'] == 310
    workflow = (ROOT / '.github/workflows/pr_checks_tests.yml').read_text()
    assert 'python -m coverage run -m tests.run || test_exit=$?' in workflow
    assert 'python -m coverage report --fail-under=92' in workflow
    assert 'python -m tests.runtime_gate' in workflow and '--enforce' in workflow


def test_plain_assertions_still_fail_on_a_recorded_control(tmp_path):
    probe = tmp_path / 'test_recorded_assertion.py'
    source = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'
    probe.write_text("from limen.data import HistoricalData\n"
        "def test_observed_price_control():\n"
        f"    assert HistoricalData().get_any_file({str(source)!r})['close'][0] < 0\n")
    result, profile = _run(tmp_path, ('-n', '0', '-q', str(probe)))
    assert result.returncode == 1, result.stdout + result.stderr
    assert profile['suite']['test_count'] == profile['suite']['failed_count'] == 1
    assert 'AssertionError' in profile['tests'][0]['error']


def test_native_runner_refuses_disabled_python_assertions():
    result = subprocess.run([sys.executable, '-O', '-m', 'tests.run', '-n', '0', RECORDED_CASES[0]],
                            cwd=ROOT, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'requires enabled Python assertions' in result.stderr
