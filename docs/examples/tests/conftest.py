from importlib.util import module_from_spec, spec_from_file_location
import sys
import os
import json
import subprocess
from pathlib import Path

import pytest

for variable in ('POLARS_MAX_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'docs/examples'))
sys.path.insert(0, str(ROOT / 'docs/examples/tests'))


def module_at(name, path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError('No executable documentation module: ' + str(path))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='session')
def python_project(tmp_path_factory):
    path = ROOT / 'docs/examples/python-research/research.py'
    module = module_at('book_python_research', path)
    import polars as pl
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_15m_20250101_20250531.parquet', n_rows=1920)
    output = tmp_path_factory.mktemp('book_python')
    from unittest.mock import patch
    from sklearn.pipeline import Pipeline
    fitted = []
    original = Pipeline.fit
    def observe(model, *args, **kwargs):
        fitted.append(model)
        return original(model, *args, **kwargs)
    with patch.object(Pipeline, 'fit', new=observe):
        result = module.run_research(bars, str(output / 'research'))
    assert {id(model) for model in fitted} == {id(model) for model in result[3].values()}
    assert len(fitted) == 4
    return module, bars, result


@pytest.fixture(scope='session')
def compiled_project(tmp_path_factory):
    path = ROOT / 'docs/examples/tests/native_api/research.py'
    module = module_at('book_compiled_research', path)
    output = tmp_path_factory.mktemp('book_compiled') / 'experiment'
    result = module.run_research(ROOT / 'tests/fixtures', output)
    return module, output, result


@pytest.fixture
def cost_model():
    return module_at('book_cost_research', ROOT / 'docs/examples/python-research/cost_research.py')


@pytest.fixture
def python_scenarios(monkeypatch):
    directory = ROOT / 'docs/examples/python-research'
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setitem(sys.modules, 'research', module_at('book_python_driver', directory / 'research.py'))
    return module_at('book_python_scenarios', directory / 'scenarios.py')


@pytest.fixture
def compiled_scenarios(monkeypatch):
    directory = ROOT / 'docs/examples/tests/native_api'
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setitem(sys.modules, 'research', module_at('book_compiled_driver', directory / 'research.py'))
    return module_at('book_compiled_scenarios', directory / 'scenarios.py')


def pytest_collection_modifyitems(items):
    # Finish timed wheel campaigns before the model-heavy in-process checks.
    items.sort(key=lambda item: 'research_proof' not in item.fixturenames)


@pytest.fixture(scope='session', autouse=True)
def wheel_execution(tmp_path_factory, request):
    if not any('research_proof' in item.fixturenames for item in request.session.items):
        yield None
        return
    temporary = tmp_path_factory.mktemp('book_wheel')
    receipt = temporary / 'proof.json'
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / 'docs/examples')]))
    argv = [sys.executable, '-u', str(ROOT / 'docs/examples/tests/run_wheel_proof.py'),
            '--root', str(ROOT), '--temporary', str(temporary), '--output', str(receipt)]
    for option in ('--released-wheel', '--released-bundle'):
        value = request.config.getoption(option)
        if value is not None:
            argv.extend([option, str(value)])
    with subprocess.Popen(argv, cwd=ROOT, env=environment) as process:
        try:
            yield process, receipt
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


@pytest.fixture(scope='session')
def research_proof(python_project, compiled_project, wheel_execution, request):
    proof = module_at('book_semantic_oracles', ROOT / 'docs/examples/tests/test_research_examples.py')
    proof.assert_python_research(python_project)
    proof.assert_compiled_research(compiled_project)
    assert wheel_execution is not None
    process, receipt = wheel_execution
    assert process.wait() == 0, 'The isolated installed-wheel proof failed'
    wheel = json.loads(receipt.read_text())
    wheel['scenario_reports'] = {(interface, scenario): report
                                for interface, scenario, report in wheel['scenario_reports']}
    request.config.stash[PROOF] = wheel
    return {'python': {'rounds': python_project[2][1].experiment_log.height},
            'compiled': {'rounds': wheel['summaries']['compiled']['summary']['rounds']},
            'acquisition': wheel['acquisition'],
            'scenarios': wheel['scenario_reports']}


PROOF = pytest.StashKey[dict]()


@pytest.fixture(scope='session')
def installed_wheel(research_proof, request):
    return request.config.stash[PROOF]


REPORTS = []


def pytest_addoption(parser):
    parser.addoption('--docs-evidence', default=str(ROOT / '.docs-evidence/current.json'))
    parser.addoption('--released-wheel', type=Path, help='Receipt from released_wheel.py; execute that published wheel instead of building a candidate.')
    parser.addoption('--released-bundle', type=Path, help='Anonymous Release download; its exact bytes must equal the current pinned project/input archive.')


def pytest_sessionstart(session):
    import json
    target = Path(session.config.getoption('--docs-evidence'))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'status': 'running'}) + '\n')


def pytest_runtest_logreport(report):
    if report.when == 'call' or report.failed or report.skipped:
        REPORTS.append((report.nodeid, report.outcome))


def pytest_sessionfinish(session, exitstatus):
    import json
    target = Path(session.config.getoption('--docs-evidence'))
    if exitstatus != 0 or PROOF not in session.config.stash or not REPORTS or any(outcome != 'passed' for _, outcome in REPORTS):
        target.write_text(json.dumps({'status': 'failed' if exitstatus else 'incomplete', 'reason': 'Complete research validation did not finish.'}) + '\n')
        return
    import os
    import platform
    import subprocess
    from check_evidence import all_bindings, all_reference_bindings, digest, validate_evidence
    from build_bundle import INPUTS
    catalogue = json.loads((ROOT / 'docs/examples/catalogue.json').read_text())
    details = catalogue['scenario_details']
    tests = {name for name, _ in REPORTS}
    required = {test for identity, entry in details.items()
                for test in [entry['test'] + '[' + identity.replace('/', '-') + ']', *entry.get('extra_tests', [])]}
    missing = sorted(required - tests)
    if missing:
        target.write_text(json.dumps({'status': 'incomplete', 'missing_tests': missing}) + '\n')
        directory = ROOT / 'docs/examples/tests'
        if any(directory.is_relative_to(Path(str(argument).split('::')[0]).resolve()) for argument in session.config.args):
            raise ValueError('Required scenario tests did not execute: ' + ', '.join(missing))
        print('Selected tests passed; complete research evidence remains unpublished because required tests were not selected.', file=sys.stderr)
        return
    wheel = session.config.stash[PROOF]
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    tested, tree = git('rev-parse', 'HEAD'), git('rev-parse', 'HEAD^{tree}')
    head = os.environ.get('LIMEN_EXPECTED_HEAD', tested)
    from provenance import validate_provenance
    provenance = json.loads((ROOT / '.docs-evidence/source-provenance.json').read_text())
    validate_provenance(ROOT, provenance)
    files = {item['path']: item for item in provenance['files']}
    sources = [item | {'region': '__file__'} for item in provenance['files']]
    excerpts = {(item['path'], item['region']): item for item in [*all_bindings(ROOT), *all_reference_bindings(ROOT)]}
    sources += [{key: item[key] for key in ('path', 'region', 'sha256')} | {key: files[item['path']][key] for key in ('commit', 'status', 'merged_pr')} for item in excerpts.values() if item['region'] != '__file__']
    record = {
        'schema_version': 1, 'mode': 'candidate',
        'revision': {'head': head, 'tested_commit': tested, 'tree': tree, 'dirty': bool(git('status', '--porcelain', '--untracked-files=normal'))},
        'package': {'version': wheel['installed']['version'], 'wheel_sha256': wheel['wheel_sha256'], 'import_location': wheel['installed']['import_location']},
        'bundle': wheel['bundle'],
        'native_outputs': wheel['native_outputs'],
        'environment': {'python': wheel['installed']['python'], 'platform': platform.platform(), 'runner': os.environ.get('ImageOS', 'local') + ':' + os.environ.get('ImageVersion', platform.machine()), 'lock_sha256': wheel['offline']['runtime_sha256'], 'lock_resolution_target': wheel['offline']['runtime_lock'] + ' / Python 3.10 resolution', 'build_lock_sha256': wheel['offline']['build_sha256'], 'wheelhouse_manifest_sha256': digest((ROOT / '.docs-evidence/environment-wheels.json').read_bytes()), 'dependencies_sha256': digest(json.dumps(wheel['installed']['dependencies']).encode())},
        'inputs': [{'path': 'tests/fixtures/' + name, 'sha256': value, 'window': 'pinned input or attribution/reproduction file; scenario windows are in the catalogue'} for name, value in INPUTS.items()],
        'sources': sources, 'displays': [],
        'validation': {'catalogue_sha256': digest((ROOT / 'docs/examples/catalogue.json').read_bytes()), 'status': 'passed', 'scenarios': [
            {'id': scenario, 'interface': interface, 'command': item['command'], 'tests': [details[interface + '/' + scenario]['test'] + '[' + interface + '-' + scenario + ']', *details[interface + '/' + scenario].get('extra_tests', [])], 'assertions': [details[interface + '/' + scenario]['assertions'], 'actual validated wheel imported outside checkout', 'unchanged archived project CLI', 'complete wheel Python source equals the tested tree'], 'output_path': item['output_path'], 'output_sha256': item['output_sha256'], 'status': 'passed'} for (interface, scenario), item in wheel['scenario_reports'].items()]},
        'publication': None,
    }
    record['displays'] = json.loads((ROOT / 'docs/examples/retained/selection.json').read_text())['displays']
    version = (ROOT / 'pyproject.toml').read_text().split('version = "', 1)[1].split('"', 1)[0]
    validate_evidence(record, ROOT, head=head, tree=tree, version=version)
    target.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
