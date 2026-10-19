import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'docs/examples'))


@pytest.fixture(scope='session')
def python_project(tmp_path_factory):
    path = ROOT / 'docs/examples/python-research/research.py'
    spec = importlib.util.spec_from_file_location('book_python_research', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    path = ROOT / 'docs/examples/compiled-research/research.py'
    spec = importlib.util.spec_from_file_location('book_compiled_research', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path_factory.mktemp('book_compiled') / 'experiment'
    result = module.run_research(ROOT / 'tests/fixtures', output)
    return module, output, result


@pytest.fixture(scope='session')
def research_proof(python_project, compiled_project, tmp_path_factory, request):
    import importlib
    spec = importlib.util.spec_from_file_location('book_semantic_oracles', ROOT / 'docs/examples/tests/test_research_examples.py')
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    proof.assert_python_research(python_project)
    proof.assert_compiled_research(compiled_project)
    spec = importlib.util.spec_from_file_location('book_wheel_proof', ROOT / 'docs/examples/tests/wheel_proof.py')
    wheel_proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wheel_proof)
    # A caller's checkout must not replace the installed wheel in project runs.
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv('PYTHONPATH', str(ROOT))
        wheel = wheel_proof.prove_wheel(ROOT, tmp_path_factory.mktemp('book_wheel'))
    request.config.stash[PROOF] = wheel
    return {'python': {'rounds': python_project[2][1].experiment_log.height},
            'compiled': {'rounds': compiled_project[2][0].experiment_log.height}}


PROOF = pytest.StashKey[dict]()
REPORTS = []


def pytest_addoption(parser):
    parser.addoption('--docs-evidence', default=str(ROOT / '.docs-evidence/current.json'))


def pytest_sessionstart(session):
    import json
    target = Path(session.config.getoption('--docs-evidence'))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'status': 'running'}) + '\n')


def pytest_runtest_logreport(report):
    if report.when == 'call' or report.failed or report.skipped:
        REPORTS.append((report.nodeid, report.outcome))


def pytest_sessionfinish(session, exitstatus):
    if exitstatus != 0 or PROOF not in session.config.stash or not REPORTS or any(outcome != 'passed' for _, outcome in REPORTS):
        return
    import hashlib
    import json
    import os
    import platform
    import subprocess
    from check_evidence import bindings, digest, validate_evidence
    from build_bundle import INPUTS
    wheel = session.config.stash[PROOF]
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    tested, tree = git('rev-parse', 'HEAD'), git('rev-parse', 'HEAD^{tree}')
    head = os.environ.get('LIMEN_EXPECTED_HEAD', tested)
    sources = [{key: item[key] for key in ('path', 'region', 'sha256')} | {'commit': git('log', '-1', '--format=%H', '--', item['path']), 'status': 'candidate', 'merged_pr': None} for item in bindings(ROOT, ROOT / 'docs/Research-Book.md')]
    record = {
        'schema_version': 1, 'mode': 'candidate',
        'revision': {'head': head, 'tested_commit': tested, 'tree': tree, 'dirty': bool(git('status', '--porcelain', '--untracked-files=normal'))},
        'package': {'version': wheel['installed']['version'], 'wheel_sha256': wheel['wheel_sha256'], 'import_location': wheel['installed']['import_location']},
        'bundle': wheel['bundle'],
        'environment': {'python': wheel['installed']['python'], 'platform': platform.platform(), 'runner': os.environ.get('ImageOS', 'local') + ':' + os.environ.get('ImageVersion', platform.machine()), 'lock_sha256': digest((ROOT / 'requirements/ci/research-env.txt').read_bytes()), 'lock_resolution_target': 'Python 3.10 / Linux (research-env.in resolution header)', 'dependencies_sha256': digest(json.dumps(wheel['installed']['dependencies']).encode())},
        'inputs': [{'path': 'tests/fixtures/' + name, 'sha256': value, 'window': 'full recorded fixture; scenario-specific row selections are in the project'} for name, value in INPUTS.items()],
        'sources': sources, 'displays': [],
        'validation': {'catalogue_sha256': digest((ROOT / 'docs/examples/catalogue.json').read_bytes()), 'status': 'passed', 'scenarios': [
            {'id': 'selection', 'interface': interface, 'command': wheel['commands'][interface], 'tests': [name for name, _ in REPORTS], 'assertions': ['independent selection/cost/clock and fitted-state oracles', 'actual candidate wheel imported outside checkout', 'unchanged archived project CLI', 'complete wheel Python source equals the tested tree'], 'output_path': wheel['summaries'][interface]['output_path'], 'output_sha256': wheel['summaries'][interface]['output_sha256'], 'status': 'passed'} for interface in ('python', 'compiled')]},
        'publication': None,
    }
    # Display records are retained separately; before capture, the actual output
    # is still reachable as current evidence, without a fabricated historical run.
    env_sha = digest(json.dumps(record['environment'], sort_keys=True).encode())
    record['displays'] = [{'id': interface + '-selection', 'scenario': 'selection', 'path': wheel['summaries'][interface]['output_path'], 'sha256': wheel['summaries'][interface]['output_sha256'], 'source_sha256': next(item['sha256'] for item in sources if f'/{interface}-research/' in item['path']), 'environment_sha256': env_sha, 'retained_commit': tested, 'metrics': {}} for interface in ('python', 'compiled')]
    version = (ROOT / 'pyproject.toml').read_text().split('version = "', 1)[1].split('"', 1)[0]
    validate_evidence(record, ROOT, head=head, tree=tree, version=version)
    target = Path(session.config.getoption('--docs-evidence'))
    target.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
