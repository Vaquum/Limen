from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_tests.yml'
RULESET_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_ruleset.yml'
EXPECTED_TEST_COMMAND = 'python -m coverage run -m tests.run'
CONTRACT_COMMAND = 'python -m pytest -q governance/tests/test_tests_ci_contract.py'
COVERAGE_ENV = {'LIMEN_COVERAGE_RUN': '1', 'COVERAGE_CORE': 'sysmon'}
EXPECTED_COVERED_RUN = (
    'set -euxo pipefail\n'
    'mkdir -p coverage-artifacts\n'
    "python - <<'PY'\n"
    'import coverage\n'
    'import sys\n'
    'measurement = coverage.Coverage()\n'
    'measurement.start()\n'
    "active = sys.monitoring.get_tool(sys.monitoring.COVERAGE_ID) == 'coverage.py'\n"
    'measurement.stop()\n'
    'if not active:\n'
    "    raise SystemExit('SysMonitor coverage is not active')\n"
    'PY\n'
    'test_exit=0\n'
    'python -m coverage run -m tests.run || test_exit=$?\n'
    'if [ -f .coverage ]; then\n'
    '  python -m coverage report\n'
    '  mv .coverage coverage-artifacts/coverage.db\n'
    'fi\n'
    'exit $test_exit\n'
)
EXPECTED_RULESET_COMMAND = (
    'python -m pytest '
    'governance/tests/test_cc_gate.py '
    'governance/tests/test_ruleset_gate.py '
    'governance/tests/test_privileged_ruleset_audit.py '
    'governance/tests/test_ci_contract.py '
    'governance/tests/test_lint_ci_contract.py '
    'governance/tests/test_tests_ci_contract.py '
    'governance/tests/test_governance_config.py '
    'governance/tests/test_prd_gate.py '
    'governance/tests/test_supply_chain.py '
    '-q'
)


def _workflow(path: Path):
    return YAML(typ='safe', pure=True).load(path.read_text(encoding='utf-8'))


def _step(steps, name):
    matches = [step for step in steps if step.get('name') == name]
    assert len(matches) == 1, f'missing or duplicated step: {name}'
    return matches[0]


def _check_workflows(workflow, ruleset) -> None:
    for config, job_name in ((workflow, 'pr_checks_tests'), (ruleset, 'pr_checks_ruleset')):
        assert not config.get('env', {}).get('PYTEST_ADDOPTS'), 'workflow pytest selection changed'
        assert not config['jobs'][job_name].get('env', {}).get('PYTEST_ADDOPTS'), 'job pytest selection changed'
    steps = workflow['jobs']['pr_checks_tests']['steps']
    setup = [step for step in steps if step.get('uses', '').startswith('actions/setup-python@')]
    assert [step['with']['python-version'] for step in setup] == ['3.12'], 'covered interpreter changed'
    covered = _step(steps, 'Run limen tests with coverage')
    contract = _step(steps, 'Verify coverage backend contract')
    assert covered == {
        'name': 'Run limen tests with coverage', 'env': COVERAGE_ENV, 'run': EXPECTED_COVERED_RUN,
    }, 'covered step changed'
    assert contract == {
        'name': 'Verify coverage backend contract', 'env': COVERAGE_ENV, 'run': CONTRACT_COMMAND,
    }, 'backend contract step changed'
    install = _step(steps, 'Install limen package (and deps)')
    assert steps.index(setup[0]) < steps.index(install) < steps.index(contract) < steps.index(covered), 'backend contract step order changed'
    collection = _step(ruleset['jobs']['pr_checks_ruleset']['steps'], 'Run CI contract tests')
    assert collection == {
        'name': 'Run CI contract tests', 'run': EXPECTED_RULESET_COMMAND,
    }, 'ruleset contract command changed'


def test_ruleset_workflow_runs_gate_regressions() -> None:
    assert TESTS_WORKFLOW.exists()
    assert RULESET_WORKFLOW.exists()
    _check_workflows(_workflow(TESTS_WORKFLOW), _workflow(RULESET_WORKFLOW))


def test_pr_checks_tests_pins_python_and_runtime_suite_command() -> None:
    workflow_text = TESTS_WORKFLOW.read_text(encoding='utf-8')
    payload = _workflow(TESTS_WORKFLOW)
    _check_workflows(payload, _workflow(RULESET_WORKFLOW))
    assert 'pip install --require-hashes -r requirements/ci/research-env.txt' in workflow_text
    assert 'pip install --no-deps -e .' in workflow_text
    # Only the informational artifact download may soft-fail.
    gating_jobs = {name: job for name, job in payload['jobs'].items() if name != 'publish_coverage_pr_comment'}
    assert gating_jobs
    for name, job in gating_jobs.items():
        assert 'continue-on-error' not in job, name
        for step in job.get('steps', []):
            assert 'continue-on-error' not in step, (name, step.get('name'))


def test_validation_artifacts_follow_the_producer_attempt():
    jobs = _workflow(TESTS_WORKFLOW)['jobs']
    producer = jobs['pr_checks_tests']
    assert producer['outputs']['artifact_attempt'] == '${{ steps.artifact_identity.outputs.attempt }}'
    for kind, prefix in (('runtime', 'test-runtime-profile'), ('coverage', 'coverage-data')):
        upload = _step(producer['steps'], f'Upload {kind} artifact')
        assert upload['with']['name'] == prefix + '-attempt-${{ github.run_attempt }}'
        assert 'overwrite' not in upload['with']
    for job, kind, prefix in (('pr_checks_runtime', 'runtime', 'test-runtime-profile'),
                              ('pr_checks_coverage', 'coverage', 'coverage-data'),
                              ('publish_coverage_pr_comment', 'coverage', 'coverage-data')):
        download = _step(jobs[job]['steps'], f'Download {kind} artifact')
        assert download['with'] == {
            'name': prefix + '-attempt-${{ needs.pr_checks_tests.outputs.artifact_attempt }}',
            'path': 'coverage-artifacts',
        }


@pytest.mark.parametrize(('mutation', 'reason'), [
    ('suite_selection', 'covered step changed'),
    ('backend_override', 'covered step changed'),
    ('missing_probe', 'covered step changed'),
    ('missing_governance_module', 'ruleset contract command changed'),
    ('missing_workflow_contract', 'ruleset contract command changed'),
    ('missing_prd_gate', 'ruleset contract command changed'),
    ('missing_supply_chain', 'ruleset contract command changed'),
    ('filtered_collection', 'ruleset contract command changed'),
    ('missing_backend_step', 'missing or duplicated step: Verify coverage backend contract'),
    ('conditional_backend_step', 'backend contract step changed'),
    ('filtered_backend_step', 'backend contract step changed'),
    ('late_backend_step', 'backend contract step order changed'),
    ('workflow_selection', 'workflow pytest selection changed'),
    ('job_selection', 'job pytest selection changed'),
    ('ruleset_workflow_selection', 'workflow pytest selection changed'),
    ('ruleset_job_selection', 'job pytest selection changed'),
])
def test_coverage_contract_rejects_workflow_mutations(mutation, reason, monkeypatch):
    workflow, ruleset = _workflow(TESTS_WORKFLOW), _workflow(RULESET_WORKFLOW)
    _check_workflows(workflow, ruleset)
    original = deepcopy((workflow, ruleset))
    steps = workflow['jobs']['pr_checks_tests']['steps']
    covered = _step(steps, 'Run limen tests with coverage')
    contract = _step(steps, 'Verify coverage backend contract')
    collection = _step(ruleset['jobs']['pr_checks_ruleset']['steps'], 'Run CI contract tests')
    match mutation:
        case 'suite_selection':
            covered['run'] = covered['run'].replace(EXPECTED_TEST_COMMAND, EXPECTED_TEST_COMMAND + ' --ignore=tests/test_factorize_equivalence.py')
        case 'backend_override':
            covered['run'] = covered['run'].replace(EXPECTED_TEST_COMMAND, 'COVERAGE_CORE=ctrace ' + EXPECTED_TEST_COMMAND)
        case 'missing_probe':
            start, end = covered['run'].index("python - <<'PY'\n"), covered['run'].index('test_exit=0\n')
            covered['run'] = covered['run'][:start] + covered['run'][end:]
        case 'missing_governance_module':
            collection['run'] = collection['run'].replace('governance/tests/test_governance_config.py ', '')
        case 'missing_workflow_contract':
            collection['run'] = collection['run'].replace('governance/tests/test_tests_ci_contract.py ', '')
        case 'missing_prd_gate':
            collection['run'] = collection['run'].replace('governance/tests/test_prd_gate.py ', '')
        case 'missing_supply_chain':
            collection['run'] = collection['run'].replace('governance/tests/test_supply_chain.py ', '')
        case 'filtered_collection':
            collection['run'] += ' -k "not governance_config"'
        case 'missing_backend_step':
            steps.remove(contract)
        case 'conditional_backend_step':
            contract['if'] = 'false'
        case 'filtered_backend_step':
            contract['run'] += ' -k "not actual_coverage_probe_rejects_fallback"'
        case 'late_backend_step':
            steps.remove(contract)
            steps.insert(steps.index(covered) + 1, contract)
        case 'workflow_selection':
            workflow['env'] = {'PYTEST_ADDOPTS': '--ignore=tests/test_factorize_equivalence.py'}
        case 'job_selection':
            workflow['jobs']['pr_checks_tests']['env'] = {'PYTEST_ADDOPTS': '-k "test_pr_checks_tests_workflow_exists or not tests_ci_contract"'}
        case 'ruleset_workflow_selection':
            ruleset['env'] = {'PYTEST_ADDOPTS': '-k "not tests_ci_contract"'}
        case 'ruleset_job_selection':
            ruleset['jobs']['pr_checks_ruleset']['env'] = {'PYTEST_ADDOPTS': '-k "not governance_config"'}
    assert (workflow, ruleset) != original, 'mutation did not change workflow'
    if mutation == 'job_selection':
        monkeypatch.setattr(sys.modules[__name__], '_workflow',
                            {TESTS_WORKFLOW: workflow, RULESET_WORKFLOW: ruleset}.__getitem__)
    with pytest.raises(AssertionError, match=reason):
        if mutation == 'job_selection':
            test_ruleset_workflow_runs_gate_regressions()
        else:
            _check_workflows(workflow, ruleset)


@pytest.mark.parametrize(('branch', 'optimize'), [(False, '0'), (True, '0'), (True, '1')])
def test_actual_coverage_probe_rejects_fallback(tmp_path, branch, optimize):
    workflow, ruleset = _workflow(TESTS_WORKFLOW), _workflow(RULESET_WORKFLOW)
    _check_workflows(workflow, ruleset)
    steps = workflow['jobs']['pr_checks_tests']['steps']
    pinned = next(step['with']['python-version'] for step in steps if step.get('uses', '').startswith('actions/setup-python@'))
    if sys.version_info[:2] < (3, 12):
        assert os.environ.get('LIMEN_COVERAGE_RUN') != '1', f'covered probe requires Python {pinned}'
        return
    assert f'{sys.version_info.major}.{sys.version_info.minor}' == pinned, 'probe interpreter differs from workflow pin'
    covered = _step(steps, 'Run limen tests with coverage')
    config = tmp_path / 'pyproject.toml'
    shutil.copyfile(REPO_ROOT / 'pyproject.toml', config)
    if branch:
        text = config.read_text(encoding='utf-8')
        assert text.count('[tool.coverage.run]') == 1
        config.write_text(text.replace('[tool.coverage.run]', '[tool.coverage.run]\nbranch = true', 1), encoding='utf-8')
    env = {'PATH': str(Path(sys.executable).parent) + os.pathsep + os.defpath,
           'HOME': str(tmp_path), 'PYTHONOPTIMIZE': optimize, **covered['env']}
    assert shutil.which('bash', path=env['PATH']), 'coverage workflow proof requires bash'
    prefix = covered['run'].split('test_exit=0\n', 1)[0]
    result = subprocess.run(['bash', '-c', prefix + "printf '%s\n' 'SUITE-START'\n"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30, check=False)
    output = result.stdout + result.stderr
    if branch:
        assert result.returncode != 0, output
        assert 'SUITE-START' not in result.stdout, output
        assert 'SysMonitor coverage is not active' in result.stderr, output
    else:
        assert result.returncode == 0, output
        assert result.stdout.strip() == 'SUITE-START', output
    assert not (tmp_path / '.coverage').exists()
