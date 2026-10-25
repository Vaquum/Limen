from __future__ import annotations

from copy import deepcopy
import hashlib
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
PRODUCER_ATTEMPT = '${{ needs.pr_checks_tests.outputs.artifact_attempt }}'
REQUIRE_IDENTITY = 'set -euo pipefail\ntest -n "$ARTIFACT_ATTEMPT"\ntest -n "$EXPECTED_SHA256"\n'
CONSUMERS = (('pr_checks_runtime', 'runtime', 'test-runtime-profile', 'test_runtime_profile.json'),
             ('pr_checks_coverage', 'coverage', 'coverage-data', 'coverage.db'),
             ('publish_coverage_pr_comment', 'coverage', 'coverage-data', 'coverage.db'))
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
    _check_artifact_contract(workflow)


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


def _verify_command(filename):
    return f'''printf '%s  %s\\n' "$EXPECTED_SHA256" coverage-artifacts/{filename} | sha256sum -c -'''


def _check_artifact_contract(workflow):
    jobs = workflow['jobs']
    producer = jobs['pr_checks_tests']
    assert producer['outputs'] == {
        'artifact_attempt': '${{ steps.artifact_identity.outputs.attempt }}',
        'runtime_sha256': '${{ steps.artifact_identity.outputs.runtime_sha256 }}',
        'coverage_sha256': '${{ steps.artifact_identity.outputs.coverage_sha256 }}',
    }
    identity = _step(producer['steps'], 'Record validation artifact identity')
    assert identity['id'] == 'artifact_identity' and identity['if'] == '${{ always() }}'
    assert 'echo "attempt=$GITHUB_RUN_ATTEMPT" >> "$GITHUB_OUTPUT"' in identity['run']
    assert 'runtime:test_runtime_profile.json coverage:coverage.db' in identity['run']
    assert 'file="coverage-artifacts/${artifact#*:}"' in identity['run']
    assert 'echo "${artifact%%:*}_sha256=$(sha256sum "$file" | cut -d\' \' -f1)" >> "$GITHUB_OUTPUT"' in identity['run']
    assert producer['steps'].index(identity) > producer['steps'].index(_step(producer['steps'], 'Run limen tests with coverage'))
    for kind, prefix in (('runtime', 'test-runtime-profile'), ('coverage', 'coverage-data')):
        upload = _step(producer['steps'], f'Upload {kind} artifact')
        assert upload['with']['name'] == prefix + '-attempt-${{ github.run_attempt }}'
        assert 'overwrite' not in upload['with']
        assert producer['steps'].index(identity) < producer['steps'].index(upload)
    assert jobs['pr_checks_runtime']['if'] == '${{ always() }}'
    assert jobs['pr_checks_coverage'].get('if') is None
    for job, kind, prefix, filename in CONSUMERS:
        assert jobs[job]['needs'] == 'pr_checks_tests'
        steps = jobs[job]['steps']
        download = _step(steps, f'Download {kind} artifact')
        assert download['with'] == {
            'name': prefix + '-attempt-' + PRODUCER_ATTEMPT,
            'path': 'coverage-artifacts',
        }
        expected_hash = '${{ needs.pr_checks_tests.outputs.' + kind + '_sha256 }}'
        if job == 'publish_coverage_pr_comment':
            assert download.get('if') == "${{ needs.pr_checks_tests.outputs.artifact_attempt != '' && needs.pr_checks_tests.outputs.coverage_sha256 != '' }}"
            verify = _step(steps, 'Check coverage artifact presence')
            assert verify['env'] == {'EXPECTED_SHA256': expected_hash}
            assert 'test -n "$EXPECTED_SHA256"' in verify['run']
            assert _verify_command(filename) in verify['run']
            assert verify['run'].index(_verify_command(filename)) < verify['run'].index('echo "available=true"')
            assert steps.index(download) < steps.index(verify) < steps.index(_step(steps, 'Restore coverage data')) < steps.index(_step(steps, 'Publish coverage comment'))
            assert _step(steps, 'Publish coverage comment')['if'] == "steps.coverage_artifact.outputs.available == 'true'"
        else:
            require = _step(steps, f'Require {kind} artifact identity')
            verify = _step(steps, f'Verify {kind} artifact identity')
            condition = '${{ needs.pr_checks_tests.result == \'success\' }}' if kind == 'runtime' else None
            assert all(step.get('if') == condition for step in (require, download, verify))
            assert require['run'] == REQUIRE_IDENTITY
            assert require['env'] == {'ARTIFACT_ATTEMPT': PRODUCER_ATTEMPT, 'EXPECTED_SHA256': expected_hash}
            assert verify['env'] == {'EXPECTED_SHA256': expected_hash} and verify['run'] == _verify_command(filename)
            gate = _step(steps, 'Enforce suite runtime budget' if kind == 'runtime' else 'Restore coverage data')
            assert gate.get('if') == condition
            assert steps.index(require) < steps.index(download) < steps.index(verify) < steps.index(gate)
    assert _step(jobs['pr_checks_coverage']['steps'], 'Enforce package coverage floor')['run'] == 'python -m coverage report --fail-under=92'
    assert '--enforce' in _step(jobs['pr_checks_runtime']['steps'], 'Enforce suite runtime budget')['run']


def test_validation_artifacts_follow_the_producer_attempt():
    _check_artifact_contract(_workflow(TESTS_WORKFLOW))


@pytest.mark.parametrize('consumer', CONSUMERS)
@pytest.mark.parametrize('fault', ['fixed_name', 'consumer_attempt', 'empty_name', 'artifact_ids', 'pattern', 'missing_identity', 'missing_hash', 'late_verify', 'skipped_gate'])
def test_artifact_contract_rejects_consumer_mutations(consumer, fault):
    workflow = _workflow(TESTS_WORKFLOW)
    _check_artifact_contract(workflow)
    job, kind, prefix, _ = consumer
    steps = workflow['jobs'][job]['steps']
    download = _step(steps, f'Download {kind} artifact')
    comment = job == 'publish_coverage_pr_comment'
    verify = _step(steps, 'Check coverage artifact presence' if comment else f'Verify {kind} artifact identity')
    match fault:
        case 'fixed_name':
            download['with']['name'] = prefix
        case 'consumer_attempt':
            download['with']['name'] = prefix + '-attempt-${{ github.run_attempt }}'
        case 'empty_name':
            download['with']['name'] = ''
        case 'artifact_ids':
            download['with'] = {'artifact-ids': '${{ needs.pr_checks_tests.outputs.artifact_id }}', 'path': 'coverage-artifacts'}
        case 'pattern':
            download['with'] = {'pattern': prefix + '*', 'merge-multiple': True, 'path': 'coverage-artifacts'}
        case 'missing_identity':
            if comment:
                download.pop('if')
            else:
                steps.remove(_step(steps, f'Require {kind} artifact identity'))
        case 'missing_hash':
            verify['run'] = 'echo unverified'
        case 'late_verify':
            steps.remove(verify)
            steps.append(verify)
        case 'skipped_gate':
            gate = _step(steps, 'Publish coverage comment' if comment else
                'Enforce suite runtime budget' if kind == 'runtime' else 'Restore coverage data')
            gate['if'] = "${{ needs.pr_checks_tests.outputs.artifact_attempt != '' }}"
    with pytest.raises(AssertionError):
        _check_artifact_contract(workflow)


@pytest.mark.parametrize('fault', ['fixed_runtime', 'fixed_coverage', 'overwrite', 'missing_output', 'consumer_output', 'conditional_record'])
def test_artifact_contract_rejects_producer_mutations(fault):
    workflow = _workflow(TESTS_WORKFLOW)
    _check_artifact_contract(workflow)
    producer = workflow['jobs']['pr_checks_tests']
    match fault:
        case 'fixed_runtime' | 'fixed_coverage':
            kind = fault.removeprefix('fixed_')
            _step(producer['steps'], f'Upload {kind} artifact')['with']['name'] = kind
        case 'overwrite':
            _step(producer['steps'], 'Upload runtime artifact')['with']['overwrite'] = True
        case 'missing_output':
            del producer['outputs']['artifact_attempt']
        case 'consumer_output':
            producer['outputs']['artifact_attempt'] = '${{ github.run_attempt }}'
        case 'conditional_record':
            _step(producer['steps'], 'Record validation artifact identity')['if'] = 'success()'
    with pytest.raises(AssertionError):
        _check_artifact_contract(workflow)


@pytest.mark.parametrize('kind', ['runtime', 'coverage'])
@pytest.mark.parametrize('missing', ['ARTIFACT_ATTEMPT', 'EXPECTED_SHA256'])
def test_required_consumer_refuses_missing_producer_identity(tmp_path, kind, missing):
    jobs = _workflow(TESTS_WORKFLOW)['jobs']
    require = _step(jobs['pr_checks_' + kind]['steps'], f'Require {kind} artifact identity')
    env = {**os.environ, 'ARTIFACT_ATTEMPT': '1', 'EXPECTED_SHA256': hashlib.sha256(TESTS_WORKFLOW.read_bytes()).hexdigest()}
    env[missing] = ''
    result = subprocess.run(['bash', '-c', require['run']], cwd=tmp_path, env=env, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode != 0
    assert not (tmp_path / 'coverage-artifacts').exists()


def test_missing_payload_does_not_publish_old_coverage(tmp_path):
    jobs = _workflow(TESTS_WORKFLOW)['jobs']
    output = tmp_path / 'outputs'
    env = {**os.environ, 'GITHUB_RUN_ATTEMPT': '2', 'GITHUB_OUTPUT': str(output), 'EXPECTED_SHA256': ''}
    record = _step(jobs['pr_checks_tests']['steps'], 'Record validation artifact identity')
    result = subprocess.run(['bash', '-c', record['run']], cwd=tmp_path, env=env, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == 'attempt=2\n'
    output.unlink()
    presence = _step(jobs['publish_coverage_pr_comment']['steps'], 'Check coverage artifact presence')
    result = subprocess.run(['bash', '-c', presence['run']], cwd=tmp_path, env=env, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == 'available=false\n'


@pytest.mark.parametrize('consumer', CONSUMERS)
@pytest.mark.parametrize('wrong_payload', [False, True])
def test_transport_verifies_the_exact_repository_bytes(tmp_path, consumer, wrong_payload):
    jobs = _workflow(TESTS_WORKFLOW)['jobs']
    job, kind, _, filename = consumer
    destination = tmp_path / 'coverage-artifacts' / filename
    destination.parent.mkdir()
    # Transport treats payloads as opaque bytes; use real repository files.
    shutil.copyfile(TESTS_WORKFLOW, destination)
    expected = hashlib.sha256(destination.read_bytes()).hexdigest()
    if wrong_payload:
        shutil.copyfile(RULESET_WORKFLOW, destination)
    output = tmp_path / 'outputs'
    env = {**os.environ, 'EXPECTED_SHA256': expected, 'GITHUB_OUTPUT': str(output)}
    comment = job == 'publish_coverage_pr_comment'
    verify = _step(jobs[job]['steps'], 'Check coverage artifact presence' if comment else f'Verify {kind} artifact identity')
    result = subprocess.run(['bash', '-c', verify['run']], cwd=tmp_path, env=env, capture_output=True, text=True, check=False, timeout=10)
    assert (result.returncode == 0) is not wrong_payload, result.stdout + result.stderr
    if comment:
        assert (output.exists() and 'available=true' in output.read_text()) is not wrong_payload


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
