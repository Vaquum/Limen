from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_BACKEND_MODULES = ('lightgbm', 'pyarrow', 'scipy', 'statsmodels', 'talib', 'tabpfn', 'xgboost')
MODEL_BACKEND_MODULES = ('lightgbm', 'tabpfn', 'xgboost')
BARE_CI_TOOLS = ('build', 'cyclonedx-bom', 'pip-audit')


def _project_version() -> str:
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    return str(project['version'])


def test_package_audit_source_contract() -> None:
    result = subprocess.run(
        [sys.executable, 'scripts/package_audit.py'],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'package audit passed' in result.stdout


def test_root_import_is_light_and_versioned() -> None:
    code = f"""
import sys
import limen

assert limen.__version__ == {_project_version()!r}
for name in {OPTIONAL_BACKEND_MODULES!r}:
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


def test_module_entrypoint_version() -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'limen', '--version'],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert _project_version() in result.stdout


def test_optional_model_modules_import_without_optional_backends() -> None:
    code = f"""
import sys

from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
from limen.sfd.reference_architecture.tabpfn_binary import TabPFNBinary
from limen.sfd.reference_architecture.xgboost_regressor import XGBoostRegressor

assert LightGBMBinary.__name__ == 'LightGBMBinary'
assert TabPFNBinary.__name__ == 'TabPFNBinary'
assert XGBoostRegressor.__name__ == 'XGBoostRegressor'
for name in {MODEL_BACKEND_MODULES!r}:
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


def test_slice_closeout_guard_and_ratchet_surfaces() -> None:
    guard = (ROOT / '.github' / 'workflows' / 'slice_closeout_guard.yml').read_text(encoding='utf-8')
    assert 'issues:' in guard
    assert 'gh issue reopen' in guard
    pyright_workflow = (ROOT / '.github' / 'workflows' / 'pr_checks_pyright.yml').read_text(encoding='utf-8')
    assert re.search(r'PYRIGHT_WARNING_BASELINE: 0\b', pyright_workflow) is not None
    assert '--outputjson' in pyright_workflow
    slice_template = (ROOT / '.github' / 'ISSUE_TEMPLATE' / 'slice.yml').read_text(encoding='utf-8')
    assert 'slice_closeout_guard' in slice_template
    assert 'is reverted' in slice_template


def test_slice_gate_and_closeout_surfaces() -> None:
    for name in ('__init__.py', '_common.py', 'slice_gate.py'):
        assert (ROOT / 'governance' / name).is_file(), name
    gate = (ROOT / 'governance' / 'slice_gate.py').read_text(encoding='utf-8')
    assert 'rule 9' in gate
    assert 'rule 10' in gate
    assert 'OVERRULED' in gate
    assert 'sub_issues' in gate
    slice_workflow = (ROOT / '.github' / 'workflows' / 'pr_checks_slice.yml').read_text(encoding='utf-8')
    assert 'name: pr_checks_slice' in slice_workflow
    assert 'types: [opened, edited, synchronize, reopened, ready_for_review]' in slice_workflow
    assert 'governance/slice_gate.py' in slice_workflow
    on_issue_workflow = (ROOT / '.github' / 'workflows' / 'pr_checks_slice_on_issue.yml').read_text(encoding='utf-8')
    assert 'types: [edited, labeled, unlabeled, closed, reopened, deleted]' in on_issue_workflow
    assert 'name=pr_checks_slice' in on_issue_workflow
    guard_workflow = (ROOT / '.github' / 'workflows' / 'slice_closeout_guard.yml').read_text(encoding='utf-8')
    assert 'closing PR' in guard_workflow
    assert 'gh issue edit' in guard_workflow
    assert 'gh issue reopen' in guard_workflow
    slice_template = (ROOT / '.github' / 'ISSUE_TEMPLATE' / 'slice.yml').read_text(encoding='utf-8')
    assert 'OVERRULED: <reason>' in slice_template
    assert 'rule 10' in slice_template
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert 'governance' in pyproject['tool']['pyright']['exclude']
    assert 'governance/*' in pyproject['tool']['check-manifest']['ignore']


def test_pyright_gate_config() -> None:
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    pyright_config = pyproject['tool']['pyright']
    assert pyright_config['typeCheckingMode'] == 'strict'
    assert pyright_config['pythonVersion'] == '3.10'
    assert pyright_config['include'] == ['limen']
    promoted_extras = [
        'reportCallInDefaultInitializer',
        'reportImplicitOverride',
        'reportImplicitStringConcatenation',
        'reportImportCycles',
        'reportMissingSuperCall',
        'reportUninitializedInstanceVariable',
        'reportUnusedCallResult',
    ]
    for rule in promoted_extras:
        assert pyright_config[rule] == 'error', rule
    promoted_strict_defaults = [
        'reportMissingTypeArgument',
        'reportPrivateUsage',
        'reportUnknownArgumentType',
        'reportUnknownLambdaType',
        'reportUnknownMemberType',
        'reportUnknownParameterType',
        'reportUnknownVariableType',
        'reportUnnecessaryComparison',
        'reportUnnecessaryIsInstance',
        'reportUnsupportedDunderAll',
        'reportUnusedFunction',
    ]
    for rule in promoted_strict_defaults:
        assert pyright_config.get(rule, 'error') == 'error', rule
    remaining_downgrades = {k for k, v in pyright_config.items() if v == 'warning'}
    assert remaining_downgrades == set()
    silenced_rules = {k for k, v in pyright_config.items() if v == 'none'}
    assert silenced_rules == {'reportMissingTypeStubs'}
    rule_values = {v for k, v in pyright_config.items() if k.startswith('report')}
    assert rule_values <= {'error', 'none'}
    runtime_deps = pyproject['project']['dependencies']
    assert 'typing_extensions>=4.12,<5' in runtime_deps
    dev_extra = pyproject['project']['optional-dependencies']['dev']
    assert 'pandas-stubs>=2.3,<2.4' in dev_extra
    assert 'pyright>=1.1.408,<1.1.409' in dev_extra
    assert 'scipy-stubs>=1.15,<1.16' in dev_extra
    assert 'tomli>=2.0,<3' in dev_extra


def test_constraints_mirror_runtime_envelope() -> None:
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    project = pyproject['project']
    optional = project['optional-dependencies']
    ci_tool_pins = [
        pin for pin in optional['dev']
        if pin.partition('>=')[0] in BARE_CI_TOOLS
    ]
    assert sorted(pin.partition('>=')[0] for pin in ci_tool_pins) == sorted(BARE_CI_TOOLS)
    envelope = {
        # Constraints files reject extras, so the mirror strips them.
        re.sub(r'\[[^]]+]', '', entry, count=1)
        for entry in (
            *project['dependencies'],
            *optional['all'],
            *optional['release'],
            *optional['test'],
            *ci_tool_pins,
        )
    }
    constraints_path = ROOT / 'requirements' / 'constraints.txt'
    constraints = [
        line for line in constraints_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    assert sorted(constraints) == sorted(envelope)


def test_supply_chain_surfaces() -> None:
    workflows = sorted((ROOT / '.github' / 'workflows').glob('*.yml'))
    assert workflows
    for workflow in workflows:
        text = workflow.read_text(encoding='utf-8')
        assert 'permissions:' in text, workflow.name
        assert re.search(r'uses:\s*\S+@v\d', text) is None, workflow.name
        checkouts = text.count('uses: actions/checkout@')
        persisted = text.count('persist-credentials: false')
        if workflow.name == 'pr_post_release.yml':
            assert checkouts == 1
            assert persisted == 0
        else:
            assert checkouts == persisted, workflow.name

    assert (ROOT / '.github' / 'dependabot.yml').is_file()
    assert (ROOT / '.github' / 'vuln_exceptions.json').is_file()
    assert (ROOT / 'governance' / 'check_dependency_vulnerabilities.py').is_file()

    supply = (ROOT / '.github' / 'workflows' / 'pr_checks_supply.yml').read_text(encoding='utf-8')
    assert 'governance/check_dependency_vulnerabilities.py' in supply

    publish = (ROOT / '.github' / 'workflows' / 'pr_publish_pypi.yml').read_text(encoding='utf-8')
    assert 'Guard PyPI filename availability' in publish
    assert 'environment: pypi' in publish
    assert 'gh release upload' in publish
    assert 'sbom.json' in publish
    assert publish.count('provenance/provenance.intoto.jsonl') == 3
    assert 'id: attest' in publish
    assert 'attach_release_assets' not in publish
    assert 'skip-existing' not in publish

    tests_workflow = (ROOT / '.github' / 'workflows' / 'pr_checks_tests.yml').read_text(encoding='utf-8')
    assert '--require-hashes -r requirements/ci/research-env.txt' in tests_workflow
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert 'hypothesis>=6,<7' in pyproject['project']['optional-dependencies']['test']
    assert 'hypothesis' in (ROOT / 'requirements' / 'ci' / 'research-env.in').read_text(encoding='utf-8')

    unhashed_allowed = ('python -m pip install dist/*.whl',)
    for workflow in workflows:
        for raw_line in workflow.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if 'pip install' not in line:
                continue
            if line in unhashed_allowed:
                continue
            assert '--require-hashes' in line or '--no-deps' in line, (workflow.name, line)
    hashed_sets = sorted((ROOT / 'requirements' / 'ci').glob('*.txt'))
    assert [hashed.name for hashed in hashed_sets] == [
        'build-tools.txt',
        'coverage-tools.txt',
        'dev-env.txt',
        'gate-tools.txt',
        'release-tools.txt',
        'research-env.txt',
        'runtime-env.txt',
        'sbom-tools.txt',
        'supply-tools.txt',
    ]
    for hashed in hashed_sets:
        assert '--hash=sha256' in hashed.read_text(encoding='utf-8'), hashed.name
        assert hashed.with_suffix('.in').is_file(), hashed.name
        assert '-c requirements/constraints.txt' in hashed.with_suffix('.in').read_text(encoding='utf-8'), hashed.name
    policy = (ROOT / 'docs' / 'Developer' / 'Release-Policy.md').read_text(encoding='utf-8')
    assert 'require-hashes' in policy

    site_package = json.loads((ROOT / 'docs-site' / 'package.json').read_text(encoding='utf-8'))
    assert site_package['overrides']['js-yaml@^4'] == '^4.2.0'
    assert site_package['overrides']['markdown-it'] == '^14.2.0'
    site_lock = json.loads((ROOT / 'docs-site' / 'package-lock.json').read_text(encoding='utf-8'))
    js_yaml_versions = {
        tuple(int(part) for part in pkg['version'].split('.')[:3])
        for key, pkg in site_lock['packages'].items()
        if key.endswith('node_modules/js-yaml')
    }
    assert js_yaml_versions
    assert all(v >= (3, 15, 0) if v[0] == 3 else v >= (4, 2, 0) for v in js_yaml_versions)
    markdown_it_versions = {
        tuple(int(part) for part in pkg['version'].split('.')[:3])
        for key, pkg in site_lock['packages'].items()
        if key.endswith('node_modules/markdown-it')
    }
    assert markdown_it_versions
    assert all(v >= (14, 2, 0) for v in markdown_it_versions)

    release_script = (ROOT / 'scripts' / 'create_release.py').read_text(encoding='utf-8')
    assert 'TAG_RE' in release_script
    assert "os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-5')" in release_script
    assert 'OMIT_THINKING_MODEL_PREFIXES' in release_script
    assert "{'type': 'disabled'}" in release_script
    assert 'claude-opus' not in release_script


def test_governance_hardening_surfaces() -> None:
    codeowners_lines = (ROOT / '.github' / 'CODEOWNERS').read_text(encoding='utf-8').splitlines()
    assert '/governance/ @mikkokotila @pdey @bit-mis @zero-bang' in codeowners_lines
    assert '/.github/ @mikkokotila @pdey @bit-mis @zero-bang' in codeowners_lines
    assert '/tests/test_packaging_surface.py @mikkokotila @pdey @bit-mis @zero-bang' in codeowners_lines
    on_issue = (ROOT / '.github' / 'workflows' / 'pr_checks_slice_on_issue.yml').read_text(encoding='utf-8')
    assert 'actions/runs' in on_issue
    assert on_issue.count('actions: write') == 1
    readiness = (ROOT / '.github' / 'workflows' / 'pr_merge_readiness.yml').read_text(encoding='utf-8')
    assert 'pull_request_review:' in readiness
    assert 'pull_request_review_comment:' in readiness
    assert 'check_suite:' in readiness
    assert 'pull-requests: write' in readiness

    sweep_workflow = (ROOT / '.github' / 'workflows' / 'pr_checks_slice_sweep.yml').read_text(encoding='utf-8')
    assert sweep_workflow.count('schedule:') == 1
    assert 'workflow_dispatch:' in sweep_workflow
    assert 'name=pr_checks_slice' in sweep_workflow
    assert 'file enumeration incomplete' in sweep_workflow
    assert 'governance/slice_gate.py' in sweep_workflow
    assert 'actions/runs' in sweep_workflow
    assert sweep_workflow.count('actions: write') == 1
    common_path = ROOT / 'governance' / '_common.py'
    spec = importlib.util.spec_from_file_location('governance_common', common_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__all__ == ['CC_RE', 'CLOSING_KEYWORD_RE', 'REPO_ROOT', 'fail_setup']
    removed_helpers = ('TYPING_BUDGET', 'find_python_files', 'resolve_package_dir', 'significant_lines')
    for name in removed_helpers:
        assert not hasattr(module, name), name


def test_cc_gate_surfaces() -> None:
    gate_path = ROOT / 'governance' / 'cc_gate.py'
    assert gate_path.is_file()
    gate = gate_path.read_text(encoding='utf-8')
    assert 'from _common import CC_RE, CLOSING_KEYWORD_RE' in gate
    assert "'slice'" in gate
    cc_workflow = (ROOT / '.github' / 'workflows' / 'pr_checks_cc.yml').read_text(encoding='utf-8')
    assert 'name: pr_checks_cc' in cc_workflow
    assert 'types: [opened, edited, synchronize, reopened, ready_for_review]' in cc_workflow
    assert 'python governance/cc_gate.py' in cc_workflow
    assert 'fetch-depth: 0' in cc_workflow
    spec = importlib.util.spec_from_file_location('governance_common_cc', ROOT / 'governance' / '_common.py')
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__all__ == ['CC_RE', 'CLOSING_KEYWORD_RE', 'REPO_ROOT', 'fail_setup']
    slice_template = (ROOT / '.github' / 'ISSUE_TEMPLATE' / 'slice.yml').read_text(encoding='utf-8')
    assert 'Conventional Commits' in slice_template
    result = subprocess.run(
        [
            sys.executable, 'governance/cc_gate.py',
            '--pr-title', 'feat: x',
            '--pr-body-file', '/dev/null',
            '--base-ref', 'HEAD',
            '--head-ref', 'HEAD',
            '--repo', 'Vaquum/Limen',
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CC GATE -- PASS' in result.stdout
    attribution_code = '''
import sys
sys.path.insert(0, 'governance')
from cc_gate import attribution_hit

for clean in (
    'feat: add Gemini exchange connector',
    'fix: reuse the database cursor between batches',
    'docs: describe artifacts generated with the manifest runner',
    'ci: adjust copilot_code_review rule parameters',
):
    assert attribution_hit(clean) is None, clean

for attributed in (
    'Generated with Claude Code',
    'Co-authored-by: Gemini <bot@google.com>',
    'feat: add Google Gemini client',
    'chore: llm-generated cleanup',
):
    assert attribution_hit(attributed) is not None, attributed
'''
    subprocess.run([sys.executable, '-c', attribution_code], cwd=ROOT, check=True)
