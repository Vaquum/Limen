from __future__ import annotations

import importlib.util
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


def test_governance_hardening_surfaces() -> None:
    codeowners = (ROOT / '.github' / 'CODEOWNERS').read_text(encoding='utf-8').splitlines()
    for path in ('/governance/', '/.github/', '/tests/test_packaging_surface.py'):
        assert f'{path} @mikkokotila @zero-bang' in codeowners, path
    # The global `* @zero-bang` rule must not survive: under
    # require_code_owner_review it would make one owner required on every PR.
    assert not any(line.strip().startswith('* ') for line in codeowners)

    sweep = (ROOT / '.github' / 'workflows' / 'pr_checks_slice_sweep.yml').read_text(encoding='utf-8')
    assert re.search(r'^\s*schedule:', sweep, re.MULTILINE) is not None
    assert 'name=pr_checks_slice' in sweep
    assert 'governance/slice_gate.py' in sweep
    # Truncation-refusal marker: the sweep must fail closed rather than
    # scope-check a truncated changed-file set.
    assert 'refuses to run a scope check on a truncated set' in sweep

    common = (ROOT / 'governance' / '_common.py').read_text(encoding='utf-8')
    assert "__all__ = ['CLOSING_KEYWORD_RE']" in common
    assert 'cc_gate' not in common
    # The trimmed module defines exactly one public symbol.
    spec = importlib.util.spec_from_file_location('_limen_common', ROOT / 'governance' / '_common.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__all__ == ['CLOSING_KEYWORD_RE']


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
    envelope = [
        *project['dependencies'],
        *project['optional-dependencies']['all'],
        *project['optional-dependencies']['release'],
    ]
    constraints_path = ROOT / 'requirements' / 'constraints.txt'
    constraints = [
        line for line in constraints_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    assert sorted(constraints) == sorted(envelope)
