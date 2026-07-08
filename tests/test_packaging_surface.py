from __future__ import annotations

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
    assert re.search(r'PYRIGHT_WARNING_BASELINE: \d+', pyright_workflow) is not None
    assert '--outputjson' in pyright_workflow
    slice_template = (ROOT / '.github' / 'ISSUE_TEMPLATE' / 'slice.yml').read_text(encoding='utf-8')
    assert 'slice_closeout_guard' in slice_template
    assert 'reverted automatically' in slice_template


def test_pyright_gate_config() -> None:
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    pyright_config = pyproject['tool']['pyright']
    assert pyright_config['typeCheckingMode'] == 'strict'
    assert pyright_config['pythonVersion'] == '3.10'
    assert pyright_config['include'] == ['limen']
    promoted_extras = [
        'reportCallInDefaultInitializer',
        'reportImplicitStringConcatenation',
        'reportImportCycles',
        'reportMissingSuperCall',
        'reportUninitializedInstanceVariable',
        'reportUnusedCallResult',
    ]
    for rule in promoted_extras:
        assert pyright_config[rule] == 'error', rule
    promoted_strict_defaults = [
        'reportPrivateUsage',
        'reportUnnecessaryComparison',
        'reportUnnecessaryIsInstance',
        'reportUnsupportedDunderAll',
        'reportUnusedFunction',
    ]
    for rule in promoted_strict_defaults:
        assert pyright_config.get(rule, 'error') == 'error', rule
    remaining_downgrades = {k for k, v in pyright_config.items() if v == 'warning'}
    assert remaining_downgrades == {
        'reportImplicitOverride',
        'reportMissingTypeArgument',
        'reportMissingTypeStubs',
        'reportUnknownArgumentType',
        'reportUnknownLambdaType',
        'reportUnknownMemberType',
        'reportUnknownParameterType',
        'reportUnknownVariableType',
    }
    dev_extra = pyproject['project']['optional-dependencies']['dev']
    assert 'pandas-stubs>=2.3,<2.4' in dev_extra
    assert 'pyright>=1.1.408,<1.1.409' in dev_extra
    assert 'scipy-stubs>=1.15,<1.16' in dev_extra
    assert 'tomli>=2.0,<3' in dev_extra
