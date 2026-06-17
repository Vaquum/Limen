from __future__ import annotations

from pathlib import Path
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
