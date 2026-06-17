from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


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
    code = """
import sys
import limen

assert limen.__version__ == '4.0.0'
for name in ('lightgbm', 'pyarrow', 'tabpfn', 'xgboost'):
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
    assert '4.0.0' in result.stdout


def test_optional_model_modules_import_without_optional_backends() -> None:
    code = """
from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
from limen.sfd.reference_architecture.tabpfn_binary import TabPFNBinary
from limen.sfd.reference_architecture.xgboost_regressor import XGBoostRegressor

assert LightGBMBinary.__name__ == 'LightGBMBinary'
assert TabPFNBinary.__name__ == 'TabPFNBinary'
assert XGBoostRegressor.__name__ == 'XGBoostRegressor'
"""
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)
