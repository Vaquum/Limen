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
    result = module.run_research(bars, str(output / 'research'))
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
