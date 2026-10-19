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
def research_proof(python_project, compiled_project):
    import importlib
    spec = importlib.util.spec_from_file_location('book_semantic_oracles', ROOT / 'docs/examples/tests/test_research_examples.py')
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    proof.assert_python_research(python_project)
    proof.assert_compiled_research(compiled_project)
    return {'python': {'rounds': python_project[2][1].experiment_log.height},
            'compiled': {'rounds': compiled_project[2][0].experiment_log.height}}
