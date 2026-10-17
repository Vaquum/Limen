"""Parameter materialization parity against the recorded implementation."""
import hashlib
import importlib
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from ruamel.yaml import YAML

from limen.utils.param_space import ParamSpace
from tests.test_param_space import _build_huge_params

FIXTURES = Path(__file__).parent / 'fixtures'
ARCHIVE = FIXTURES / 'param_space_performance_reference.zip'
REFERENCE = json.loads((FIXTURES / 'param_space_performance_reference.json').read_text())
sys.path.insert(0, str(ARCHIVE))
try:
    BASELINE = importlib.import_module('_limen_param_space_baseline').ParamSpace
finally:
    sys.path.remove(str(ARCHIVE))
MANIFEST = YAML(typ='safe').load((Path(__file__).parents[1] / 'limen/yaml/templates/logreg_binary.yaml').read_text())


@pytest.mark.parametrize(('params', 'count'), [
    (MANIFEST['sfd']['params'], 257),
    (_build_huge_params(19), 257),
    (_build_huge_params(25), 257),
    ({}, 0),
    ({}, 1),
    ({'a': []}, 1),
    ({'a': [0, 1]}, 0),
    ({'a': [0, 1], 'b': [10, 20]}, 3),
    ({'a': [None, 'balanced'], 'b': [1, 1.5]}, 3),
    ({'a': [np.float32(.1), np.float32(.2)]}, 2),
    ({'a': [date(2025, 1, 1), date(2025, 1, 2)]}, 2),
    ({'a': [[1], [1.5]]}, 2),
    ({'a': [{'first': 1}, {'second': 2}]}, 2),
    ({'a': [None] * 100 + ['balanced']}, 101),
])
@pytest.mark.parametrize('sample', [False, True])
def test_param_space_equivalence(params, count, sample):
    for random_search in (False, True):
        try:
            expected = BASELINE(params, count, seed=42, sample=sample)
        except (pl.exceptions.PolarsError, TypeError, ValueError) as error:
            with pytest.raises(type(error)):
                ParamSpace(params, count, seed=42, sample=sample)
        else:
            result = ParamSpace(params, count, seed=42, sample=sample)
            assert_frame_equal(result.df_params, expected.df_params, check_exact=True)
            assert result.n_permutations == expected.n_permutations
            assert result._rng.getstate() == expected._rng.getstate()
            for _ in range(3):
                assert result.generate(random_search) == expected.generate(random_search)
                assert result._rng.getstate() == expected._rng.getstate()
                assert_frame_equal(result.df_params, expected.df_params, check_exact=True)


def test_reference_provenance():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == REFERENCE['archive_sha256']
    with zipfile.ZipFile(ARCHIVE) as saved:
        assert hashlib.sha256(saved.read('_limen_param_space_baseline.py')).hexdigest() == REFERENCE['source_sha256']
