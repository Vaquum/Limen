'''Property-based tests for manifest validation and frame contracts.

Hypothesis generates adversarial input families against three
already-contractual surfaces: the YAML manifest validator's fail-loud
``uel.n_permutations`` rules, the ``ParamSpace`` cardinality and
determinism laws, and the sequential splitter's partition invariants.
The registered profile is deterministic so the required CI gate cannot
flake: random seeds are replaced with a fixed derivation and
``deadline=None`` removes wall-clock variance.
'''

import copy
import math
from pathlib import Path
from typing import Any

import polars as pl
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from limen.data.utils.splits import split_sequential
from limen.utils.param_space import ParamSpace
from limen.yaml.parser import parse
from limen.yaml.validator import validate


MAX_EXAMPLES = 50
PARAM_KEY_POOL = ('alpha', 'beta', 'gamma', 'delta')

settings.register_profile(
    'ci',
    settings(
        derandomize=True,
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    ),
)
settings.load_profile('ci')

_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / 'limen' / 'yaml' / 'templates' / 'logreg_binary.yaml'
_BASE_MANIFEST, _BASE_ERRORS = parse(_TEMPLATE_PATH)
if _BASE_ERRORS:
    raise RuntimeError(f'test_property_contracts baseline template failed to parse: {_BASE_ERRORS}')
_PARAM_SPACE_TOTAL = math.prod(len(values) for values in _BASE_MANIFEST['sfd']['params'].values())


def _fresh_manifest() -> dict[str, Any]:
    '''Return a deep copy of the known-valid bundled manifest.

    Returns:
        dict: Parsed manifest dict safe to mutate per example.
    '''

    return copy.deepcopy(_BASE_MANIFEST)


_INVALID_PERMUTATION_VALUES = st.one_of(
    st.booleans(),
    st.none(),
    st.text(max_size=8),
    st.floats(allow_nan=True, allow_infinity=True),
    st.integers(max_value=0),
    st.lists(st.integers(), max_size=3),
)


@given(value=_INVALID_PERMUTATION_VALUES)
def test_manifest_validator_rejects_generated_invalid_values(value: Any) -> None:
    '''Every generated non-positive, non-int, or bool value fails validation.

    Args:
        value (Any): Generated invalid candidate for uel.n_permutations.
    '''

    manifest = _fresh_manifest()
    manifest['uel']['n_permutations'] = value

    result = validate(manifest)

    assert result.valid is False
    assert any(error.path == 'uel.n_permutations' for error in result.errors)


@given(data=st.data())
def test_manifest_validator_accepts_in_budget_and_rejects_over_budget(data: st.DataObject) -> None:
    '''In-budget permutation counts validate; over-budget counts fail loud.

    Args:
        data (st.DataObject): Hypothesis draw handle for dependent values.
    '''

    manifest = _fresh_manifest()
    in_budget = data.draw(st.integers(min_value=1, max_value=_PARAM_SPACE_TOTAL))
    manifest['uel']['n_permutations'] = in_budget

    assert validate(manifest).valid is True

    over_budget = data.draw(
        st.integers(min_value=_PARAM_SPACE_TOTAL + 1, max_value=_PARAM_SPACE_TOTAL * 2),
    )
    manifest['uel']['n_permutations'] = over_budget

    result = validate(manifest)

    assert result.valid is False
    assert any(error.path == 'uel.n_permutations' for error in result.errors)


_PARAM_GRIDS = st.dictionaries(
    keys=st.sampled_from(PARAM_KEY_POOL),
    values=st.lists(st.integers(min_value=-9, max_value=9), min_size=1, max_size=4, unique=True),
    min_size=1,
    max_size=len(PARAM_KEY_POOL),
)


@given(params=_PARAM_GRIDS, seed=st.integers(min_value=0, max_value=2**16))
def test_param_space_expansion_matches_predicted_cardinality(
    params: dict[str, list[int]],
    seed: int,
) -> None:
    '''ParamSpace enumerates the exact cartesian product without duplicates.

    Args:
        params (dict): Generated parameter grid with unique values per key.
        seed (int): Generated seed proving sampling determinism.
    '''

    total = math.prod(len(values) for values in params.values())

    full = ParamSpace(params, n_permutations=total, seed=seed)

    assert full.df_params.height == total
    assert len({tuple(row) for row in full.df_params.rows()}) == total

    sample_size = min(total, 3)
    sampled = ParamSpace(params, n_permutations=sample_size, seed=seed)
    twin = ParamSpace(params, n_permutations=sample_size, seed=seed)

    assert sampled.df_params.height == sample_size
    assert len({tuple(row) for row in sampled.df_params.rows()}) == sample_size
    assert sampled.df_params.equals(twin.df_params)


@given(
    n_rows=st.integers(min_value=0, max_value=200),
    ratios=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=5),
)
def test_sequential_split_partitions_preserve_rows(n_rows: int, ratios: list[int]) -> None:
    '''Sequential splits partition every generated frame without loss or reorder.

    Args:
        n_rows (int): Generated frame length.
        ratios (list): Generated positive split proportions.
    '''

    data = pl.DataFrame({'x': list(range(n_rows))})

    parts = split_sequential(data, ratios)

    assert len(parts) == len(ratios)
    assert sum(part.height for part in parts) == n_rows

    if n_rows > 0:
        recombined = pl.concat([part for part in parts if part.height > 0])
        assert recombined['x'].to_list() == list(range(n_rows))
