import random

import pytest

from limen.utils.param_space import ParamSpace
from limen.utils.param_space import sample_range_exact


HUGE_SPACE_FIRST_INDICES = [
    2053695854357871005,
    4517457392071889495,
    2574020394472462046,
    1890702223848595625,
    586287033698423193,
]


def _build_huge_params(dimensions: int = 19) -> dict[str, list[int]]:
    return {f'p{i}': list(range(10)) for i in range(dimensions)}


def _collect_generated_params(
    seed: int,
    params: dict[str, list[int]],
    n_permutations: int,
    count: int,
    *,
    random_search: bool,
) -> list[dict]:
    previous_state = random.getstate()
    try:
        random.seed(seed)
        param_space = ParamSpace(params, n_permutations)
        return [
            param_space.generate(random_search=random_search)
            for _ in range(count)
        ]
    finally:
        random.setstate(previous_state)


def test_sample_range_exact_matches_stdlib_and_preserves_rng_state():
    cases = [
        (0, 0),
        (1, 0),
        (1, 1),
        (10, 10),
        (30, 5),
        (100, 10),
        (1_000, 50),
        (1_000_000, 500),
    ]

    for seed in range(10):
        for population_size, sample_size in cases:
            exact_rng = random.Random(seed)
            stdlib_rng = random.Random(seed)

            exact_sample = sample_range_exact(exact_rng, population_size, sample_size)
            stdlib_sample = stdlib_rng.sample(range(population_size), sample_size)

            assert exact_sample == stdlib_sample
            assert exact_rng.randrange(1_000_000_000) == stdlib_rng.randrange(1_000_000_000)


def test_sample_range_exact_retries_rejected_and_duplicate_draws_in_order():
    class StubRandom:
        def __init__(self) -> None:
            self.values = iter([130, 7, 7, 9])

        def getrandbits(self, _bits: int) -> int:
            return next(self.values)

        def sample(self, _population, _sample_size: int) -> list[int]:
            raise AssertionError('small-range fallback should not be used')

    assert sample_range_exact(StubRandom(), 100, 2) == [7, 9]


def test_sample_range_exact_handles_edge_cases():
    assert sample_range_exact(random.Random(1), 0, 0) == []
    assert sample_range_exact(random.Random(2), 10**19, 0) == []

    with pytest.raises(ValueError, match='Sample larger than population or is negative'):
        sample_range_exact(random.Random(3), 5, 6)


def test_sample_range_exact_preserves_prefix_for_huge_ranges():
    full_sample = sample_range_exact(random.Random(42), 10**19, 50)
    prefix_sample = sample_range_exact(random.Random(42), 10**19, 10)

    assert full_sample[:10] == prefix_sample
    assert prefix_sample[:5] == HUGE_SPACE_FIRST_INDICES


def test_param_space_small_legacy_sequence_is_unchanged():
    params = {'a': [0, 1], 'b': [10, 20]}
    expected_sequence = [
        {'a': 1, 'b': 10},
        {'a': 1, 'b': 20},
        {'a': 0, 'b': 10},
    ]

    assert _collect_generated_params(
        seed=123,
        params=params,
        n_permutations=3,
        count=3,
        random_search=True,
    ) == expected_sequence


def test_param_space_enumerates_full_space_and_returns_none_when_exhausted():
    params = {'a': [0, 1], 'b': [10, 20]}
    previous_state = random.getstate()

    try:
        random.seed(999)
        param_space = ParamSpace(params, 10)
    finally:
        random.setstate(previous_state)

    assert param_space.n_permutations == 4

    generated = [
        param_space.generate(random_search=False),
        param_space.generate(random_search=False),
        param_space.generate(random_search=False),
        param_space.generate(random_search=False),
    ]
    assert generated == [
        {'a': 0, 'b': 10},
        {'a': 1, 'b': 10},
        {'a': 0, 'b': 20},
        {'a': 1, 'b': 20},
    ]
    assert param_space.generate(random_search=False) is None


def test_param_space_large_space_no_longer_overflows_and_is_reproducible():
    params = _build_huge_params()
    larger_params = _build_huge_params(25)
    previous_state = random.getstate()

    try:
        random.seed(42)
        first = ParamSpace(params, 10_000)

        random.seed(42)
        second = ParamSpace(params, 10_000)

        random.seed(7)
        larger = ParamSpace(larger_params, 64)
    finally:
        random.setstate(previous_state)

    assert first.n_permutations == 10_000
    assert second.n_permutations == 10_000
    assert larger.n_permutations == 64
    assert first.df_params.head(5).rows() == second.df_params.head(5).rows()
    assert set(larger.generate(random_search=False).keys()) == set(larger_params.keys())


def test_param_space_large_space_sequences_match_for_both_generation_modes():
    params = _build_huge_params()

    sequential_first = _collect_generated_params(
        seed=42,
        params=params,
        n_permutations=128,
        count=12,
        random_search=False,
    )
    sequential_second = _collect_generated_params(
        seed=42,
        params=params,
        n_permutations=128,
        count=12,
        random_search=False,
    )
    random_first = _collect_generated_params(
        seed=42,
        params=params,
        n_permutations=128,
        count=12,
        random_search=True,
    )
    random_second = _collect_generated_params(
        seed=42,
        params=params,
        n_permutations=128,
        count=12,
        random_search=True,
    )

    assert sequential_first == sequential_second
    assert random_first == random_second
