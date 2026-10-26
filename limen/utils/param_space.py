import math
import random
import sys
from typing import Any

import numpy as np
import polars as pl
from limen._parameter_identity import admit_choices


SAMPLE_SELECTION_MIN_SIZE = 21
SAMPLE_SELECTION_TIPPING_POINT = 5
SAMPLE_SELECTION_LOG_BASE = 4
SAMPLE_SELECTION_MULTIPLIER = 3


def sample_range_exact(
    rng: Any,
    population_size: int,
    sample_size: int,
) -> list[int]:

    '''
    Sample ordered indices with exact `random.sample(range(N), k)` semantics.

    Args:
        rng (Any): Random provider with `sample()` and `getrandbits()`
        population_size (int): Size of the conceptual `range(N)` population
        sample_size (int): Number of unique ordered indices to draw

    Returns:
        list[int]: Ordered sampled indices matching stdlib range-sampling behavior
    '''

    # Mirror CPython Lib/random.py for range(N) sampling so huge-space ParamSpace
    # keeps legacy random.sample semantics even when the wrapper overflows.
    if not 0 <= sample_size <= population_size:
        raise ValueError('sample_range_exact Sample larger than population or is negative')

    setsize = SAMPLE_SELECTION_MIN_SIZE
    if sample_size > SAMPLE_SELECTION_TIPPING_POINT:
        setsize += SAMPLE_SELECTION_LOG_BASE ** math.ceil(
            math.log(
                sample_size * SAMPLE_SELECTION_MULTIPLIER,
                SAMPLE_SELECTION_LOG_BASE,
            )
        )

    if population_size <= setsize and population_size <= sys.maxsize:
        return rng.sample(range(population_size), sample_size)

    bits = population_size.bit_length()
    get_bits = rng.getrandbits
    seen: set[int] = set()
    ordered_indices: list[int] = []

    for _ in range(sample_size):
        index = get_bits(bits)
        while index >= population_size or index in seen:
            index = get_bits(bits)
        seen.add(index)
        ordered_indices.append(index)

    return ordered_indices


class ParamSpace:

    """Sample parameter combinations without constructing the Cartesian product.

    Preserve stdlib sampling beyond Py_ssize_t; sample=False selects the grid prefix.
    """

    def __init__(self, params: dict[str, list[Any]], n_permutations: int, seed: int | None = None, *, sample: bool = True) -> None:

        super().__init__()

        self.params, self.keys = admit_choices(params), list(params.keys())
        self._rng = random.Random(seed)
        self.param_sizes = [len(params[k]) for k in self.keys]
        self.total_space = 1
        for size in self.param_sizes:
            self.total_space *= size

        # Select the requested prefix or sample unique indices
        if n_permutations >= self.total_space:
            indices = list(range(self.total_space))
        elif not sample and n_permutations >= 0:
            indices = list(range(n_permutations))
        elif self.total_space <= sys.maxsize:
            indices = self._rng.sample(range(self.total_space), n_permutations)
        else:
            indices = sample_range_exact(self._rng, self.total_space, n_permutations)

        if indices and self.keys and self.total_space <= sys.maxsize and all(
            type(values[0]) in (bool, int, float, str)
            and all(type(value) is type(values[0]) for value in values)
            and (type(values[0]) is not int or all(-(2**63) <= value < 2**63 for value in values))
            for values in params.values()
        ):
            remaining = np.asarray(indices, dtype=np.int64)
            columns: list[pl.Series] = []
            for key, size in zip(self.keys, self.param_sizes, strict=True):
                remaining, offsets = np.divmod(remaining, size)
                columns.append(pl.Series(key, params[key]).gather(offsets))
            self.df_params = pl.DataFrame(columns)
        else:
            self.df_params = pl.DataFrame([self._index_to_combo(idx) for idx in indices])
        self.n_permutations = self.df_params.height

    def _index_to_combo(self, index: int) -> dict[str, Any]:
        combo: dict[str, Any] = {}
        remaining_index = index
        for i, key in enumerate(self.keys):
            size = self.param_sizes[i]
            combo[key] = self.params[key][remaining_index % size]
            remaining_index //= size
        return combo

    def generate(self, random_search: bool = True) -> dict[str, Any] | None:

        '''
        Compute next parameter combination from the parameter space.

        Args:
            random_search (bool): Whether to select parameters randomly or sequentially

        Returns:
            dict: Dictionary of parameter names and selected values, or None if space is exhausted
        '''

        if self.df_params.is_empty():
            return None

        row_no = self._rng.randrange(self.df_params.height) if random_search else 0

        round_params = dict(zip(self.df_params.columns, self.df_params.row(row_no), strict=True))

        self.df_params = (
            self.df_params
              .with_row_index('__idx')
              .filter(pl.col('__idx') != row_no)
              .drop('__idx')
        )

        return round_params
