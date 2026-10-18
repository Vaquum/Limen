from __future__ import annotations

import math
from typing import Any

from typing_extensions import override

from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search._grid_permutation import (
    _lcg_map, _lcg_params, _permuted_index, _SHUFFLE_VERSION, domain_digest,
)
from limen.experiment.param_search.search_strategy import SearchStrategy

__all__ = ['GridStrategy']


class GridStrategy(SearchStrategy):
    """Finite mixed-radix traversal; seeded shuffle uses a versioned permutation."""

    def __init__(self, domain: ParamDomain, *, seed: int | None = None,
                 shuffle: bool = False) -> None:
        if type(shuffle) is not bool:
            raise ValueError('GridStrategy shuffle must be a bool')
        if seed is not None and type(seed) is not int:
            raise ValueError('GridStrategy seed must be a non-boolean integer or None')
        super().__init__(domain, seed=seed)
        self._shuffle = shuffle
        self._permutation_version = _SHUFFLE_VERSION if shuffle and seed is not None else 1
        self._rebuild()

    @property
    @override
    def is_finite(self) -> bool:
        return True

    def _rebuild(self) -> None:
        params = self._domain.params
        self._keys = sorted(params.keys())
        self._values = [params[k] for k in self._keys]
        self._sizes = [len(v) for v in self._values]
        self._total = math.prod(self._sizes)
        self._current_index = 0
        self._generated_count = 0
        if self._shuffle and self._total > 0 and self._permutation_version == 1:
            self._lcg_multiplier, self._lcg_increment = _lcg_params(self._total, self._seed)

    def _permute_index(self, logical_index: int) -> int:
        if not self._shuffle:
            return logical_index
        if self._permutation_version == _SHUFFLE_VERSION:
            assert self._seed is not None
            return _permuted_index(logical_index, self._total, self._seed)
        return _lcg_map(logical_index, self._total, self._lcg_multiplier, self._lcg_increment)

    def _index_to_combo(self, index: int) -> dict[str, Any]:
        combo: dict[str, Any] = {}
        remaining = index
        for i, key in enumerate(self._keys):
            combo[key] = self._values[i][remaining % self._sizes[i]]
            remaining //= self._sizes[i]
        return combo

    @override
    def __next__(self) -> dict[str, Any]:
        while self._current_index < self._total:
            physical = self._permute_index(self._current_index)
            combo = self._index_to_combo(physical)
            self._current_index += 1
            if self._is_unseen(combo):
                self._generated_count += 1
                return combo
        raise StopIteration

    @override
    def on_domain_changed(self, _domain: ParamDomain, _changed_params: list[str]) -> None:
        self._rebuild()

    @override
    def get_state(self) -> dict[str, Any]:
        return {
            'current_index': self._current_index,
            'generated_count': self._generated_count,
            'shuffle': self._shuffle,
            'seed': self._seed,
            'permutation_version': self._permutation_version,
            'domain_digest': domain_digest(self._domain.params),
        }

    @override
    def set_state(self, state: dict[str, Any]) -> None:
        saved_shuffle, saved_seed = state.get('shuffle'), state.get('seed')
        version = state.get('permutation_version', 1)
        if type(version) is not int or version not in (1, _SHUFFLE_VERSION):
            raise ValueError('GridStrategy checkpoint permutation version is unsupported')
        if saved_shuffle is not None and saved_shuffle != self._shuffle:
            raise ValueError('GridStrategy checkpoint shuffle does not match current shuffle')
        if saved_seed is not None and saved_seed != self._seed:
            raise ValueError('GridStrategy checkpoint seed does not match current seed')
        if version == _SHUFFLE_VERSION:
            if not self._shuffle or saved_shuffle is not True or type(saved_seed) is not int or saved_seed != self._seed:
                raise ValueError('GridStrategy version 2 requires matching seeded shuffle')
            if state.get('domain_digest') != domain_digest(self._domain.params):
                raise ValueError('GridStrategy checkpoint ordered domain does not match current domain')
        index, generated = state['current_index'], state['generated_count']
        if type(index) is not int or type(generated) is not int or not 0 <= generated <= index <= self._total:
            raise ValueError('GridStrategy checkpoint traversal counters are invalid')
        if self._shuffle and self._total > 0 and version == 1 and self._permutation_version != version:
            self._lcg_multiplier, self._lcg_increment = _lcg_params(self._total, self._seed)
        self._permutation_version = version
        self._current_index, self._generated_count = index, generated
