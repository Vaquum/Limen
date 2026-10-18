from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search._grid_permutation import domain_digest
from limen.experiment.param_search.random_strategy import MAX_DEDUP_RETRIES
from limen.experiment.param_search.search_strategy import SearchStrategy

if TYPE_CHECKING:
    from scipy.stats import qmc

__all__ = ['SobolStrategy']


class SobolStrategy(SearchStrategy):
    """Scrambled Sobol points projected onto the current discrete domain."""

    def __init__(self, domain: ParamDomain, *, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError('SobolStrategy seed must be a non-negative non-boolean integer')
        if not domain.keys:
            raise ValueError('SobolStrategy requires at least one parameter')
        from scipy import __version__

        super().__init__(domain, seed=seed)
        self._scipy_version = __version__
        self._raw_index = 0
        self._refresh_cache()
        self._engine = self._new_engine(seed)

    def _new_engine(self, seed: int) -> qmc.Sobol:
        import numpy as np
        from scipy.stats import qmc

        return qmc.Sobol(d=len(self._keys), scramble=True, bits=30,
                         rng=np.random.default_rng(seed))

    def _refresh_cache(self) -> None:
        params = self._domain.params
        self._keys = sorted(params)
        self._values = [params[key] for key in self._keys]

    @override
    def on_domain_changed(self, _domain: ParamDomain, _changed_params: list[str]) -> None:
        self._refresh_cache()

    @override
    def __next__(self) -> dict[str, object]:
        for _ in range(MAX_DEDUP_RETRIES):
            if self._raw_index == 2 ** 30:
                raise RuntimeError('SobolStrategy exhausted its 2**30 raw Sobol points')
            point = self._engine.random(1)[0]
            self._raw_index += 1
            combo: dict[str, object] = {
                key: values[int(coordinate * len(values))]
                for key, values, coordinate in zip(self._keys, self._values, point, strict=True)
            }
            if self._is_unseen(combo):
                self._generated_count += 1
                return combo
        raise RuntimeError(
            f'SobolStrategy failed to generate a novel combination after {MAX_DEDUP_RETRIES} ' +
            f'retries. Domain may be nearly exhausted ({len(self._seen)} seen).'
        )

    @override
    def get_state(self) -> dict[str, object]:
        return {
            'version': 1,
            'seed': self._seed,
            'raw_index': self._raw_index,
            'generated_count': self._generated_count,
            'domain_digest': domain_digest(self._domain.params),
            'scipy_version': self._scipy_version,
        }

    @override
    def set_state(self, state: dict[str, object]) -> None:
        if set(state) != {'version', 'seed', 'raw_index', 'generated_count',
                          'domain_digest', 'scipy_version'}:
            raise ValueError('SobolStrategy checkpoint fields are invalid')
        version, seed = state['version'], state['seed']
        if type(version) is not int or version != 1:
            raise ValueError('SobolStrategy checkpoint version is unsupported')
        if type(seed) is not int or seed != self._seed:
            raise ValueError('SobolStrategy checkpoint seed does not match current seed')
        if state['domain_digest'] != domain_digest(self._domain.params):
            raise ValueError('SobolStrategy checkpoint ordered domain does not match current domain')
        if state['scipy_version'] != self._scipy_version:
            raise ValueError('SobolStrategy checkpoint SciPy version does not match current version')
        index, generated = state['raw_index'], state['generated_count']
        if type(index) is not int or type(generated) is not int or not 0 <= generated <= index <= 2 ** 30:
            raise ValueError('SobolStrategy checkpoint traversal counters are invalid')
        engine = self._new_engine(seed)
        if index:
            _ = engine.fast_forward(index)
        self._engine = engine
        self._raw_index, self._generated_count = index, generated
