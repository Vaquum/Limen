from __future__ import annotations

import random
from typing import Any

from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.search_strategy import SearchStrategy

MAX_DEDUP_RETRIES = 1000


class RandomStrategy(SearchStrategy):

    '''
    Lazy random sampling from the parameter domain.

    Samples each parameter independently via uniform random choice.
    Infinite — never raises StopIteration. Uses _is_unseen for
    dedup to avoid repeating previously seen combinations.

    '''

    def __init__(self, domain: ParamDomain, *, seed: int | None = None) -> None:

        super().__init__(domain, seed=seed)
        self._rng = random.Random(seed)


    @property
    def is_finite(self) -> bool:

        return False


    def __next__(self) -> dict[str, Any]:

        params = self._domain.params
        for _ in range(MAX_DEDUP_RETRIES):
            combo = {k: self._rng.choice(v) for k, v in params.items()}
            if self._is_unseen(combo):
                self._generated_count += 1
                return combo

        raise RuntimeError(
            f"RandomStrategy failed to generate a novel combination "
            f"after {MAX_DEDUP_RETRIES} retries. "
            f"Domain may be nearly exhausted ({len(self._seen)} seen)."
        )


    def on_domain_changed(
        self, _domain: ParamDomain, _changed_params: list[str],
    ) -> None:

        return


    def get_state(self) -> dict[str, Any]:

        return {
            'rng_state': self._rng.getstate(),
            'generated_count': self._generated_count,
        }


    def set_state(self, state: dict[str, Any]) -> None:

        rng_state = state['rng_state']
        # JSON serialization converts tuples to lists; restore tuple structure
        if isinstance(rng_state, list):
            rng_state = (rng_state[0], tuple(rng_state[1]), rng_state[2])
        self._rng.setstate(rng_state)
        self._generated_count = state['generated_count']
