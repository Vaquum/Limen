from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from collections.abc import Iterator
from typing import Any

from limen.experiment.param_domain import ParamDomain


class SearchStrategy(ABC):

    '''
    Abstract base for all search strategies.

    Strategies are iterators that yield dict[str, Any] combination of parameters.
    They hold a reference to a ParamDomain and register as observers
    so they can react to domain mutations.

    '''

    def __init__(self, domain: ParamDomain, *, seed: int | None = None) -> None:

        '''
        Initialize the SearchStrategy.

        Args:
            domain (ParamDomain): ParamDomain to generate combinations from
            seed (int | None): Optional random seed for reproducibility

        '''

        self._domain = domain
        self._seed = seed
        self._generated_count: int = 0  # subclasses must increment in __next__
        self._seen: set[str] = set()
        self._last_param_hash: str | None = None
        self._domain.add_observer(self)


    def __iter__(self) -> Iterator[dict[str, Any]]:

        return self


    @abstractmethod
    def __next__(self) -> dict[str, Any]:

        '''Generate next combination. Raise StopIteration when exhausted.'''

        ...


    def on_domain_changed(
        self, _domain: ParamDomain, _changed_params: list[str],
    ) -> None:

        '''
        Called when ParamDomain is mutated. Override if strategy
        maintains state dependent on the domain.

        '''

        return


    def update_from_feedback(
        self, _log: Any, _interventions: list[dict],
    ) -> None:

        '''
        Hook for strategies that adapt based on experiment feedback.
        Default is no-op. Stateful strategies (e.g. TPE) override this.

        '''

        return


    @property
    def domain(self) -> ParamDomain:

        return self._domain


    @property
    def generated_count(self) -> int:

        return self._generated_count


    @property
    def is_finite(self) -> bool:

        '''
        Whether this strategy has a finite number of combinations.
        Override to return True for exhaustive strategies (e.g. Grid).

        '''

        return False


    @property
    def last_param_hash(self) -> str | None:

        '''Hash of the last generated combination, or None.'''

        return self._last_param_hash


    def _compute_param_hash(self, combo: dict[str, Any]) -> str:

        '''
        Compute deterministic SHA-256 hash of a parameter combination.

        Args:
            combo (dict[str, Any]): Parameter combination

        Returns:
            str: 16-character hex hash
        '''

        canonical = json.dumps(combo, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


    def _is_novel(self, combo: dict[str, Any]) -> bool:

        '''
        Check if a combination has been seen before.

        Computes the hash, checks against _seen set. If novel, adds
        to _seen and stores in _last_param_hash. Stochastic strategies
        call this in __next__ to skip duplicates.

        Args:
            combo (dict[str, Any]): Parameter combination to check

        Returns:
            bool: True if novel (not seen before)
        '''

        h = self._compute_param_hash(combo)
        if h in self._seen:
            return False
        self._seen.add(h)
        self._last_param_hash = h
        return True


    def rebuild_seen(self, hashes: Iterable[str]) -> None:

        '''
        Populate _seen from an iterable of hashes. Used on resume
        to reconstruct dedup state from the experiment log.

        Args:
            hashes (Iterable[str]): Hash strings from previous runs
        '''

        self._seen = set(hashes)


    @abstractmethod
    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        ...


    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        ...
