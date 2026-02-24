from __future__ import annotations

from abc import ABC, abstractmethod
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


    @abstractmethod
    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        ...


    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        ...
