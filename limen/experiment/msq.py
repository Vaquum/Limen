from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from limen.experiment.param_domain import ParamDomain
from limen.experiment.search_strategy import SearchStrategy


class FilterExhaustedError(Exception):

    '''Raised when custom filters aggressively reject too many consecutive combinations.'''


class MSQ:

    '''
    Mutable Search Queue — wraps a SearchStrategy with intervention bookkeeping.

    Routes interventions to the correct layer:
    - Single-param operations (remove_is, keep_is, etc.) go to ParamDomain
    - Multi-param filters (remove_custom) stay here as post-generation checks
    - Injected combinations go to the priority deque

    '''

    def __init__(
        self,
        strategy: SearchStrategy,
        domain: ParamDomain,
        *,
        n_permutations: int | None = None,
        max_filter_retries: int = 1000,
    ) -> None:

        '''
        Initialize the MSQ.

        Args:
            strategy (SearchStrategy): SearchStrategy instance to generate combinations
            domain (ParamDomain): ParamDomain shared with the strategy
            n_permutations (int | None): Optional hard cap on total combinations yielded
            max_filter_retries (int): Max consecutive filter rejections before error

        '''

        if strategy.domain is not domain:
            raise ValueError(
                'strategy.domain and domain must reference the same ParamDomain'
            )
        self._strategy = strategy
        self._domain = domain
        self._n_permutations = n_permutations
        self._priority_queue: deque[dict[str, Any]] = deque()
        self._custom_filters: list[Callable[[dict[str, Any]], bool]] = []
        self._max_filter_retries = max_filter_retries
        self._trim_budget: int | None = None
        self._yielded_count: int = 0
        self._intervention_log: list[dict[str, Any]] = []


    def __iter__(self) -> MSQ:

        return self


    def __next__(self) -> dict[str, Any]:

        '''Return next parameter combination. Priority queue first, then strategy.'''

        if self._trim_budget is not None and self._trim_budget <= 0:
            raise StopIteration

        if self._n_permutations is not None and self._yielded_count >= self._n_permutations:
            raise StopIteration

        if self._priority_queue:
            combo = self._priority_queue.popleft()
            return self._yield_combo(combo, injected=True)

        for _ in range(self._max_filter_retries):
            try:
                combo = next(self._strategy)
            except StopIteration:
                raise

            if self._passes_filters(combo):
                return self._yield_combo(combo, injected=False)

        raise FilterExhaustedError(
            f"Custom filters rejected {self._max_filter_retries} "
            f"consecutive combinations. "
            f"Active filters: {len(self._custom_filters)}. "
            f"Consider relaxing filters or removing them."
        )


    def _yield_combo(
        self, combo: dict[str, Any], *, injected: bool,
    ) -> dict[str, Any]:

        '''Enrich combo with metadata, update counters, and return.'''

        combo = dict(combo)
        combo['_id'] = self._yielded_count
        combo['_injected'] = injected
        self._yielded_count += 1
        if self._trim_budget is not None:
            self._trim_budget -= 1

        return combo


    def _passes_filters(self, combo: dict[str, Any]) -> bool:

        '''Return True if no filter wants to remove this combo.'''

        return not any(f(combo) for f in self._custom_filters)


    def remove_is(self, param: str, value: Any) -> bool:

        '''Remove a specific value from a parameter's domain.'''

        self._log_intervention('remove_is', param=param, value=value)

        return self._domain.remove_value(param, value)


    def remove_ge(self, param: str, threshold: Any) -> int:

        '''Remove all values >= threshold from a parameter's domain.'''

        self._log_intervention('remove_ge', param=param, threshold=threshold)

        return self._domain.remove_values_ge(param, threshold)


    def remove_le(self, param: str, threshold: Any) -> int:

        '''Remove all values <= threshold from a parameter's domain.'''

        self._log_intervention('remove_le', param=param, threshold=threshold)

        return self._domain.remove_values_le(param, threshold)


    def remove_custom(
        self, condition: Callable[[dict[str, Any]], bool],
    ) -> None:

        '''
        Add a multi-parameter filter applied post-generation.

        Args:
            condition (Callable): Function taking a combo dict, returning True
                to REMOVE, False to keep

        '''

        self._log_intervention('remove_custom', condition=repr(condition))
        self._custom_filters.append(condition)


    def trim(self, target_count: int) -> None:

        '''
        Limit remaining combinations to target_count.

        Sets a budget counter. After this many more combinations are
        yielded, StopIteration is raised.

        '''

        self._log_intervention('trim', target_count=target_count)
        remaining = max(target_count - self._yielded_count, 0)
        self._trim_budget = remaining


    def keep_is(self, param: str, value: Any) -> int:

        '''Keep only a specific value for a parameter, removing all others.'''

        self._log_intervention('keep_is', param=param, value=value)

        return self._domain.keep_values(param, [value])


    def keep_between(self, param: str, lower: Any, upper: Any) -> int:

        '''Keep only values within [lower, upper] for a parameter.'''

        self._log_intervention(
            'keep_between', param=param, lower=lower, upper=upper,
        )

        return self._domain.keep_between(param, lower, upper)


    def inject(
        self, combo: dict[str, Any], *, prioritize: bool = False,
    ) -> None:

        '''
        Inject a specific combination into the search.

        Args:
            combo (dict[str, Any]): Full parameter combination dict
            prioritize (bool): If True, inserted at front of queue (next to yield).
                If False, appended to back

        '''

        combo = dict(combo)
        self._log_intervention(
            'inject', combo=dict(combo), prioritize=prioritize,
        )
        combo_keys = set(combo.keys())
        domain_keys = set(self._domain.keys)
        missing = domain_keys - combo_keys
        if missing:
            raise ValueError(f"Injected combo missing parameters: {missing}")
        extra = combo_keys - domain_keys
        if extra:
            raise ValueError(f"Injected combo has extra keys: {extra}")

        if prioritize:
            self._priority_queue.appendleft(combo)
        else:
            self._priority_queue.append(combo)


    def inject_value(self, param: str, value: Any) -> bool:

        '''Inject a new value into a parameter's domain (expands param space).'''

        self._log_intervention('inject_value', param=param, value=value)

        return self._domain.inject_value(param, value)


    def remaining_count(self) -> int | None:

        '''
        Return count of remaining combinations, or None if unknown.

        Uses trim budget if set, else n_permutations if set,
        else total_combinations for finite strategies.
        Returns None for infinite strategies without a budget.

        '''

        if self._trim_budget is not None:
            return self._trim_budget

        queue_size = len(self._priority_queue)

        if self._n_permutations is not None:
            remaining = max(
                self._n_permutations - self._yielded_count, 0,
            )
            return remaining + queue_size

        if self._strategy.is_finite:
            remaining = max(
                self._domain.total_combinations - self._strategy.generated_count, 0,
            )
            return remaining + queue_size

        return None


    def distribution(self, param: str | None = None) -> dict:

        '''
        Return estimated count of pending combinations per value.

        Assumes uniform sampling across the domain. Accurate for Grid
        and Random strategies. Adaptive strategies (e.g. TPE, Bayesian)
        may distribute runs non-uniformly; treat these counts as
        approximate in that case.

        Args:
            param (str | None): If given, return {value: count} for that param.
                If None, return {param: {value: count, ...}, ...} for all

        Returns:
            dict: Value→count, or param→{value: count}.
                Empty dict if remaining count is unknown

        '''

        remaining = self.remaining_count()
        if remaining is None:
            return {}

        remaining_from_strategy = max(
            remaining - len(self._priority_queue), 0,
        )

        if param is not None:
            n_values = len(self._domain.values_for(param))
            count = remaining_from_strategy // n_values
            return dict.fromkeys(self._domain.values_for(param), count)

        result = {}
        for p in self._domain.keys:
            vals = self._domain.values_for(p)
            count = remaining_from_strategy // len(vals)
            result[p] = dict.fromkeys(vals, count)

        return result


    @property
    def priority_queue_size(self) -> int:

        return len(self._priority_queue)


    @property
    def yielded_count(self) -> int:

        return self._yielded_count


    @property
    def intervention_log(self) -> list[dict[str, Any]]:

        return list(self._intervention_log)


    def get_state(self) -> dict[str, Any]:

        '''
        Export state for checkpointing.

        NOTE: Does not include ParamDomain state. Callers must
        persist and restore the domain separately.

        '''

        return {
            'yielded_count': self._yielded_count,
            'n_permutations': self._n_permutations,
            'trim_budget': self._trim_budget,
            'priority_queue': list(self._priority_queue),
            'custom_filters_count': len(self._custom_filters),
            'intervention_log': list(self._intervention_log),
            'strategy_state': self._strategy.get_state(),
        }


    def set_state(self, state: dict[str, Any]) -> None:

        '''
        Restore state from checkpoint.

        NOTE: Does not restore ParamDomain state. Callers must
        restore the domain separately before calling this method.

        '''

        required = ('yielded_count', 'trim_budget', 'priority_queue', 'intervention_log', 'strategy_state')
        for key in required:
            if key not in state:
                raise ValueError(f"Invalid MSQ state: missing required key '{key}'.")

        self._yielded_count = state['yielded_count']
        self._n_permutations = state.get('n_permutations')
        self._trim_budget = state['trim_budget']
        self._priority_queue = deque(state['priority_queue'])
        self._custom_filters = []
        self._intervention_log = list(state['intervention_log'])
        self._strategy.set_state(state['strategy_state'])

        if state.get('custom_filters_count', 0) > 0:
            import warnings
            warnings.warn(
                f"Checkpoint had {state['custom_filters_count']} custom "
                f"filters that cannot be restored from serialized state. "
                f"Re-register them after resume.",
                stacklevel=2,
            )


    def _log_intervention(self, operation: str, **kwargs: Any) -> None:
        self._intervention_log.append({
            'operation': operation,
            'at_yielded_count': self._yielded_count,
            **kwargs,
        })
