from __future__ import annotations

from typing import Any, Protocol


class DomainObserver(Protocol):

    '''Observer that receives notifications when ParamDomain is mutated.'''

    def on_domain_changed(self, domain: ParamDomain, changed_params: list[str]) -> None:
        ...


class ParamDomain:

    '''Mutable parameter space definition with observer pattern.'''

    def __init__(self, params: dict[str, list[Any]]) -> None:

        '''
        Initialize the ParamDomain.

        Args:
            params (dict[str, list[Any]]): Dict mapping parameter names to non-empty lists of values

        '''

        for k, v in params.items():
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError(
                    f"Parameter '{k}' must be a non-empty list, got {v!r}"
                )

        self._params: dict[str, list[Any]] = {k: list(v) for k, v in params.items()}
        self._observers: list[DomainObserver] = []
        self._version: int = 0


    @property
    def params(self) -> dict[str, list[Any]]:

        '''Current parameter values. Returns a defensive copy.'''

        return {k: list(v) for k, v in self._params.items()}


    @property
    def version(self) -> int:
        return self._version


    @property
    def keys(self) -> list[str]:
        return list(self._params.keys())


    def values_for(self, param: str) -> list[Any]:

        '''Return current legal values for a parameter.'''

        return list(self._params[param])


    @property
    def total_combinations(self) -> int:

        '''Product of all parameter list lengths.'''

        result = 1
        for v in self._params.values():
            result *= len(v)
        return result


    def add_observer(self, observer: DomainObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)


    def remove_observer(self, observer: DomainObserver) -> None:
        self._observers.remove(observer)


    def _notify(self, changed_params: list[str]) -> None:
        self._version += 1
        for obs in self._observers:
            obs.on_domain_changed(self, changed_params)


    def remove_value(self, param: str, value: Any) -> bool:

        '''
        Remove a single value from a parameter's domain.

        Args:
            param (str): Parameter name
            value (Any): Value to remove

        Returns:
            bool: True if removed, False if value was not present

        Raises:
            ValueError: If removal would leave the parameter with no values

        '''

        values = self._params[param]
        if value not in values:
            return False
        if len(values) == 1:
            raise ValueError(
                f"Cannot remove '{value}' from '{param}': "
                f"would leave domain empty."
            )
        values.remove(value)
        self._notify([param])
        return True


    def remove_values_ge(self, param: str, threshold: Any) -> int:

        '''
        Remove all values >= threshold.

        Returns:
            int: Count of values removed

        Raises:
            TypeError: If values are not comparable with threshold
            ValueError: If removal would leave the parameter with no values

        '''

        original = self._params[param]
        try:
            kept = [v for v in original if v < threshold]
        except TypeError as e:
            raise TypeError(
                f"Cannot compare values of '{param}' with threshold "
                f"{threshold!r}: {e}"
            ) from e
        if len(kept) == 0:
            raise ValueError(
                f"remove_ge({param}, {threshold}) would remove all "
                f"{len(original)} values. Current values: {original}"
            )
        removed_count = len(original) - len(kept)
        if removed_count > 0:
            self._params[param] = kept
            self._notify([param])
        return removed_count


    def remove_values_le(self, param: str, threshold: Any) -> int:

        '''
        Remove all values <= threshold.

        Returns:
            int: Count of values removed

        Raises:
            TypeError: If values are not comparable with threshold
            ValueError: If removal would leave the parameter with no values

        '''

        original = self._params[param]
        try:
            kept = [v for v in original if v > threshold]
        except TypeError as e:
            raise TypeError(
                f"Cannot compare values of '{param}' with threshold "
                f"{threshold!r}: {e}"
            ) from e
        if len(kept) == 0:
            raise ValueError(
                f"remove_le({param}, {threshold}) would remove all "
                f"{len(original)} values. Current values: {original}"
            )
        removed_count = len(original) - len(kept)
        if removed_count > 0:
            self._params[param] = kept
            self._notify([param])
        return removed_count


    def keep_values(self, param: str, values: list[Any]) -> int:

        '''
        Keep only specified values for a parameter.

        Returns:
            int: Count of values removed

        Raises:
            ValueError: If no overlap with current values

        '''

        original = self._params[param]
        kept = [v for v in original if v in values]
        if len(kept) == 0:
            raise ValueError(
                f"keep_values({param}, {values}) would remove all values. "
                f"No overlap with current: {original}"
            )
        removed_count = len(original) - len(kept)
        if removed_count > 0:
            self._params[param] = kept
            self._notify([param])
        return removed_count


    def keep_between(self, param: str, lower: Any, upper: Any) -> int:

        '''
        Keep only values where lower <= value <= upper.

        Returns:
            int: Count of values removed

        Raises:
            TypeError: If values are not comparable with bounds
            ValueError: If no values fall within range

        '''

        original = self._params[param]
        try:
            kept = [v for v in original if lower <= v <= upper]
        except TypeError as e:
            raise TypeError(
                f"Cannot compare values of '{param}' with bounds "
                f"[{lower!r}, {upper!r}]: {e}"
            ) from e
        if len(kept) == 0:
            raise ValueError(
                f"keep_between({param}, {lower}, {upper}) would remove "
                f"all values. Current values: {original}"
            )
        removed_count = len(original) - len(kept)
        if removed_count > 0:
            self._params[param] = kept
            self._notify([param])
        return removed_count


    def inject_value(self, param: str, value: Any) -> bool:

        '''
        Add a new value to a parameter's domain.

        Returns:
            bool: True if added, False if value was already present

        '''

        if value in self._params[param]:
            return False
        self._params[param].append(value)
        self._notify([param])
        return True


    def is_valid_combination(self, combo: dict[str, Any]) -> bool:

        '''Check whether a combination only uses currently-legal values.'''

        for k, v in combo.items():
            if k not in self._params or v not in self._params[k]:
                return False
        return True


    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        return {k: list(v) for k, v in self._params.items()}


    def set_state(self, state: dict[str, Any]) -> None:

        '''
        Restore state from checkpoint.

        NOTE: Version resets to 0 and all observers are notified so
        strategies rebuild their internal state.

        Args:
            state (dict): State dict from get_state()

        '''

        self._params = {k: list(v) for k, v in state.items()}
        self._version = 0
        changed_params = list(self._params.keys())
        for obs in self._observers:
            obs.on_domain_changed(self, changed_params)
