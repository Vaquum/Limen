from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Literal

__all__ = ['ReuseContract', 'declare_reuse']


@dataclass(frozen=True)
class ReuseContract:
    """Declare deterministic snapshot work and its complete input dependencies.

    The author owns the purity/dependency claim. A validator raises ValueError
    for unsupported effective settings; replay recomputes every economic field.
    """

    validate: Callable[[Mapping[str, object]], None]
    parameter_keys: tuple[str, ...] = ()
    validation_keys: tuple[str, ...] | None = None
    consumes: tuple[str, ...] | None = None
    prediction_kind: Literal['binary', 'continuous'] | None = None
    replay: Callable[[Mapping[str, object], Mapping[str, object], Mapping[str, object]], dict[str, object]] | None = None
    replay_inputs: tuple[str, ...] = ()


def declare_reuse(component: object, contract: ReuseContract) -> None:
    """Attach an explicit invocation-local reuse contract to a pipeline callable."""
    if not callable(component) or not isinstance(contract, ReuseContract):
        raise TypeError('Reuse declarations require a callable and ReuseContract')
    if not callable(contract.validate) or contract.prediction_kind not in (None, 'binary', 'continuous'):
        raise ValueError('Invalid reuse validator or prediction kind')
    if contract.replay is not None and not callable(contract.replay):
        raise TypeError('Reuse replay must be callable')
    for keys in (contract.parameter_keys, contract.validation_keys, contract.consumes, contract.replay_inputs):
        if keys is not None and (not isinstance(keys, tuple) or any(not isinstance(key, str) or not key for key in keys)):
            raise ValueError('Reuse dependency declarations require tuples of nonempty paths')
    setattr(component, '__limen_reuse__', contract)


def declared(component: object) -> ReuseContract | None:
    contract = getattr(component, '__limen_reuse__', None)
    if isinstance(contract, ReuseContract):
        return contract
    if isinstance(component, partial):
        return declared(component.func)
    return None
