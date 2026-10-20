from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from inspect import ismethod
from typing import Literal, Protocol, cast

_CPU_KEYS = ('n_jobs', 'nthread', 'num_threads', 'device', 'device_type')
_SEED_KEYS = (*_CPU_KEYS, 'random_state', 'seed')
_VALIDATION_KEYS = {
    'deterministic_stage': _CPU_KEYS, 'seeded_stage': _SEED_KEYS,
    'sklearn_stage': (*_SEED_KEYS, 'solver'), 'lightgbm_stage': _SEED_KEYS,
    'xgboost_stage': (*_SEED_KEYS, 'tree_method', 'grow_policy', 'sampling_method', 'multi_strategy', 'booster', 'num_parallel_tree'),
    'composed_stage': (*_CPU_KEYS, 'seed'), 'unfrozen_stage': (),
}


@dataclass(frozen=True)
class ParallelRoundContract:
    validate: Callable[[Mapping[str, object]], None]
    parameter_keys: tuple[str, ...] = ()
    backend_imports: tuple[str, ...] = ()
    prediction_kind: Literal['binary', 'continuous', 'target_exposure'] | None = None
    probabilities: bool = False
    validation_keys: tuple[str, ...] | None = None


class _Declared(Protocol):
    __parallel_round_contract__: ParallelRoundContract


def declare_parallel(component: object, contract: ParallelRoundContract) -> None:
    if not callable(component):
        raise TypeError('Parallel declarations require a callable and ParallelRoundContract')
    cast(_Declared, component.__func__ if ismethod(component) else component).__parallel_round_contract__ = contract


def _validate(name: str, params: Mapping[str, object]) -> None:
    module = import_module('limen.experiment._parallel_resources')
    cast(Callable[[Mapping[str, object]], None], getattr(module, name))(params)


def deterministic_stage(params: Mapping[str, object]) -> None:
    _validate('deterministic_stage', params)


def seeded_stage(params: Mapping[str, object]) -> None:
    _validate('seeded_stage', params)


def sklearn_stage(params: Mapping[str, object]) -> None:
    _validate('sklearn_stage', params)


def lightgbm_stage(params: Mapping[str, object]) -> None:
    _validate('lightgbm_stage', params)


def xgboost_stage(params: Mapping[str, object]) -> None:
    _validate('xgboost_stage', params)


def composed_stage(params: Mapping[str, object]) -> None:
    _validate('composed_stage', params)


def unfrozen_stage(params: Mapping[str, object]) -> None:
    _validate('unfrozen_stage', params)


def attach(component: object, validator: str = 'deterministic_stage', *,
           parameter_keys: tuple[str, ...] = (), backend_imports: tuple[str, ...] = (),
           prediction_kind: Literal['binary', 'continuous', 'target_exposure'] | None = None,
           probabilities: bool = False) -> None:
    validate = cast(Callable[[Mapping[str, object]], None], getattr(import_module(__name__), validator))
    declare_parallel(component, ParallelRoundContract(validate, parameter_keys, backend_imports,
                                                     prediction_kind, probabilities, _VALIDATION_KEYS[validator]))
