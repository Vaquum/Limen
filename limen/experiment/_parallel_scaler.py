from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, getattr_static, isdatadescriptor, isfunction, ismethod, ismethoddescriptor, signature
from typing import Protocol, cast

import polars as pl

from limen._parallel_contract import ParallelRoundContract, attach
from limen.experiment._parallel_identity import identity_snapshot
from limen.scalers.registry import SCALER_REGISTRY


class TransformClass(Protocol):
    transform: Callable[..., pl.DataFrame]


@dataclass(frozen=True)
class CapturedTransform:
    """Bind captured behavior without changing the fitted object's type or state."""

    owner: type
    function: Callable[..., pl.DataFrame]
    bind_instance: bool
    __parallel_round_contract__: ParallelRoundContract | None

    def __call__(self, instance: object, data: pl.DataFrame, **params: object) -> pl.DataFrame:
        if getattr_static(type(instance), 'transform') is not getattr_static(self.owner, 'transform'):
            raise TypeError('Parallel transform has an uncaptured subclass implementation')
        if getattr_static(instance, 'transform') is not getattr_static(type(instance), 'transform'):
            defaults = cast(Callable[..., dict[str, object]], vars(import_module('limen.experiment._parallel_resources'))['_defaults'])
            assert self.__parallel_round_contract__ is not None
            self.__parallel_round_contract__.validate(defaults(cast(TransformClass, instance).transform, params, skip=1))
            return cast(TransformClass, instance).transform(data, **params)
        if isinstance(getattr_static(self.owner, 'transform'), classmethod) and ismethod(self.function):
            return cast(Callable[..., pl.DataFrame], self.function.__func__)(type(instance), data, **params)
        return self.function(instance, data, **params) if self.bind_instance and not (hasattr(instance, '__dict__') and 'transform' in vars(instance)) else self.function(data, **params)

    def resource_defaults(self) -> dict[str, object]:
        return {name: cast(object, parameter.default) for name, parameter in signature(self.function).parameters.items()
                if parameter.default is not Parameter.empty}


def capture_transform(cls: type) -> CapturedTransform:
    function = cast(TransformClass, cls).transform
    if ismethod(function) and isinstance(function.__self__, type) and not isinstance(getattr_static(cls, 'transform'), classmethod):
        raise TypeError('Parallel transform has unfrozen class-method owner state')
    return CapturedTransform(cls, function, isfunction(getattr_static(cls, 'transform')),
                             getattr(cls, '__parallel_round_contract__', None))


def constructor_state(cls: type) -> tuple[tuple[type, dict[str, object]], ...]:
    states: list[tuple[type, dict[str, object]]] = []
    for base in cls.__mro__:
        values = {name: value for name, value in vars(base).items() if not name.startswith('__') and not callable(value) and not isdatadescriptor(value) and not ismethoddescriptor(value)}
        if base.__init__ is not cls.__init__ and (isfunction(base.__init__) or ismethod(base.__init__)):
            values['__init__'] = base.__init__
        if values:
            states.append((base, values))
    return tuple(states)


@identity_snapshot()
def validate_constructor(cls: type, parents: tuple[tuple[type, dict[str, object]], ...], fingerprint: Callable[[object], str]) -> None:
    if fingerprint(constructor_state(cls)) != fingerprint(parents):
        raise TypeError('Parallel constructor has unfrozen inherited initializer state or class state')


@dataclass(frozen=True)
class ScalerFactory:
    transform_class: object
    static_params: dict[str, object]
    dynamic_selector: str | None
    transforms: dict[type, CapturedTransform]

    def selected(self, params: Mapping[str, object]) -> object:
        return self.transform_class if self.dynamic_selector is None else cast(dict[str, object], self.transform_class)[cast(str, params['scaler_type'])]

    def variants(self) -> tuple[object, ...]:
        return (self.transform_class,) if self.dynamic_selector is None else tuple(cast(dict[str, object], self.transform_class).values())

    def owns_transform(self, value: object) -> bool:
        return getattr(value, '__self__', None) is self and getattr(getattr(value, '__func__', None), '__code__', None) is ScalerFactory.transform.__code__

    def __call__(self, data: pl.DataFrame, **params: object) -> object:
        extras = {key: value for key, value in params.items() if self.dynamic_selector is None or key != 'scaler_type'}
        return cast(Callable[..., object], self.selected(params))(data, **self.static_params, **extras)

    def transform(self, data: pl.DataFrame, fitted_transform: object) -> pl.DataFrame:
        for cls in type(fitted_transform).__mro__:
            if cls in self.transforms:
                return self.transforms[cls](fitted_transform, data)
        raise TypeError('Parallel scaler allocated an uncaptured transform class')


attach(ScalerFactory)
attach(ScalerFactory.transform)


def scaler_factory(cls: object, params: dict[str, object], selector: str | None,
                   constructor: Callable[[object], object]) -> ScalerFactory:
    classes = {'': cast(type, cls)} if selector is None else SCALER_REGISTRY
    transforms = {value: capture_transform(value) for value in classes.values()}
    return ScalerFactory(constructor(cls) if selector is None else {name: constructor(value) for name, value in classes.items()}, params, selector, transforms)
