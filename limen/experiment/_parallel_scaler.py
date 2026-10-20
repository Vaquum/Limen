from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import Signature, getattr_static, signature
from typing import Protocol, cast

import polars as pl

from limen._parallel_contract import ParallelRoundContract, attach
from limen.scalers.registry import SCALER_REGISTRY


class TransformClass(Protocol):
    transform: Callable[..., pl.DataFrame]


@dataclass(frozen=True)
class ScalerTransform:
    """Bind captured behavior without changing the fitted object's type or state."""

    function: Callable[..., pl.DataFrame]
    bind_instance: bool
    __parallel_round_contract__: ParallelRoundContract | None

    @property
    def __signature__(self) -> Signature:
        result = signature(self.function)
        return result.replace(parameters=list(result.parameters.values())[1:]) if self.bind_instance else result

    def __call__(self, data: pl.DataFrame, fitted_transform: object) -> pl.DataFrame:
        if getattr_static(fitted_transform, 'transform') is not getattr_static(type(fitted_transform), 'transform'):
            return cast(TransformClass, fitted_transform).transform(data)
        return self.function(fitted_transform, data) if self.bind_instance else self.function(data)


@dataclass(frozen=True)
class ScalerFactory:
    transform_class: object
    static_params: dict[str, object]
    dynamic_selector: str | None
    transforms: dict[type, ScalerTransform]

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
        return self.transforms[type(fitted_transform)](data, fitted_transform)


attach(ScalerFactory)
attach(ScalerFactory.transform)


def scaler_factory(cls: object, params: dict[str, object], selector: str | None,
                   constructor: Callable[[object], object]) -> ScalerFactory:
    classes = {'': cast(type, cls)} if selector is None else SCALER_REGISTRY
    transforms = {value: ScalerTransform(cast(TransformClass, value).transform,
                  not isinstance(getattr_static(value, 'transform'), (classmethod, staticmethod)),
                  getattr(value, '__parallel_round_contract__', None)) for value in classes.values()}
    return ScalerFactory(constructor(cls) if selector is None else {name: constructor(value) for name, value in classes.items()}, params, selector, transforms)
