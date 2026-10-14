from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from inspect import isfunction, ismethod
import math
from typing import cast

import polars as pl

from limen.backtest.trade_contract import JsonValue
from limen.experiment._prepare_trade_context import load_source as _load, source_settings as _source_settings
from limen.experiment._resolve_trade_policy import BacktestConfig

__all__ = ['fixed_trade_sources']


def _fixed_params(value: object, names: Collection[str]) -> bool:
    if isinstance(value, str):
        return value not in names and '{' not in value and '}' not in value
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _fixed_params(item, names)
                   for key, item in cast(Mapping[object, object], value).items())
    if isinstance(value, (tuple, list)):
        return all(_fixed_params(item, names) for item in cast(tuple[object, ...] | list[object], value))
    return value is None or isinstance(value, (bool, int)) or (isinstance(value, float) and math.isfinite(value))


def _same_sources(config: BacktestConfig, params: Mapping[str, object],
                  expected: tuple[tuple[Callable[..., object], JsonValue] | None, ...]) -> bool:
    return all((actual is None and saved is None) or (actual is not None and saved is not None
               and getattr(actual[0], '__func__', actual[0]) is getattr(saved[0], '__func__', saved[0])
               and actual[1] == saved[1])
               for actual, saved in zip(_source_settings(config, params), expected, strict=True))


@contextmanager
def fixed_trade_sources(manifest: object, parameter_names: Collection[str]) -> Iterator[None]:
    config = getattr(manifest, 'backtest_config', None)
    if not isinstance(config, BacktestConfig):
        yield
        return
    sources = (config.execution_data_source, config.funding.data_source if config.funding is not None else None)
    if not any(source is not None for source in sources) or not all(
            source is None or ((isfunction(source.method) or ismethod(source.method))
                               and _fixed_params(source.params, parameter_names)) for source in sources
    ) or '_frozen_trade_sources' in vars(manifest):
        yield
        return
    settings = _source_settings(config, {})
    frames: tuple[pl.DataFrame | None, pl.DataFrame | None] | None = None

    def snapshot(current: BacktestConfig, params: Mapping[str, object]) -> tuple[pl.DataFrame | None, pl.DataFrame | None] | None:
        nonlocal frames
        if vars(manifest).get('_frozen_trade_sources') is not snapshot:
            return None
        live = getattr(manifest, 'backtest_config', None)
        if not isinstance(live, BacktestConfig) or _source_settings(live, params) != settings or not _same_sources(current, params, settings):
            raise ValueError('Fixed trade source declaration changed during this run')
        if frames is None:
            frames = (None if sources[0] is None else _load(sources[0], params).clone(),
                      None if sources[1] is None else _load(sources[1], params).clone())
        return frames

    vars(manifest)['_frozen_trade_sources'] = snapshot
    try:
        yield
    finally:
        frames = None
        del vars(manifest)['_frozen_trade_sources']
