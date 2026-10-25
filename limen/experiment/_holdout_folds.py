from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Concatenate, ParamSpec, TypeVar

import polars as pl

import limen.experiment._holdout_research as research
from limen.experiment.manifest_core import Manifest

__all__: list[str] = []
_T = TypeVar('_T')
_P = ParamSpec('_P')


def construction(func: Callable[Concatenate[_T, Manifest, pl.DataFrame, _P], None]) -> Callable[Concatenate[_T, Manifest, pl.DataFrame, _P], None]:
    @wraps(func)
    def guarded(host: _T, manifest: Manifest, raw: pl.DataFrame, *args: _P.args, **kwargs: _P.kwargs) -> None:
        with research.operation(manifest, 'experiment'):
            func(host, manifest, research.admit(manifest, raw), *args, **kwargs)
    return guarded
