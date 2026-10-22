import csv
import json
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import cast

import polars as pl

from limen.experiment import _round_storage as _storage

__all__: list[str] = []


def restore_parameters(log: pl.DataFrame, path: Path, expected: Sequence[tuple[int, str]],
                       domain: Mapping[str, Sequence[object]], context: Mapping[str, object] | None,
                       *, factorize: bool) -> pl.DataFrame:
    """Restore successful occurrences from records and failed cells from scalar type hints."""
    _storage.validate_prefix(path, expected)
    if factorize:
        return log
    context = context or {}
    with (path.parent / 'results.csv').open(newline='') as stream:
        rows = list(islice(csv.DictReader(stream), log.height))
    recorded: dict[int, Mapping[str, object]] = {}
    if expected:
        with _storage.open_text(path, limit=len(expected)) as lines:
            for (index, _), line in zip(expected, lines, strict=True):
                entry = cast(Mapping[str, object], json.loads(line))
                parameters = entry.get('round_params', {})
                if isinstance(parameters, Mapping):
                    recorded[index] = cast(Mapping[str, object], parameters)
    keys = set(domain) | set(context) | {key for parameters in recorded.values() for key in parameters}
    restored: list[pl.Series] = []
    for key in sorted(keys & set(log.columns)):
        if key.startswith('_'):
            continue
        typed = [parameters[key] for parameters in recorded.values() if key in parameters]
        if any(type(value) not in (str, bool, int, float, type(None)) for value in typed):
            continue
        values = log[key].to_list()
        hints = [context[key]] if key in context else domain.get(key, ())
        if hints and all(isinstance(value, str) for value in hints):
            values = [row[key] for row in rows]
        elif any(type(value) is float for value in hints) and all(value is None or type(value) is float for value in hints):
            values = [float(value) if type(value) in (int, float) else value for value in values]
        for index, parameters in recorded.items():
            if key in parameters:
                values[index] = parameters[key]
        restored.append(pl.Series(key, values, strict=False))
    return log.with_columns(restored)
