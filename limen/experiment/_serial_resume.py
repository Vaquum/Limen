import csv
import json
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import cast

import polars as pl

import limen.experiment._objective_run as _objective
import limen.experiment._round_storage as _storage

__all__: list[str] = []


def restore_log(csv_path: Path, path: Path, count: int,
                domain: Mapping[str, Sequence[object]], context: Mapping[str, object] | None,
                *, factorize: bool) -> tuple[pl.DataFrame, list[tuple[int, str]]]:
    """Restore successful occurrences from records and failed cells from scalar type hints."""
    expected = _storage.csv_prefix(path, count)
    _storage.validate_prefix(path, expected)
    context = context or {}
    recorded: dict[int, Mapping[str, object]] = {}
    if expected and not factorize:
        with _storage.open_text(path, limit=len(expected)) as lines:
            for (index, _), line in zip(expected, lines, strict=True):
                entry = cast(Mapping[str, object], json.loads(line))
                parameters = entry.get('round_params', {})
                if isinstance(parameters, Mapping):
                    recorded[index] = cast(Mapping[str, object], parameters)
    hints_by_key = dict(domain) if factorize else dict(domain) | {key: [value] for key, value in context.items()}
    string_keys = {key for key, hints in hints_by_key.items() if (factorize or not key.startswith('_')) and (hints or factorize) and all(isinstance(value, str) for value in hints)}
    string_keys.update(key for parameters in recorded.values() for key, value in parameters.items() if isinstance(value, str) and not key.startswith('_'))
    keys = set(domain) | set(context) | {key for parameters in recorded.values() for key in parameters}
    overrides: dict[str, type[pl.DataType]] = {'_convergence_warning': pl.Boolean, **dict.fromkeys(string_keys, pl.String)}
    log = _objective.restored_log(csv_path, count, overrides, keys)
    if '_convergence_warning' not in log.columns or log['_convergence_warning'].null_count() == log.height:
        log = log.with_columns(pl.lit(None, dtype=pl.Boolean).alias('_convergence_warning'))
    if factorize:
        return log, expected
    with csv_path.open(newline='') as stream:
        rows = list(islice(csv.DictReader(stream), log.height))
    restored: list[pl.Series] = []
    for key in sorted(keys & set(log.columns)):
        if key.startswith('_'):
            continue
        typed = [parameters[key] for parameters in recorded.values() if key in parameters]
        if any(type(value) not in (str, bool, int, float, type(None)) for value in typed):
            continue
        values = log[key].to_list()
        hints = hints_by_key.get(key, ())
        if hints and all(isinstance(value, str) for value in hints):
            values = [row[key] for row in rows]
        elif any(type(value) is float for value in hints) and all(value is None or type(value) is float for value in hints):
            values = [float(value) if type(value) in (int, float) else value for value in values]
        for index, parameters in recorded.items():
            if key in parameters:
                values[index] = parameters[key]
        restored.append(pl.Series(key, values, strict=False))
    return log.with_columns(restored), expected
