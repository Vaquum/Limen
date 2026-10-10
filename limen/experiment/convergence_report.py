import csv
import sys
import tempfile
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import polars as pl
from sklearn.exceptions import ConvergenceWarning

__all__ = ['convergence_header', 'convergence_report', 'convergence_warning', 'convergence_warnings']


def convergence_header(path: Path, header: list[str]) -> list[str]:
    """Extend older result headers without inferring past warning evidence."""
    if '_convergence_warning' in header:
        return header
    updated = [*header, '_convergence_warning']
    with path.open(newline='') as source, tempfile.NamedTemporaryFile('w', dir=path.parent, newline='', delete=False) as output:
        reader = csv.reader(source)
        _ = next(reader)
        writer = csv.writer(output)
        writer.writerow(updated)
        for row in reader:
            writer.writerow([*row, ''])
    _ = Path(output.name).replace(path)
    return updated


@contextmanager
def convergence_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """Observe filtered warnings while preserving errors and external display."""
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.filters = [(action if action == 'error' else 'always', message, category, module, line)
                                  for action, message, category, module, line in warnings.filters]
            yield caught
    finally:
        for warning in caught:
            module = next((module for module in sys.modules.copy().values() if getattr(module, '__file__', None) == warning.filename), None)
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno,
                                   module=module.__name__ if module else Path(warning.filename).stem,
                                   registry=vars(module).setdefault('__warningregistry__', {}) if module else None)


def convergence_warning(caught: Sequence[warnings.WarningMessage], succeeded: bool) -> bool | None:
    """Record category evidence only for completed rounds."""
    return any(issubclass(warning.category, ConvergenceWarning) for warning in caught) if succeeded else None


def convergence_report(results: pl.DataFrame, *, parameter_columns: Sequence[str]) -> dict[str, object]:
    """Describe recorded warning frequency, without inferring missing evidence."""
    missing = [name for name in parameter_columns if name not in results.columns]
    if results.height and missing:
        raise ValueError(f'Convergence report requires declared parameter columns: {missing}')
    evidence = '_convergence_warning'
    if evidence in results.columns:
        if results.height and results.schema[evidence] != pl.Boolean and results[evidence].null_count() != results.height:
            raise ValueError('_convergence_warning must contain Boolean or null evidence')
        observed = results.filter(pl.col(evidence).is_not_null())
    else:
        observed = results.head(0)
    warnings = cast(list[bool], observed[evidence].to_list()) if observed.height else []
    warning_count = sum(warnings)
    patterns: list[dict[str, object]] = []
    for parameter in parameter_columns:
        if not observed.height:
            break
        counts: dict[str, tuple[int, int]] = {}
        values = cast(list[object], observed[parameter].to_list())
        for value, warning in zip(values, warnings, strict=True):
            display = repr(value)
            total, warned = counts.get(display, (0, 0))
            counts[display] = total + 1, warned + int(warning)
        patterns.extend({
            'parameter': parameter, 'value': value,
            'observed_rounds': total, 'convergence_warning_rounds': warned,
            'convergence_warning_pct': 100.0 * warned / total,
        } for value, (total, warned) in sorted(counts.items()) if warned)
    return {
        'rounds': results.height,
        'observed_rounds': observed.height,
        'unavailable_rounds': results.height - observed.height,
        'convergence_warning_rounds': warning_count,
        'convergence_warning_pct': 100.0 * warning_count / observed.height if observed.height else None,
        'parameter_patterns': patterns,
    }
