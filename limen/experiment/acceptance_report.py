import limen.experiment._round_storage as _storage
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.metrics import deflated_sharpe_ratio, probability_of_backtest_overfitting
from limen.experiment._recording import validate_recorded_pbo
from limen.experiment._acceptance_settings import read_acceptance as read_acceptance, report_settings, partition_geometry, require_partition

__all__ = ['acceptance_report', 'read_acceptance']

_MINIMUM_DSR_BARS = 4


def _matrix(frame: pl.DataFrame, trials: list[str]) -> npt.NDArray[np.float64]:
    tracks: list[npt.NDArray[np.float64]] = []
    for trial in trials:
        rows = frame.filter(pl.col('trial') == trial).sort('bar')
        if rows['bar'].to_list() != list(range(rows.height)):
            raise ValueError('Acceptance requires unique contiguous per-trial bar identities')
        tracks.append(np.asarray(rows['net_return'].to_numpy(), dtype=np.float64))
    if not tracks or len({track.size for track in tracks}) != 1:
        raise ValueError('Acceptance requires equally sized non-empty per-trial return tracks')
    values = np.stack(tracks)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('Acceptance requires finite recorded execution returns')
    return values


def _require_synchronous(directory: Path, values: npt.NDArray[np.float64], trials: list[str]) -> None:
    if not (path := _storage.round_path(directory)).exists():
        raise ValueError('PBO requires recorded ordered test timestamps for every trial')
    records: dict[str, Mapping[str, object]] = {}
    with _storage.open_text(path) as stream:
        for line in stream:
            entry = cast(object, json.loads(line))
            if not isinstance(entry, Mapping):
                raise ValueError('PBO requires recorded fold evidence')
            record = cast(Mapping[str, object], entry)
            trial = record.get('round_id')
            if not isinstance(trial, str) or trial in records:
                raise ValueError('PBO requires unique recorded trial identities')
            records[trial] = record
    reference: list[list[datetime]] | None = None
    for trial, track in zip(trials, values, strict=True):
        folds = records.get(trial, {}).get('folds')
        if not isinstance(folds, list) or not folds:
            raise ValueError('PBO requires recorded folds for every trial')
        identities: list[list[datetime]] = []
        returns: list[float] = []
        for fold in cast(list[object], folds):
            if not isinstance(fold, Mapping):
                raise ValueError('PBO requires recorded fold evidence')
            evidence = cast(Mapping[str, object], fold)
            alignment = evidence.get('alignment')
            dates = cast(Mapping[str, object], alignment).get('test_datetimes') if isinstance(alignment, Mapping) else None
            net = evidence.get('net_returns')
            if not isinstance(dates, list) or not isinstance(net, list) or not dates or len(cast(list[object], dates)) != len(cast(list[object], net)):
                raise ValueError('PBO requires recorded ordered test timestamps aligned with each fold return')
            if not all(isinstance(value, str) for value in cast(list[object], dates)):
                raise ValueError('PBO requires recorded ISO test timestamps')
            timestamps = [datetime.fromisoformat(value) for value in cast(list[str], dates)]
            if len({value.tzinfo is None for value in timestamps}) != 1:
                raise ValueError('PBO requires consistently zoned test timestamps')
            if timestamps != sorted(set(timestamps)):
                raise ValueError('PBO requires unique increasing test timestamps')
            identities.append(timestamps)
            returns.extend(cast(list[float], net))
        try:
            recorded = np.asarray(returns, dtype=np.float64)
        except (TypeError, OverflowError) as exc:
            raise ValueError('PBO requires numeric recorded fold returns') from exc
        if not np.array_equal(recorded, track):
            raise ValueError('PBO fold returns differ from trial_returns.parquet')
        if reference is not None and identities != reference:
            raise ValueError('PBO requires identical ordered test timestamps across trials')
        reference = identities


def _score(directory: Path, values: npt.NDArray[np.float64], trials: list[str], count: int, report: dict[str, object], errors: dict[str, str]) -> None:
    try:
        if values.shape[1] < _MINIMUM_DSR_BARS:
            raise ValueError('DSR requires at least four recorded returns per trial')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            spread = np.std(values, axis=1, ddof=1)
            if not np.isfinite(spread).all() or np.any(spread <= 0):
                raise ValueError('DSR selection requires positive finite variance for every recorded trial')
            sharpes = np.mean(values, axis=1) / spread
            variance = float(np.var(sharpes, ddof=1)) if len(trials) > 1 else 0.0
        if not np.isfinite(sharpes).all():
            raise ValueError('DSR selection requires finite trial Sharpe ratios')
        winner = int(np.argmax(sharpes))
        report.update(winner_trial=trials[winner], winner_sharpe=float(sharpes[winner]), trial_sharpe_variance=variance)
        report['deflated_sharpe_probability'] = deflated_sharpe_ratio(values[winner], n_trials=len(trials), trial_sharpe_variance=variance)
    except (ValueError, FloatingPointError) as exc:
        errors['deflated_sharpe_probability'] = str(exc)
    try:
        _require_synchronous(directory, values, trials)
        require_partition(values.shape[1], count)
        report['pbo'] = probability_of_backtest_overfitting(values, n_blocks=count)
    except (ValueError, FloatingPointError) as exc:
        _storage.propagate(exc)
        errors['pbo'] = str(exc)


def acceptance_report(directory: Path, *, acceptance: Mapping[str, float] | None = None) -> dict[str, object] | None:
    if not (path := directory / 'trial_returns.parquet').exists():
        return None
    settings, count = report_settings(directory, acceptance)
    validate_recorded_pbo(directory, settings)
    thresholds = {key: value for key, value in settings.items() if key != 'pbo_blocks'}
    if (frame := pl.read_parquet(path)).schema != {'trial': pl.String, 'bar': pl.Int64, 'net_return': pl.Float64}:
        raise ValueError('Acceptance requires the recorded trial_returns.parquet schema')
    trials = cast(list[str], frame['trial'].unique(maintain_order=True).to_list())
    errors: dict[str, str] = {}
    report: dict[str, object] = {
        'n_trials': len(trials), 'n_bars': None, 'n_blocks': count, **partition_geometry(None, count),
        'winner_trial': None, 'winner_sharpe': None, 'trial_sharpe_variance': None,
        'deflated_sharpe_probability': None, 'pbo': None,
        'thresholds': thresholds, 'verdicts': {}, 'errors': errors,
    }
    try:
        values = _matrix(frame, trials)
        report.update(n_bars=values.shape[1], **partition_geometry(values.shape[1], count))
        _score(directory, values, trials, count, report, errors)
    except ValueError as exc:
        _storage.propagate(exc)
        errors['returns'] = str(exc)
    verdicts: dict[str, bool | None] = {}
    for key, threshold in thresholds.items():
        metric = 'pbo' if key == 'max_pbo' else 'deflated_sharpe_probability'
        score = report[metric]
        verdicts[key] = (score <= threshold if key == 'max_pbo' else score >= threshold) if isinstance(score, float) else None
    report['verdicts'] = verdicts
    _ = (directory / 'acceptance_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Walk-forward acceptance report', '',
             f'Recorded successful trials: {len(trials)}. CSCV blocks: {count}.', '',
             'Per-bar Sharpe; the reported winner has the largest Sharpe. Exact ties keep recorded trial order.',
             'These statistics describe the recorded sweep; they do not establish independence or future performance.',
             'Nominal partitions reuse the same n_bars/2 observations per half; blocks can cross folds with no additional purge/embargo.',
             'Finer partitions can make sparse half samples unavailable; few trials limit rank resolution. Null is not pass.', '']
    lines.extend(f'- {label}: {report[key]}' for key, label in (
        ('winner_trial', 'Winner trial'), ('winner_sharpe', 'Per-bar Sharpe'),
        ('deflated_sharpe_probability', 'Deflated Sharpe probability'), ('pbo', 'PBO'),
        ('pbo_block_bars', 'Retained bars/block'), ('pbo_combinations', 'Nominal combinations'), ('pbo_resolution', 'Nominal counting step'),
    ))
    lines.extend(f'- {key} ({thresholds[key]}): {"unavailable" if passed is None else "pass" if passed else "fail"}' for key, passed in verdicts.items())
    lines.extend(f'- {key} unavailable: {reason}' for key, reason in errors.items())
    _ = (directory / 'acceptance_report.md').write_text('\n'.join(lines) + '\n')
    return report
