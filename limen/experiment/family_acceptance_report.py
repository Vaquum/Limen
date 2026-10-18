import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

import limen.experiment._family_evidence as evidence
from limen.metrics import deflated_sharpe_ratio, probability_of_backtest_overfitting

__all__ = ['family_acceptance_report']


def _assessment(trials: list[evidence.Trial], n_blocks: int) -> dict[str, object]:
    values = np.stack([trial.values for trial in trials])
    errors: dict[str, str] = {}
    report: dict[str, object] = {
        'evaluation': trials[0].evaluation, 'members': [trial.identity for trial in trials],
        'evidence': trials[0].evidence,
        'n_trials': len(trials), 'n_bars': values.shape[1], 'n_blocks': n_blocks,
        'winner': None, 'winner_sharpe': None, 'trial_sharpe_variance': None,
        'deflated_sharpe_probability': None, 'pbo': None, 'errors': errors,
    }
    try:
        if values.shape[1] < 4:
            raise ValueError('DSR requires at least four recorded returns per trial')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            spread = np.std(values, axis=1, ddof=1)
            if not np.isfinite(spread).all() or np.any(spread <= 0):
                raise ValueError('DSR selection requires positive finite variance for every trial')
            sharpes = np.mean(values, axis=1) / spread
            variance = float(np.var(sharpes, ddof=1)) if len(trials) > 1 else 0.0
        winner = int(np.argmax(sharpes))
        report.update(winner=trials[winner].identity, winner_sharpe=float(sharpes[winner]), trial_sharpe_variance=variance)
        report['deflated_sharpe_probability'] = deflated_sharpe_ratio(values[winner], n_trials=len(trials), trial_sharpe_variance=variance)
    except (ValueError, FloatingPointError) as exc:
        errors['deflated_sharpe_probability'] = str(exc)
    try:
        report['pbo'] = probability_of_backtest_overfitting(values, n_blocks=n_blocks)
    except (ValueError, FloatingPointError) as exc:
        errors['pbo'] = str(exc)
    return report


def family_acceptance_report(
    directories: Sequence[Path], *, project_root: Path, n_blocks: int = 2,
) -> dict[str, object]:
    '''Assess the distinct comparable trials retained in explicitly supplied runs.

    Source artifacts are read without replay or modification. Each comparison
    uses one recorded evaluation sample; its statistics remain conditional on
    supplied successful trials and do not estimate independent trial counts.
    '''
    if not directories:
        raise ValueError('Family report requires at least one experiment directory')
    if type(n_blocks) is not int or n_blocks < 2 or n_blocks % 2:
        raise ValueError('Family PBO n_blocks must be an even integer of at least two')
    roots: dict[Path, dict[str, object]] = {}
    duplicate_inputs: list[str] = []
    for supplied in directories:
        directory = Path(supplied).resolve()
        if not directory.is_dir():
            raise ValueError(f'Experiment directory does not exist: {directory}')
        if directory in roots:
            duplicate_inputs.append(str(directory))
            continue
        path = directory / 'metadata.json'
        roots[directory] = evidence.mapping(cast(object, json.loads(path.read_text())), str(path)) if path.exists() else {}
    ordered = sorted(roots, key=lambda path: (evidence.chronology(roots[path]) is None,
                                            '' if (created := evidence.chronology(roots[path])) is None else created.isoformat(), str(path)))
    runs: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    groups: dict[str, list[evidence.Trial]] = {}
    seen: dict[str, dict[str, str]] = {}
    for directory in ordered:
        run, trials = evidence.read_run(directory, roots[directory], Path(project_root))
        runs.append(run)
        for trial in trials:
            entries.append(trial.report)
            if trial.evaluation is None:
                continue
            trial.report['comparison'] = trial.evaluation
            repeat_key = evidence.digest([trial.candidate, trial.evaluation, trial.values.tolist()])
            if repeat_key in seen:
                trial.report['repeat_of'] = seen[repeat_key]
                trial.report['repeat_reason'] = 'Same declared candidate, implementation version, evaluation evidence and net returns'
                continue
            seen[repeat_key] = trial.identity
            groups.setdefault(trial.evaluation, []).append(trial)
    return {
        'runs': runs, 'trials': entries, 'duplicate_inputs': duplicate_inputs,
        'comparisons': [_assessment(trials, n_blocks) for trials in groups.values()],
        'comparability': 'Equal resolved declared data source, ordered fold samples and recorded market evidence within one return convention. No sample intersection or replay.',
        'coverage': {'recorded_trials': len(entries), 'distinct_compared_trials': sum(map(len, groups.values())),
                     'repeats': sum(entry['repeat_of'] is not None for entry in entries),
                     'unavailable': sum(entry['unavailable'] is not None for entry in entries)},
        'limitations': [
            'Each comparison is conditional on its supplied retained successful trials; groups are not one complete family statistic.',
            'Missing/failed/unrecorded searches are not counted. Recorded chronology does not establish when a researcher saw results.',
            'Trial counts are not effective independent counts; correlation, serial dependence and changing regimes limit DSR/PBO.',
            'These reused test observations are not an untouched holdout or evidence of future profitability.',
        ],
    }
