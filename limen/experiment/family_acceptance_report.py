import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

import limen.experiment._family_evidence as evidence
from limen.metrics import deflated_sharpe_ratio, probability_of_backtest_overfitting

__all__ = ['family_acceptance_report']

_MINIMUM_DSR_BARS = 4
_MINIMUM_PBO_BLOCKS = 2


def _assessment(trials: list[evidence.Trial], n_blocks: int, n_family: int,
                n_groups: int, n_ungrouped: int, coverage_known: bool) -> dict[str, object]:
    values = np.stack([trial.values for trial in trials])
    errors: dict[str, str] = {}
    report: dict[str, object] = {
        'evaluation': trials[0].evaluation, 'members': [trial.identity for trial in trials],
        'evidence': trials[0].evidence,
        'n_trials': len(trials), 'n_bars': values.shape[1], 'n_blocks': n_blocks,
        'n_family_trials': n_family, 'n_group_trials': len(trials),
        'n_trials_without_group': n_ungrouped, 'n_trials_outside_group': n_family - len(trials), 'n_groups': n_groups,
        'winner': None, 'winner_sharpe': None, 'trial_sharpe_variance': None,
        'deflated_sharpe_probability': None, 'pbo': None, 'errors': errors,
    }
    try:
        if values.shape[1] < _MINIMUM_DSR_BARS:
            raise ValueError('DSR requires at least four recorded returns per trial')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            spread = np.std(values, axis=1, ddof=1)
            invalid = [trial.identity for trial, value in zip(trials, spread, strict=True) if not np.isfinite(value) or value <= 0]
            if invalid:
                raise ValueError(f'DSR selection requires positive finite variance: {invalid}')
            sharpes = np.mean(values, axis=1) / spread
            variance = float(np.var(sharpes, ddof=1)) if len(trials) > 1 else 0.0
        winner = int(np.argmax(sharpes))
        report.update(winner=trials[winner].identity, winner_sharpe=float(sharpes[winner]), trial_sharpe_variance=variance)
        if not coverage_known or len(trials) != n_family:
            raise ValueError('DSR selection is incomplete: the group does not cover all supplied retained successful tracks')
        if len(trials) < _MINIMUM_PBO_BLOCKS:
            raise ValueError('DSR selection requires at least two retained alternatives; a singleton has no selection deflation')
        report['deflated_sharpe_probability'] = deflated_sharpe_ratio(values[winner], n_trials=len(trials), trial_sharpe_variance=variance)
    except (ValueError, FloatingPointError) as exc:
        errors['deflated_sharpe_probability'] = str(exc)
    try:
        if trials[0].evidence is not None and trials[0].evidence['pbo_unavailable'] is not None:
            raise ValueError(str(trials[0].evidence['pbo_unavailable']))
        report['pbo'] = probability_of_backtest_overfitting(values, n_blocks=n_blocks)
    except (ValueError, FloatingPointError) as exc:
        errors['pbo'] = str(exc)
    return report


def family_acceptance_report(
    directories: Sequence[Path], *, project_root: Path, n_blocks: int = 2,
) -> dict[str, object]:
    '''Assess the recorded alternatives retained in explicitly supplied runs.

    Source artifacts are read without replay or modification. Each comparison
    uses one recorded evaluation sample; its statistics remain conditional on
    supplied successful trials and do not estimate independent trial counts.
    '''
    if not directories:
        raise ValueError('Family report requires at least one experiment directory')
    if type(n_blocks) is not int or n_blocks < _MINIMUM_PBO_BLOCKS or n_blocks % 2:
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
        roots[directory] = evidence.mapping(cast(object, json.loads(path.read_text())), str(path))
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
                trial.report['evidence_match_of'] = seen[repeat_key]
            else:
                seen[repeat_key] = trial.identity
            groups.setdefault(trial.evaluation, []).append(trial)
    n_ungrouped = sum(entry['comparison'] is None for entry in entries)
    coverage_known = all(run['track_coverage_known'] for run in runs)
    return {
        'runs': runs, 'trials': entries, 'duplicate_inputs': duplicate_inputs,
        'comparisons': [_assessment(trials, n_blocks, len(entries), len(groups), n_ungrouped, coverage_known) for trials in groups.values()],
        'comparability': 'Equal ordered fold samples and retained evaluation evidence within one return convention. Declared sources are unverified context. No sample intersection or replay.',
        'coverage': {'recorded_trials': len(entries), 'distinct_compared_trials': sum(map(len, groups.values())),
                     'repeats': sum(entry['repeat_of'] is not None for entry in entries),
                     'evidence_matches': sum(entry['evidence_match_of'] is not None for entry in entries),
                     'unavailable': sum(entry['disposition'] == 'missing_evidence' for entry in entries),
                     'incompatible': sum(entry['disposition'] == 'incompatible_evidence' for entry in entries),
                     'known_failed_trials': sum(cast(int, run['n_failed_trials']) for run in runs if run['n_failed_trials'] is not None),
                     'runs_without_terminal_inventory': sum(run['n_failed_trials'] is None for run in runs)},
        'limitations': [
            'Each comparison is conditional on supplied retained successful tracks. Fragmented or incomplete pools receive no DSR; choosing a group adds uncorrected selection.',
            'Failed attempts have no return tracks; only retained CSV failures are counted. Deleted, crashed, pruned and unrecorded searches remain unknown.',
            'Run-start and logical record order do not establish global trial chronology or when a researcher saw results. Sequential adaptation is uncorrected.',
            'Evidence matches do not prove identical training inputs, fits or implementations. Cross-directory alternatives are never deduplicated.',
            'Snapshot close observations do not prove open/high/low execution identity; snapshot joint PBO is unavailable. Signed hash differences leave comparability unproven across unknown dependency environments.',
            'Two-block PBO resolves only to 0, 0.5 or 1. Declared sources and implementation versions are not verified provenance.',
            'Trial counts are not effective independent counts; correlation, serial dependence and changing regimes limit DSR/PBO.',
            'These reused test observations are not an untouched holdout or evidence of future profitability.',
        ],
    }
