from typing import Any

import polars as pl

from limen.experiment.reducer.pruning_strategy import PruningStrategy


class SanityReducer(PruningStrategy):

    '''
    Remove parameter values whose trials produce NaN in the target
    metric above a configurable threshold.

    Only considers NaN occurrences in the specified metric column.
    Values are pruned when nan_count / total > nan_threshold and
    total >= min_observations.

    Optionally emits suggestion interventions (logged but not dispatched)
    for zero-metric, execution timeout, and warning rate signals.

    '''

    def __init__(self,
                 *,
                 metric: str,
                 nan_threshold: float = 0.1,
                 min_observations: int = 1,
                 zero_metric_threshold: float | None = None,
                 execution_time_column: str = 'execution_time',
                 execution_time_threshold: float | None = None,
                 timeout_rate_threshold: float = 0.5,
                 warning_threshold: float | None = None,
                 active: bool = True) -> None:

        '''
        Initialize the SanityReducer.

        Args:
            metric (str): Name of the target metric column to check for NaN
            nan_threshold (float): NaN rate above which a value is pruned
                (strict >)
            min_observations (int): Minimum trial count before a value can
                be considered for pruning
            zero_metric_threshold (float | None): Suggest if zero-metric
                rate exceeds this (None to disable)
            execution_time_column (str): Column name for execution time
            execution_time_threshold (float | None): Suggest if execution
                time exceeds this value (None to disable)
            timeout_rate_threshold (float): Fraction of trials exceeding
                execution_time_threshold to trigger suggestion
            warning_threshold (float | None): Suggest if warning rate
                exceeds this (None to disable)
            active (bool): Whether this reducer is enabled

        '''

        super().__init__(active=active)
        self._metric = metric
        self._nan_threshold = nan_threshold
        self._min_observations = min_observations
        self._zero_metric_threshold = zero_metric_threshold
        self._execution_time_column = execution_time_column
        self._execution_time_threshold = execution_time_threshold
        self._timeout_rate_threshold = timeout_rate_threshold
        self._warning_threshold = warning_threshold
        self._removed: set[tuple[str, Any]] = set()
        self._suggested: set[tuple[str, Any]] = set()


    def analyze_and_intervene(self,
                              log: pl.DataFrame,
                              msq: Any) -> list[dict[str, Any]]:

        '''
        Analyze experiment log and return interventions.

        Returns remove_is interventions for NaN rate exceeding the
        threshold, plus suggestion interventions for zero-metric,
        execution timeout, and warning rate signals.

        '''

        if not self._active:
            return []

        df = log

        if df.is_empty() or self._metric not in df.columns:
            return []

        param_names = msq._domain.keys
        interventions = self._detect_nan(df, param_names)
        interventions.extend(self._detect_suggestions(df, param_names))
        return interventions


    def _detect_nan(self,
                    df: pl.DataFrame,
                    param_names: list[str]) -> list[dict[str, Any]]:

        '''Detect NaN metric values and return remove_is interventions.'''

        interventions: list[dict[str, Any]] = []

        for param in param_names:
            if param not in df.columns:
                continue

            stats = (
                df.group_by(param)
                .agg(
                    pl.len().alias('_total'),
                    (pl.col(self._metric).is_null() | pl.col(self._metric).is_nan()).sum().alias('_nan_count'),
                )
            )

            for row in stats.iter_rows(named=True):
                value = row[param]

                if (param, value) in self._removed:
                    continue

                if row['_total'] < self._min_observations:
                    continue

                if row['_nan_count'] / row['_total'] > self._nan_threshold:
                    self._removed.add((param, value))
                    interventions.append({
                        'op': 'remove_is',
                        'param': param,
                        'value': value,
                    })

        return interventions


    def _detect_suggestions(self,
                            df: pl.DataFrame,
                            param_names: list[str]) -> list[dict[str, Any]]:

        '''Detect soft signals and return suggestion interventions.'''

        suggestions: list[dict[str, Any]] = []

        for param in param_names:
            if param not in df.columns:
                continue

            stats = df.group_by(param).agg(pl.len().alias('_total'))

            if self._zero_metric_threshold is not None:
                suggestions.extend(
                    self._detect_zero_metric(df, param, stats)
                )

            if (self._execution_time_threshold is not None
                    and self._execution_time_column in df.columns):
                suggestions.extend(
                    self._detect_execution_timeout(df, param, stats)
                )

            if (self._warning_threshold is not None
                    and '_warnings' in df.columns):
                suggestions.extend(
                    self._detect_warnings(df, param, stats)
                )

        return suggestions


    def _detect_zero_metric(self,
                            df: pl.DataFrame,
                            param: str,
                            stats: pl.DataFrame) -> list[dict[str, Any]]:

        '''Suggest removal for parameter values with high zero-metric rate.'''

        zero_stats = (
            df.group_by(param)
            .agg((pl.col(self._metric) == 0.0).sum().alias('_zero_count'))
        )
        merged = stats.join(zero_stats, on=param)
        suggestions: list[dict[str, Any]] = []

        for row in merged.iter_rows(named=True):
            value = row[param]
            if (param, value) in self._suggested:
                continue
            if row['_total'] < self._min_observations:
                continue
            rate = row['_zero_count'] / row['_total']
            if rate > self._zero_metric_threshold:
                self._suggested.add((param, value))
                suggestions.append({
                    'op': 'remove_is',
                    'param': param,
                    'value': value,
                    'action': 'suggest',
                    'reason': f"zero_metric rate {rate:.2f} for {param}={value}",
                })

        return suggestions


    def _detect_execution_timeout(self,
                                  df: pl.DataFrame,
                                  param: str,
                                  stats: pl.DataFrame) -> list[dict[str, Any]]:

        '''Suggest removal for parameter values with high execution timeout rate.'''

        timeout_stats = (
            df.group_by(param)
            .agg(
                (pl.col(self._execution_time_column) > self._execution_time_threshold)
                .sum().alias('_timeout_count')
            )
        )
        merged = stats.join(timeout_stats, on=param)
        suggestions: list[dict[str, Any]] = []

        for row in merged.iter_rows(named=True):
            value = row[param]
            if (param, value) in self._suggested:
                continue
            if row['_total'] < self._min_observations:
                continue
            rate = row['_timeout_count'] / row['_total']
            if rate > self._timeout_rate_threshold:
                self._suggested.add((param, value))
                suggestions.append({
                    'op': 'remove_is',
                    'param': param,
                    'value': value,
                    'action': 'suggest',
                    'reason': f"execution_timeout rate {rate:.2f} for {param}={value}",
                })

        return suggestions


    def _detect_warnings(self,
                         df: pl.DataFrame,
                         param: str,
                         stats: pl.DataFrame) -> list[dict[str, Any]]:

        '''Suggest removal for parameter values with high warning rate.'''

        warning_stats = (
            df.group_by(param)
            .agg(
                (pl.col('_warnings') != '[]').sum().alias('_warning_count')
            )
        )
        merged = stats.join(warning_stats, on=param)
        suggestions: list[dict[str, Any]] = []

        for row in merged.iter_rows(named=True):
            value = row[param]
            if (param, value) in self._suggested:
                continue
            if row['_total'] < self._min_observations:
                continue
            rate = row['_warning_count'] / row['_total']
            if rate > self._warning_threshold:
                self._suggested.add((param, value))
                suggestions.append({
                    'op': 'remove_is',
                    'param': param,
                    'value': value,
                    'action': 'suggest',
                    'reason': f"warning rate {rate:.2f} for {param}={value}",
                })

        return suggestions


    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        return {
            'removed': list(self._removed),
            'suggested': list(self._suggested),
        }


    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        self._removed = set(tuple(x) for x in state['removed'])
        self._suggested = set(tuple(x) for x in state.get('suggested', []))
