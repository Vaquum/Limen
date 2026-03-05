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

    '''

    def __init__(self,
                 *,
                 metric: str,
                 nan_threshold: float = 0.1,
                 min_observations: int = 1,
                 active: bool = True) -> None:

        '''
        Initialize the SanityReducer.

        Args:
            metric (str): Name of the target metric column to check for NaN
            nan_threshold (float): NaN rate above which a value is pruned
                (strict >)
            min_observations (int): Minimum trial count before a value can
                be considered for pruning
            active (bool): Whether this reducer is enabled

        '''

        super().__init__(active=active)
        self._metric = metric
        self._nan_threshold = nan_threshold
        self._min_observations = min_observations
        self._removed: set[tuple[str, Any]] = set()


    def analyze_and_intervene(self,
                              log: pl.DataFrame,
                              msq: Any) -> list[dict[str, Any]]:

        '''
        Analyze experiment log and return remove_is interventions for
        parameter values with NaN rate exceeding the threshold.

        '''

        if not self._active:
            return []

        df = log

        if df.is_empty() or self._metric not in df.columns:
            return []

        param_names = msq._domain.keys
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


    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        return {'removed': list(self._removed)}


    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        self._removed = set(tuple(x) for x in state['removed'])
