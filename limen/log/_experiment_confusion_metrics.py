from typing import Any

import polars as pl

from limen.log._experiment_backtest_results import captured_metrics

_COLUMNS = ['tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'tp_mean_return_pct', 'fp_mean_return_pct', 'tn_mean_return_pct', 'fn_mean_return_pct']


def experiment_confusion_metrics(self: Any, x: str | None = None) -> pl.DataFrame:
    '''Read captured confusion metrics, or retain explicit-x row analysis.'''
    if x is None:
        rows = captured_metrics(self, _COLUMNS, 'confusion_')
        if rows.is_empty():
            raise ValueError('No objects to concatenate')
        return rows
    params = getattr(self, 'round_params', None)
    if params is None or (not params and isinstance(rows := getattr(self, '_captured_log', getattr(self, 'experiment_log', None)), pl.DataFrame) and not rows.is_empty()):
        raise ValueError('Row analysis requires retained serial evidence; run with post_processing=True')
    all_rows = [self.permutation_confusion_metrics(x=x, round_id=index, id_cols=parameters)
                for index, parameters in enumerate(params)]
    if not all_rows:
        raise ValueError('No objects to concatenate')
    return pl.concat(all_rows, how='diagonal_relaxed')
