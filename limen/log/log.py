import limen.experiment._holdout_research as _holdout
import polars as pl
from typing import Any
from collections.abc import Callable


class Log:

    '''Log object for storing and analyzing experiment results.'''

    from limen.log._experiment_backtest_results import experiment_backtest_results
    from limen.log._experiment_confusion_metrics import experiment_confusion_metrics
    from limen.log._experiment_parameter_correlation import experiment_parameter_correlation
    from limen.log._experiment_parameter_importance import experiment_parameter_importance
    from limen.log._permutation_confusion_metrics import permutation_confusion_metrics
    from limen.log._permutation_prediction_performance import permutation_prediction_performance as _permutation_prediction_performance

    from limen.log._read_from_file import read_from_file

    def __init__(self,
                 uel_object: Any | None = None,
                 file_path: str | None = None,
                 inverse_scaler: Callable[[pl.DataFrame, Any], pl.DataFrame] | None = None,
                 cols_to_multilabel: list[str] | None = None) -> None:

        '''Read experiment results from a live loop or a log file.'''

        super().__init__()

        self.fold_results = pl.DataFrame()
        self.data = self.prep = self.scalers = self.round_params = self.preds = self._alignment = self.manifest = None

        if uel_object is not None:

            rows = uel_object.experiment_log
            if not isinstance(rows, pl.DataFrame):
                raise ValueError('Log requires captured experiment rows')
            self.experiment_log = rows.clone()
            self.data, self.prep, self.scalers, self.round_params, self.preds, self._alignment, self.manifest = (
                getattr(uel_object, name, None) for name in ('data', 'prep', 'scalers', 'round_params', 'preds', '_alignment', 'manifest'))
            self.fold_results = getattr(uel_object, 'fold_results', pl.DataFrame()).clone()

        elif file_path is not None:
            self.experiment_log = self.read_from_file(file_path)

        else:
            raise ValueError('Log Both uel_object and file_path cannot be None')

        self._captured_log = self.experiment_log.clone()

        if cols_to_multilabel is not None:
            for col in cols_to_multilabel:
                self.experiment_log = self.experiment_log.drop(col).with_columns(
                    (pl.Series(f'{col}_{value}', self.experiment_log[col] == value)
                     .fill_null(False).cast(pl.Int64))
                    for value in self.experiment_log[col].unique().drop_nulls().sort()
                )

            self.experiment_log = self.experiment_log.with_columns(pl.col(pl.Boolean).cast(pl.Int64))

        if inverse_scaler is not None:
            self.inverse_scaler = inverse_scaler
        else:
            self.inverse_scaler = None


    @_holdout.replay_guard
    def permutation_prediction_performance(self, round_id: int) -> pl.DataFrame:
        if not self.fold_results.is_empty():
            raise ValueError('Walk-forward prediction replay requires a fold-specific test set')
        if self.round_params is None or self.preds is None or self._alignment is None:
            raise ValueError('Row analysis requires retained serial evidence; run with post_processing=True')
        if self.data is None or not callable(self.prep) or not len(self.round_params) == len(self.preds) == len(self._alignment) or not -len(self.round_params) <= round_id < len(self.round_params):
            raise ValueError('Row analysis requires retained serial evidence; run with post_processing=True')
        if self.inverse_scaler is not None and (self.scalers is None or len(self.scalers) != len(self.round_params)):
            raise ValueError('Row analysis requires the fitted scaler for the retained round')
        return self._permutation_prediction_performance(round_id)


    @_holdout.replay_guard
    def _get_test_data_with_all_cols(self, round_id: int) -> pl.DataFrame:

        '''Reconstruct a retained round's test-window columns.'''

        if not self.fold_results.is_empty():
            raise ValueError('Walk-forward prediction replay requires a fold-specific test set')
        if self.data is None or self.round_params is None or self._alignment is None:
            raise ValueError('Row analysis requires retained serial evidence; run with post_processing=True')

        missing_datetimes = self._alignment[round_id]['missing_datetimes']
        first_test_datetime = self._alignment[round_id]['first_test_datetime']
        last_test_datetime = self._alignment[round_id]['last_test_datetime']

        if self.manifest is not None:
            round_params = self.round_params[round_id]
            test_bars = self.manifest.compute_test_bars(self.data, round_params)
            data_source = test_bars
        else:
            data_source = self.data

        result = data_source.with_columns(pl.col('datetime').dt.cast_time_unit('ms'))

        if missing_datetimes:
            result = result.join(
                pl.DataFrame({'datetime': missing_datetimes})
                .with_columns(pl.col('datetime').dt.cast_time_unit('ms')),
                on='datetime',
                how='anti',
            )

        return result.filter(pl.col('datetime').is_between(first_test_datetime, last_test_datetime, closed='both'))
