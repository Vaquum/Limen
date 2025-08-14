import polars as pl
import wrangle
from typing import Optional, Callable, List, Any


class Log:

    '''Log object for storing and analyzing experiment results.'''

    from loop.log._experiment_backtest_results import _experiment_backtest_results as experiment_backtest_results
    from loop.log._experiment_confusion_metrics import _experiment_confusion_metrics as experiment_confusion_metrics
    from loop.log._experiment_feature_correlation import _experiment_feature_correlation as experiment_feature_correlation
    from loop.log._permutation_confusion_metrics import _permutation_confusion_metrics as permutation_confusion_metrics
    from loop.log._permutation_prediction_performance import _permutation_prediction_performance as permutation_prediction_performance
    
    from loop.log._read_from_file import _read_from_file as read_from_file

    def __init__(self,
                 uel_object: Optional[Any] = None,
                 file_path: Optional[str] = None,
                 inverse_scaler: Optional[Callable] = None,
                 cols_to_multilabel: Optional[List[str]] = None) -> None:
        
        '''
        Create Log object state from a UEL object or a log file.
        
        Args:
            uel_object (object, optional): Source UEL object
            file_path (str, optional): Path to the log file
            inverse_scaler (Callable, optional): Inverse scaler function
            cols_to_multilabel (list[str], optional): Columns to convert to multilabel
        '''

        if uel_object is not None:

            self.data = uel_object.data
            self.log_df = uel_object.log_df.to_pandas()
            self.prep = uel_object.prep
            self.scalers = uel_object.scalers
            self.round_params = uel_object.round_params
            self.preds = uel_object.preds
            self._alignement = uel_object._alignement

        elif file_path is not None:
            self.log_df = self._read_from_file(file_path)

        else:
            raise ValueError("Both uel_object and file_path can't be None")
        
        if cols_to_multilabel is not None:
            for col in cols_to_multilabel:
                self.log_df = wrangle.col_to_multilabel(data=self.log_df,
                                                        col=col,
                                                        extended_colname=True)
            
            for col in self.log_df.select_dtypes(include=bool):
                self.log_df[col] = self.log_df[col].astype(int)

        if inverse_scaler is not None:
            self.inverse_scaler = inverse_scaler
        else:
            self.inverse_scaler = None

    def _get_test_data_with_all_cols(self, round_id: int) -> pl.DataFrame:
        
        '''
        Compute test-period rows with all columns.
        
        Args:
            round_id (int): Round ID
        
        Returns:
            pl.DataFrame: Klines dataset filtered down to the permutation test window
        '''

        missing_datetimes = self._alignement[round_id]['missing_datetimes']
        first_test_datetime = self._alignement[round_id]['first_test_datetime']
        last_test_datetime = self._alignement[round_id]['last_test_datetime']
    
        return (
            self.data
            .with_columns(pl.col('datetime').dt.cast_time_unit('ms'))
            .join(
                pl.DataFrame({'datetime': missing_datetimes})
                .with_columns(pl.col('datetime').dt.cast_time_unit('ms')),
                on='datetime',
                how='anti',
            )
            .filter(pl.col('datetime').is_between(first_test_datetime, last_test_datetime, closed='both'))
        )
