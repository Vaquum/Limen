import polars as pl
import wrangle
from typing import Optional, Callable, List, Any


class Log:

    '''Log object for storing and analyzing experiment results.'''

    from loop.log._experiment_backtest_results import _experiment_backtest_results as experiment_backtest_results
    from loop.log._experiment_confusion_metrics import _experiment_confusion_metrics as experiment_confusion_metrics
    from loop.log._experiment_parameter_correlation import _experiment_parameter_correlation as experiment_parameter_correlation
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
            self.experiment_log = uel_object.experiment_log.to_pandas()
            self.scalers = uel_object.scalers
            self.round_params = uel_object.round_params
            self.preds = uel_object.preds
            self._alignment = uel_object._alignment

            # Handle both manifest-based and legacy approaches
            if hasattr(uel_object, 'use_manifest') and uel_object.use_manifest:
                self.manifest = uel_object.manifest
                self.prep = None  # No prep function in manifest approach
                self.use_manifest = True
            else:
                self.prep = uel_object.prep
                self.use_manifest = False
                if hasattr(uel_object, 'manifest'):
                    self.manifest = uel_object.manifest
                else:
                    self.manifest = None

        elif file_path is not None:
            self.experiment_log = self.read_from_file(file_path)
            self.use_manifest = False  # File-based logs default to legacy
            self.prep = None
            self.manifest = None

        else:
            raise ValueError("Both uel_object and file_path can't be None")
        
        if cols_to_multilabel is not None:
            for col in cols_to_multilabel:
                self.experiment_log = wrangle.col_to_multilabel(data=self.experiment_log,
                                                        col=col,
                                                        extended_colname=True)
            
            for col in self.experiment_log.select_dtypes(include=bool):
                self.experiment_log[col] = self.experiment_log[col].astype(int)

        if inverse_scaler is not None:
            self.inverse_scaler = inverse_scaler
        else:
            self.inverse_scaler = None

    def _get_round_data(self, round_id: int) -> dict:
        """
        Get processed data for a specific round, handling both manifest and legacy approaches.
        
        Args:
            round_id (int): Round ID
            
        Returns:
            dict: Data dictionary with train/val/test splits
        """
        if self.use_manifest and self.manifest:
            # Manifest-based approach - re-execute the pipeline
            full_results = self.manifest.execute(self.data, self.round_params[round_id])
            
            # Extract only the data keys needed for logging
            data_keys = {'x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test', 
                        '_train_clean', '_val_clean', '_test_clean', '_alignment'}
            
            round_data = {k: v for k, v in full_results.items() if k in data_keys}
            return round_data
        elif self.prep:
            # Legacy approach
            try:
                if hasattr(self, 'manifest') and self.manifest:
                    round_data = self.prep(self.data, self.round_params[round_id], self.manifest)
                else:
                    round_data = self.prep(self.data, self.round_params[round_id])
            except TypeError:
                round_data = self.prep(self.data)
            return round_data
        else:
            raise ValueError("Neither manifest nor prep function available for data processing")

    def _get_test_data_with_all_cols(self, round_id: int) -> pl.DataFrame:
        
        '''
        Compute test-period rows with all columns.
        
        Args:
            round_id (int): Round ID
        
        Returns:
            pl.DataFrame: Klines dataset filtered down to the permutation test window
        '''

        missing_datetimes = self._alignment[round_id]['missing_datetimes']
        first_test_datetime = self._alignment[round_id]['first_test_datetime']
        last_test_datetime = self._alignment[round_id]['last_test_datetime']
    
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