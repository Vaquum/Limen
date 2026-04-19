from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from limen.backtest.backtest_snapshot import backtest_snapshot


class ReferenceModel(ABC):

    '''Base class for class-based reference architecture models.'''

    deterministic: bool = False

    def __init__(self) -> None:

        self.model = None

    @abstractmethod
    def train(self, data: dict, **params: Any) -> 'ReferenceModel':

        '''
        Train the model on provided data.

        Args:
            data (dict): Data dictionary with x_train, y_train, and optionally x_val, y_val
            **params: Model-specific hyperparameters

        Returns:
            ReferenceModel: Self with fitted model stored
        '''

        ...

    @abstractmethod
    def predict(self, data: dict) -> dict:

        '''
        Compute predictions from feature data.

        Args:
            data (dict): Data dictionary with x_test. Some models may
                require additional keys (e.g. x_val, y_val for threshold tuning)

        Returns:
            dict: Prediction results with '_preds' key
        '''

        ...


    @abstractmethod
    def evaluate(self, data: dict, inline_metrics: bool = True) -> dict:

        '''
        Evaluate the trained model and return results.

        Args:
            data (dict): Data dictionary with x_test, y_test, and optionally price_data_for_backtest
            inline_metrics (bool): Whether to include confusion_* and backtest_* prefixed keys

        Returns:
            dict: Metrics dict, optionally with flattened confusion_* and backtest_* keys
        '''

        ...

    def _compute_confusion(self, preds: np.ndarray, y_test: np.ndarray) -> dict:

        '''
        Compute confusion matrix metrics from binary predictions.

        Args:
            preds (np.ndarray): Binary predictions (0 or 1)
            y_test (np.ndarray): Binary true labels (0 or 1)

        Returns:
            dict: Confusion metrics with 'confusion_' prefix
        '''

        preds = np.asarray(preds).astype(int)
        y_test = np.asarray(y_test).astype(int)

        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        tn = int(((preds == 0) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())

        precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0

        return {
            'confusion_tp': tp,
            'confusion_fp': fp,
            'confusion_tn': tn,
            'confusion_fn': fn,
            'confusion_precision': precision,
            'confusion_recall': recall,
        }

    def _compute_backtest(self,
                          preds: np.ndarray,
                          data: dict,
                          actuals: np.ndarray | pd.Series | None = None) -> dict:

        '''
        Compute backtest metrics if price_data_for_backtest is available.

        Args:
            preds (np.ndarray): Binary predictions (0 or 1)
            data (dict): Data dictionary, optionally containing 'price_data_for_backtest'
            actuals (np.ndarray | pd.Series | None): Optional aligned binary actuals
                for the same prediction rows. When provided, snapshot confusion-bucket
                return metrics are populated from these labels without additional
                shifting.

        Returns:
            dict: Backtest metrics with 'backtest_' prefix, or empty dict if no price data
        '''

        if 'price_data_for_backtest' not in data:
            return {}

        price_df = data['price_data_for_backtest']

        if isinstance(price_df, pd.DataFrame):
            price_pd = price_df
        elif hasattr(price_df, 'to_pandas'):
            price_pd = price_df.to_pandas()
        else:
            return {}

        preds = np.asarray(preds).astype(int)
        if len(price_pd) != len(preds):
            raise ValueError(
                'price_data_for_backtest must align one-to-one with predictions'
            )

        bt_input = pd.DataFrame({
            'predictions': preds,
            'open': price_pd['open'].values,
            'close': price_pd['close'].values,
            'price_change': (price_pd['close'] - price_pd['open']).values,
        })

        if actuals is not None:
            actuals = pd.to_numeric(
                pd.Series(actuals),
                errors='coerce',
            ).to_numpy()
            if len(actuals) != len(preds):
                raise ValueError(
                    'actuals must align one-to-one with predictions'
                )
            if np.isnan(actuals).any():
                raise ValueError(
                    'actuals must be numeric and contain no NaN values'
                )
            actuals = actuals.astype(int)
            bt_input['actuals'] = actuals

        bt_result = backtest_snapshot(
            bt_input,
            execution_lag_bars=1,
            trades_count_mode='runs',
        )

        if bt_result.empty:
            return {}

        return {
            f"backtest_{k}": v
            for k, v in bt_result.iloc[0].to_dict().items()
        }
