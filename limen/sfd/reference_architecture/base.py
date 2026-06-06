from abc import ABC
from abc import abstractmethod
import math
from typing import Any

import numpy as np
import pandas as pd

from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.log._permutation_confusion_metrics import _confusion_mean_return_pct


DEFAULT_FEE_BPS = 5.0
DEFAULT_SLIP_BPS = 5.0


class ReferenceModel(ABC):

    '''Base class for class-based reference architecture models.'''

    deterministic: bool = False

    def __init__(self) -> None:

        self.model = None
        self.fee_bps = DEFAULT_FEE_BPS
        self.slip_bps = DEFAULT_SLIP_BPS

    @property
    def fee_bps(self) -> float:
        return self._fee_bps

    @fee_bps.setter
    def fee_bps(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError('ReferenceModel fee_bps must be a non-negative finite number')
        self._fee_bps = value

    @property
    def slip_bps(self) -> float:
        return self._slip_bps

    @slip_bps.setter
    def slip_bps(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError('ReferenceModel slip_bps must be a non-negative finite number')
        self._slip_bps = value

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

    def _price_data_to_pandas(self, price_data_for_backtest: Any) -> pd.DataFrame:
        return price_data_for_backtest.to_pandas()

    def _compute_confusion(self,
                           preds: np.ndarray,
                           y_test: np.ndarray,
                           price_data_for_backtest: Any | None = None) -> dict:

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

        results = {
            'confusion_tp': tp,
            'confusion_fp': fp,
            'confusion_tn': tn,
            'confusion_fn': fn,
            'confusion_precision': precision,
            'confusion_recall': recall,
        }

        if price_data_for_backtest is None:
            raise ValueError('ReferenceModel price_data_for_backtest is required for inline confusion mean return metrics')

        price_pd = self._price_data_to_pandas(price_data_for_backtest)

        confusion_return_pct = _confusion_mean_return_pct(
            preds,
            y_test,
            price_pd['open'],
            price_pd['close'] - price_pd['open'],
        )
        results.update({f'confusion_{k}': v for k, v in confusion_return_pct.items()})

        return results

    def _compute_backtest(self, preds: np.ndarray, data: dict) -> dict:

        '''
        Compute backtest metrics if price_data_for_backtest is available.

        Args:
            preds (np.ndarray): Binary predictions (0 or 1)
            data (dict): Data dictionary, optionally containing 'price_data_for_backtest'

        Returns:
            dict: Backtest metrics with 'backtest_' prefix, or empty dict if no price data
        '''

        if 'price_data_for_backtest' not in data:
            return {}

        price_pd = self._price_data_to_pandas(data['price_data_for_backtest'])

        bt_input_data = {
            'predictions': np.asarray(preds).astype(int),
            'open': price_pd['open'].values,
            'close': price_pd['close'].values,
            'price_change': (price_pd['close'] - price_pd['open']).values,
        }
        if 'datetime' in price_pd:
            bt_input_data['datetime'] = price_pd['datetime'].values

        bt_input = pd.DataFrame(bt_input_data)

        bt_result = backtest_snapshot(
            bt_input,
            execution_lag_bars=1,
            fee_bps=self.fee_bps,
            slip_bps=self.slip_bps,
        )

        if bt_result.empty:
            return {}

        return {
            f"backtest_{k}": v
            for k, v in bt_result.iloc[0].to_dict().items()
        }
