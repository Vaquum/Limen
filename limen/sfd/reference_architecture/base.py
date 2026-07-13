from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.log._permutation_confusion_metrics import confusion_mean_return_pct


class ReferenceModel(ABC):

    '''Base class for class-based reference architecture models.'''

    deterministic: bool = False

    def __init__(self) -> None:

        super().__init__()

        self.model = None

    @abstractmethod
    def train(self, data: dict[str, Any], **params: Any) -> 'ReferenceModel':

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
    def predict(self, data: dict[str, Any]) -> dict[str, Any]:

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
    def evaluate(self, data: dict[str, Any], inline_metrics: bool = True) -> dict[str, Any]:

        '''
        Evaluate the trained model and return results.

        Args:
            data (dict): Data dictionary with x_test, y_test, and optionally price_data_for_backtest
            inline_metrics (bool): Whether to include confusion_* and backtest_* prefixed keys

        Returns:
            dict: Metrics dict, optionally with flattened confusion_* and backtest_* keys
        '''

        ...

    def _compute_confusion(self,
                           preds: npt.NDArray[np.integer[Any] | np.floating[Any]],
                           y_test: npt.NDArray[np.integer[Any] | np.floating[Any]],
                           price_data_for_backtest: Any | None = None) -> dict[str, float]:

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

        results: dict[str, float] = {
            'confusion_tp': tp,
            'confusion_fp': fp,
            'confusion_tn': tn,
            'confusion_fn': fn,
            'confusion_precision': precision,
            'confusion_recall': recall,
        }

        if price_data_for_backtest is None:
            return results

        open_arr = price_data_for_backtest['open'].to_numpy()
        close_arr = price_data_for_backtest['close'].to_numpy()

        confusion_return_pct = confusion_mean_return_pct(
            preds,
            y_test,
            open_arr,
            close_arr - open_arr,
        )
        results.update({f'confusion_{k}': v for k, v in confusion_return_pct.items()})

        return results

    def _cost_kwargs(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            key: data[f"backtest_{key}"]
            for key in ('fee_bps', 'slip_bps', 'notional_rate')
            if f"backtest_{key}" in data
        }

    def _compute_backtest(self,
                          preds: npt.NDArray[np.integer[Any] | np.floating[Any]],
                          data: dict[str, Any]) -> dict[str, Any]:

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

        price = data['price_data_for_backtest']
        open_arr = price['open'].to_numpy()
        close_arr = price['close'].to_numpy()

        bt_columns = {
            'predictions': np.asarray(preds).astype(int),
            'open': open_arr,
            'close': close_arr,
            'price_change': close_arr - open_arr,
        }

        bt_result = backtest_snapshot(bt_columns, execution_lag_bars=1, **self._cost_kwargs(data))

        return {f"backtest_{k}": v for k, v in bt_result.items()}
