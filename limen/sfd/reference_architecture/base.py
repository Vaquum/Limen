from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt

from limen.sfd.reference_architecture._backtest_evaluation import compute_backtest as _compute_backtest
from limen.log._permutation_confusion_metrics import confusion_mean_return_pct


class ReferenceModel(ABC):

    '''Base class for class-based reference architecture models.'''

    deterministic: bool = False
    prediction_mode: ClassVar[Literal['binary', 'target_exposure']] = 'binary'

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
        '''Return test metrics, optionally including confusion and backtest metrics.'''

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
        options = {
            key: data[f"backtest_{key}"]
            for key in ('fee_bps', 'slip_bps', 'notional_rate', 'take_profit_bps', 'stop_loss_bps')
            if f"backtest_{key}" in data
        }
        if '_trade_context' in data:
            options['_trade_context'] = data['_trade_context']
        return options

    def _record_probabilities(self, data: Mapping[str, object], prediction: Mapping[str, object]) -> None:
        if data.get('_record_model_outputs'):
            alignment = data['_alignment']
            if not isinstance(alignment, dict):
                raise TypeError('Model output recording requires an alignment mapping')
            probs = np.asarray(prediction['_probs'], dtype=float)
            if probs.ndim != 1 or probs.shape != np.asarray(prediction['_preds']).shape or not np.isfinite(probs).all():
                raise ValueError('Recorded probabilities must be finite and aligned with predictions')
            threshold = prediction.get('optimal_threshold')
            alignment['model_outputs'] = {
                'probs': probs.tolist(),
                'optimal_threshold': 0.5 if threshold is None else threshold,
                'threshold_rule': '>' if threshold is None else '>=',
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

        from limen.backtest.trade_contract import TradePolicy

        policy = data.get('_trade_policy')
        if isinstance(policy, TradePolicy) and policy.prediction_mode != self.prediction_mode:
            raise ValueError('Model output mode does not match manifest trade policy')
        return _compute_backtest(preds, data)
