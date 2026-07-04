import re
from typing import Any

import numpy as np

from limen._optional import require_optional
from limen.metrics.continuous_metrics import continuous_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

DEFAULT_KERNEL_SIZE = 25
DEFAULT_ALPHA = 1.0
WINDOW_COLUMN = 'ret_1'
SINGULAR_VALUE_FLOOR = 1e-12


class DLinearRegressor(ReferenceModel):

    '''DLinear regression model with train/evaluate interface.

    Implements canonical DLinear semantics (Zeng et al. 2023): the lookback
    window is decomposed by a centered moving average with replicate edge
    padding into trend and remainder components, each component gets its own
    linear head, and the head outputs are summed. The fit is the exact
    minimizer of the ridge-regularized DLinear MSE objective, computed in
    closed form by SVD, so training is deterministic and needs no seed.

    The lookback window is read from the feature columns named
    '{WINDOW_COLUMN}_lag_{i}', ordered by lag descending so each row is the
    window in time order (oldest bar first). All other feature columns are
    ignored.
    '''

    deterministic = True

    def _window_columns(self, columns: list[str]) -> list[str]:

        '''
        Select and order the lookback window columns.

        Args:
            columns (list[str]): Feature column names to search

        Returns:
            list[str]: Window columns ordered by lag descending (oldest first)
        '''

        pattern = re.compile(rf'^{re.escape(WINDOW_COLUMN)}_lag_(\d+)$')
        matches = [(int(m.group(1)), c) for c in columns if (m := pattern.match(c))]

        if not matches:
            raise ValueError(
                f'DLinearRegressor found no lookback window columns matching {WINDOW_COLUMN}_lag_{{i}}'
            )

        return [c for _, c in sorted(matches, reverse=True)]

    def _decompose(self, x: np.ndarray, kernel_size: int) -> np.ndarray:

        '''
        Decompose windows into stacked remainder and trend components.

        Args:
            x (np.ndarray): Row-wise lookback windows in time order
            kernel_size (int): Odd moving-average kernel size

        Returns:
            np.ndarray: Horizontally stacked [remainder, trend] design matrix
        '''

        ndimage = require_optional('scipy.ndimage', 'SciPy', 'stats')

        trend = ndimage.uniform_filter1d(x, size=kernel_size, axis=1, mode='nearest')

        return np.hstack([x - trend, trend])

    def train(self, data: dict, **params: Any) -> 'DLinearRegressor':

        '''
        Fit the DLinear heads in closed form on provided data.

        Args:
            data (dict): Data dictionary with x_train (polars DataFrame holding
                the lookback window columns) and y_train
            **params: kernel_size (odd int) and alpha (non-negative float)

        Returns:
            DLinearRegressor: Self with fitted weights stored
        '''

        kernel_size = params.get('kernel_size', DEFAULT_KERNEL_SIZE)
        alpha = params.get('alpha', DEFAULT_ALPHA)

        if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('DLinearRegressor kernel_size must be a positive odd integer')

        if alpha < 0:
            raise ValueError('DLinearRegressor alpha must be non-negative')

        self.cols = self._window_columns(list(data['x_train'].columns))
        self.kernel_size = kernel_size

        x = data['x_train'].select(self.cols).to_numpy().astype(np.float64)
        y = np.asarray(data['y_train'], dtype=np.float64)

        z = self._decompose(x, kernel_size)
        self.z_mean = z.mean(axis=0)
        self.y_mean = y.mean()

        u, s, vt = np.linalg.svd(z - self.z_mean, full_matrices=False)
        d = np.zeros_like(s)
        kept = s > SINGULAR_VALUE_FLOOR
        d[kept] = s[kept] / (s[kept] * s[kept] + alpha)
        self.w = vt.T @ (d * (u.T @ (y - self.y_mean)))
        self.model = self.w

        return self

    def predict(self, data: dict) -> dict:

        '''
        Compute continuous predictions from feature data.

        Args:
            data (dict): Data dictionary with x_test holding the window columns

        Returns:
            dict: Prediction results with '_preds' key
        '''

        x = data['x_test'].select(self.cols).to_numpy().astype(np.float64)
        z = self._decompose(x, self.kernel_size)

        preds = (z - self.z_mean) @ self.w + self.y_mean

        return {'_preds': preds}

    def evaluate(self, data: dict, inline_metrics: bool = True) -> dict:

        '''
        Evaluate trained model on test data.

        Args:
            data (dict): Data dictionary with x_test, y_test, and optionally price_data_for_backtest
            inline_metrics (bool): Whether to include confusion_* and backtest_* keys

        Returns:
            dict: Metrics dict, optionally with flattened confusion_* and backtest_* keys
        '''

        preds = self.predict(data)['_preds']

        results = continuous_metrics(data, preds)
        results['_preds'] = preds

        if inline_metrics:
            y_test = np.asarray(data['y_test'])
            pred_direction = (preds > 0).astype(int)
            actual_direction = (y_test > 0).astype(int)
            results.update(self._compute_confusion(pred_direction, actual_direction, data.get('price_data_for_backtest')))
            results.update(self._compute_backtest(pred_direction, data))

        return results


def dlinear_regressor(data: dict,
                      kernel_size: int = DEFAULT_KERNEL_SIZE,
                      alpha: float = DEFAULT_ALPHA) -> dict:

    '''
    Compute DLinear regression predictions and evaluation metrics.

    Args:
        data (dict): Data dictionary with x_train, y_train, x_test, y_test
        kernel_size (int): Odd moving-average kernel size for the decomposition
        alpha (float): Ridge regularization strength on the component heads

    Returns:
        dict: Results with continuous metrics, predictions, inline confusion metrics,
            and backtest metrics when price_data_for_backtest is in data
    '''

    model = DLinearRegressor().train(
        data,
        kernel_size=kernel_size,
        alpha=alpha,
    )

    result = model.evaluate(data, inline_metrics=True)
    result['_model'] = model
    return result
