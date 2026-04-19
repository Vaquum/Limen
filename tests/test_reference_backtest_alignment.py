from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from limen.sfd.reference_architecture import LogRegBinary
from limen.sfd.reference_architecture import RandomBinary
from limen.sfd.reference_architecture import XGBoostRegressor
from tests.test_reference_architecture import _make_data


def _assert_snapshot_kwargs(captured: dict) -> None:
    assert captured['kwargs']['execution_lag_bars'] == 1
    assert captured['kwargs']['trades_count_mode'] == 'runs'


def test_random_binary_inline_backtest_passes_binary_actuals_to_snapshot() -> None:

    captured = {}

    def _fake_backtest_snapshot(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        captured['df'] = df.copy()
        captured['kwargs'] = kwargs
        return pd.DataFrame([{'tp_mean_return_pct': 1.0}])

    with patch(
        'limen.sfd.reference_architecture.base.backtest_snapshot',
        _fake_backtest_snapshot,
    ):
        data = _make_data(binary=True, with_price=True)
        model = RandomBinary().train(data, random_weights=0.5)
        model.evaluate(data)

    assert 'actuals' in captured['df'].columns
    np.testing.assert_array_equal(
        captured['df']['actuals'].to_numpy(),
        np.asarray(data['y_test']).astype(int),
    )
    _assert_snapshot_kwargs(captured)


def test_xgboost_inline_backtest_passes_directional_actuals_and_kwargs() -> None:

    captured = {}

    def _fake_backtest_snapshot(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        captured['df'] = df.copy()
        captured['kwargs'] = kwargs
        return pd.DataFrame([{'tp_mean_return_pct': 1.0}])

    with patch(
        'limen.sfd.reference_architecture.base.backtest_snapshot',
        _fake_backtest_snapshot,
    ):
        data = _make_data(with_price=True)
        model = XGBoostRegressor().train(
            data,
            learning_rate=0.1,
            n_estimators=10,
            random_state=42,
        )
        model.evaluate(data)

    assert 'actuals' in captured['df'].columns
    np.testing.assert_array_equal(
        captured['df']['actuals'].to_numpy(),
        (np.asarray(data['y_test']) > 0).astype(int),
    )
    _assert_snapshot_kwargs(captured)


def test_compute_backtest_rejects_nan_actuals() -> None:

    model = LogRegBinary()
    data = {
        'price_data_for_backtest': pd.DataFrame({
            'open': [100.0, 101.0],
            'close': [110.0, 111.0],
        }),
    }

    with pytest.raises(ValueError, match='actuals must be numeric and contain no NaN values'):
        model._compute_backtest(
            np.array([1, 0]),
            data,
            actuals=np.array([1.0, np.nan]),
        )
