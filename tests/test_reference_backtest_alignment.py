import math
from unittest.mock import patch

import numpy as np
import pandas as pd

from limen.sfd.reference_architecture import RandomBinary
from limen.sfd.reference_architecture import XGBoostRegressor
from tests.test_reference_architecture import _make_data


def _assert_snapshot_kwargs(captured: dict) -> None:
    assert captured['kwargs']['execution_lag_bars'] == 1
    assert captured['kwargs']['trades_count_mode'] == 'runs'


def test_random_binary_inline_backtest_does_not_pass_actuals_to_snapshot() -> None:

    captured = {}

    def _fake_backtest_snapshot(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        captured['df'] = df.copy()
        captured['kwargs'] = kwargs
        return pd.DataFrame([{'trade_win_rate_pct': 1.0}])

    with patch(
        'limen.sfd.reference_architecture.base.backtest_snapshot',
        _fake_backtest_snapshot,
    ):
        data = _make_data(binary=True, with_price=True)
        model = RandomBinary().train(data, random_weights=0.5)
        model.evaluate(data)

    assert 'actuals' not in captured['df'].columns
    _assert_snapshot_kwargs(captured)


def test_xgboost_inline_backtest_uses_directional_predictions_without_actuals() -> None:

    captured = {}

    def _fake_backtest_snapshot(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        captured['df'] = df.copy()
        captured['kwargs'] = kwargs
        return pd.DataFrame([{'trade_win_rate_pct': 1.0}])

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

    assert 'actuals' not in captured['df'].columns
    _assert_snapshot_kwargs(captured)


def test_compute_confusion_return_metrics_uses_directional_actuals() -> None:

    model = XGBoostRegressor()
    data = {
        'price_data_for_backtest': pd.DataFrame({
            'open': [100.0, 100.0, 100.0],
            'close': [101.0, 99.0, 102.0],
        }),
    }

    result = model._compute_confusion_return_metrics(
        np.array([1, 0, 1]),
        np.array([1, 0, 1]),
        data,
        execution_lag_bars=0,
    )

    assert result['confusion_tp_mean_return_pct'] == 1.5
    assert result['confusion_tn_mean_return_pct'] == -1.0
    assert math.isnan(result['confusion_fp_mean_return_pct'])
    assert math.isnan(result['confusion_fn_mean_return_pct'])
