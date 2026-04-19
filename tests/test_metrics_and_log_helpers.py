import numpy as np
import polars as pl
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.metrics import accuracy_score, precision_score, recall_score

from limen.log._experiment_backtest_results import _experiment_backtest_results
from limen.log._permutation_prediction_performance import _permutation_prediction_performance
from limen.metrics.balanced_metric import balanced_metric
from limen.metrics.multiclass_metrics import multiclass_metrics
from limen.metrics.safe_ovr_auc import safe_ovr_auc
from limen.utils.reporting import format_report_footer
from limen.utils.reporting import format_report_header
from limen.utils.reporting import format_report_section


class _DummyPerfWithInverseScaler:

    def __init__(self) -> None:
        self.data = pl.DataFrame({'feature': [1.0, 2.0]})
        self.round_params = {0: {'alpha': 7}}
        self.preds = {0: [1, 0]}
        self.scalers = {0: 'scale-0'}
        self.inverse_scaler = self._inverse_scaler
        self.prep_called_with_params = False

    def prep(self, data: pl.DataFrame, params: dict) -> dict:
        assert data.equals(self.data)
        assert params == {'alpha': 7}
        self.prep_called_with_params = True
        return {
            'x_test': pl.DataFrame({'scaled_feature': [0.1, 0.2]}),
            'y_test': [1, 1],
        }

    def _inverse_scaler(self, x_test: pl.DataFrame, scaler: str) -> pl.DataFrame:
        assert x_test['scaled_feature'].to_list() == [0.1, 0.2]
        assert scaler == 'scale-0'
        return pl.DataFrame({'restored_feature': [10.0, 20.0]})

    def _get_test_data_with_all_cols(self, round_id: int) -> pl.DataFrame:
        assert round_id == 0
        return pl.DataFrame({'open': [100.0, 50.0], 'close': [110.0, 45.0]})


class _DummyPerfWithoutInverseScaler:

    def __init__(self) -> None:
        self.data = pl.DataFrame({'feature': [1.0, 2.0]})
        self.round_params = {0: {'alpha': 3}}
        self.preds = {0: [0, 1]}
        self.scalers = {0: 'unused'}
        self.inverse_scaler = None
        self.prep_called_without_params = False

    def prep(self, data: pl.DataFrame) -> dict:
        assert data.equals(self.data)
        self.prep_called_without_params = True
        return {
            'x_test': pl.DataFrame({'scaled_feature': [0.3, 0.4]}),
            'y_test': [0, 1],
        }

    def _get_test_data_with_all_cols(self, round_id: int) -> pl.DataFrame:
        assert round_id == 0
        return pl.DataFrame({'open': [1.0, 2.0], 'close': [1.5, 1.5]})


def test_permutation_prediction_performance_preserves_inverse_scaled_features() -> None:
    perf = _permutation_prediction_performance(_DummyPerfWithInverseScaler(), round_id=0)

    assert perf.columns.tolist() == [
        'restored_feature',
        'predictions',
        'actuals',
        'hit',
        'miss',
        'open',
        'close',
        'price_change',
    ]
    assert perf['restored_feature'].tolist() == [10.0, 20.0]
    assert perf['hit'].tolist() == [True, False]
    assert perf['miss'].tolist() == [False, True]
    assert perf['price_change'].tolist() == [10.0, -5.0]


def test_permutation_prediction_performance_falls_back_to_single_argument_prep() -> None:
    dummy = _DummyPerfWithoutInverseScaler()

    perf = _permutation_prediction_performance(dummy, round_id=0)

    assert dummy.prep_called_without_params is True
    assert perf.columns.tolist() == ['predictions', 'actuals', 'hit', 'miss', 'open', 'close', 'price_change']
    assert perf['predictions'].tolist() == [0, 1]
    assert perf['actuals'].tolist() == [0, 1]
    assert perf['price_change'].tolist() == [0.5, -0.5]


def test_experiment_backtest_results_directionalizes_regression_rounds() -> None:

    captured = {}

    class _DummyRegressionBacktestLog:

        def __init__(self) -> None:
            self.round_params = {0: {'alpha': 1}}

        def permutation_prediction_performance(self, round_id: int) -> pd.DataFrame:
            assert round_id == 0
            return pd.DataFrame({
                'predictions': [0.4, -0.2, 1.7],
                'actuals': [0.1, -0.3, 0.0],
                'open': [100.0, 101.0, 102.0],
                'close': [101.0, 100.0, 103.0],
                'price_change': [1.0, -1.0, 1.0],
            })

    def _fake_backtest_snapshot(df: pd.DataFrame, **kwargs) -> pd.DataFrame:

        captured['df'] = df.copy()
        captured['kwargs'] = kwargs
        return pd.DataFrame([{'tp_mean_return_pct': 1.0}])

    with patch(
        'limen.log._experiment_backtest_results.backtest_snapshot',
        _fake_backtest_snapshot,
    ):
        result = _experiment_backtest_results(
            _DummyRegressionBacktestLog(),
            disable_progress_bar=True,
        )

    assert result.iloc[0]['tp_mean_return_pct'] == 1.0
    assert captured['df']['predictions'].tolist() == [1, 0, 1]
    assert captured['df']['actuals'].tolist() == [1, 0, 0]
    assert captured['kwargs']['execution_lag_bars'] == 1
    assert captured['kwargs']['trades_count_mode'] == 'runs'


def test_multiclass_metrics_returns_expected_rounded_summary() -> None:
    data = {'y_test': [0, 1, 2, 0]}
    preds = [0, 2, 1, 0]
    probs = np.asarray([
        [0.90, 0.05, 0.05],
        [0.10, 0.20, 0.70],
        [0.10, 0.80, 0.10],
        [0.70, 0.20, 0.10],
    ])

    result = multiclass_metrics(data, preds, probs)

    expected = {
        'precision': round(precision_score(data['y_test'], preds, average='macro'), 3),
        'recall': round(recall_score(data['y_test'], preds, average='macro'), 3),
        'auc': round(safe_ovr_auc(data['y_test'], probs), 3),
        'accuracy': round(accuracy_score(data['y_test'], preds), 3),
    }

    assert result == expected


def test_balanced_metric_returns_zero_without_positive_predictions() -> None:
    assert balanced_metric([1, 0, 1, 0], [0, 0, 0, 0]) == 0.0


def test_balanced_metric_penalizes_sparse_precision_by_trade_rate() -> None:
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]

    expected = precision_score(y_true, y_pred, zero_division=0) * np.sqrt(0.5)

    assert balanced_metric(y_true, y_pred) == pytest.approx(expected)


def test_reporting_helpers_honor_requested_width_and_titles() -> None:
    assert format_report_header('Summary', width=6) == '\n======\nSummary\n======'
    assert format_report_section('Stats', width=4) == '\n----\nStats\n----'
    assert format_report_footer(width=5) == '====='
