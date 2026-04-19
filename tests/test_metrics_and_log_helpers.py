import numpy as np
import polars as pl
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import ClassVar

from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.log._experiment_backtest_results import _experiment_backtest_results
from limen.log._experiment_backtest_results import _prepare_snapshot_backtest_input
from limen.log._permutation_confusion_metrics import _confusion_mean_return_pct
from limen.log._permutation_confusion_metrics import _permutation_confusion_metrics
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


class _DummyConfusionMetrics:

    def permutation_prediction_performance(self, round_id: int) -> pd.DataFrame:
        assert round_id == 0
        return pd.DataFrame({
            'predictions': [1, 1, 0, 0, 0],
            'actuals': [1, 0, 0, 1, 0],
            'open': [100.0, 100.0, 100.0, 100.0, 100.0],
            'close': [150.0, 110.0, 90.0, 95.0, 105.0],
            'price_change': [50.0, 10.0, -10.0, -5.0, 5.0],
        })


class _DummyConfusionMetricsMissingPriceChange:

    def permutation_prediction_performance(self, round_id: int) -> pd.DataFrame:
        assert round_id == 0
        return pd.DataFrame({
            'predictions': [1, 1, 0, 0],
            'actuals': [1, 0, 0, 1],
            'open': [100.0, 100.0, 100.0, 100.0],
        })


class _DummyCompletedBarSignal:

    def __init__(self) -> None:
        self.data = None
        self.round_params = [{}]
        self.inverse_scaler = None
        self.scalers = {}
        # Signal uses the completed current bar and would look perfect if same-row priced.
        self.preds = [np.array([1, 0, 1, 0])]

    def prep(self, data: object, params: dict) -> dict:
        _ = data, params
        return {
            'x_test': np.array([[1.0], [0.0], [1.0], [0.0]]),
            # Next-bar direction is the opposite of the current-bar direction.
            'y_test': np.array([0, 1, 0, 1]),
        }

    def _get_test_data_with_all_cols(self, round_id: int) -> pd.DataFrame:
        assert round_id == 0
        return pd.DataFrame({
            'open': [100.0, 100.0, 100.0, 100.0],
            'close': [110.0, 90.0, 110.0, 90.0],
        })


class _DummyRegressionBacktestLog:

    round_params: ClassVar[list[dict]] = [{}]

    def permutation_prediction_performance(self, round_id: int) -> pd.DataFrame:
        assert round_id == 0
        return pd.DataFrame({
            'predictions': [0.4, -0.2, 0.7],
            'actuals': [0.6, -0.1, 0.2],
            'open': [100.0, 100.0, 100.0],
            'close': [100.0, 110.0, 90.0],
            'price_change': [0.0, 10.0, -10.0],
        })


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


def test_permutation_confusion_metrics_adds_mean_return_pct_columns() -> None:
    result = _permutation_confusion_metrics(
        _DummyConfusionMetrics(),
        x='price_change',
        round_id=0,
        outlier_quantiles=(0.0, 1.0),
    ).iloc[0]

    assert result['tp_x_mean'] == 50.0
    assert result['fp_x_mean'] == 10.0
    assert result['tp_mean_return_pct'] == 10.0
    assert result['fp_mean_return_pct'] == -10.0
    assert result['tn_mean_return_pct'] == -5.0
    assert result['fn_mean_return_pct'] == 5.0


def test_permutation_confusion_metrics_uses_positional_alignment_for_returns() -> None:
    result = _confusion_mean_return_pct(
        pd.Series([1, 1, 0, 0, 0]),
        pd.Series([1, 0, 0, 1, 0]),
        pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=[10, 11, 12, 13, 14]),
        pd.Series([50.0, 10.0, -10.0, -5.0, 5.0], index=[10, 11, 12, 13, 14]),
    )

    assert result['tp_mean_return_pct'] == 10.0
    assert result['fp_mean_return_pct'] == -10.0
    assert result['tn_mean_return_pct'] == -5.0
    assert result['fn_mean_return_pct'] == 5.0


def test_permutation_confusion_metrics_requires_return_columns() -> None:
    with pytest.raises(ValueError, match='column \"price_change\" not found'):
        _permutation_confusion_metrics(
            _DummyConfusionMetricsMissingPriceChange(),
            x='open',
            round_id=0,
            outlier_quantiles=(0.0, 1.0),
        )


def test_backtest_snapshot_adds_mean_kelly_pct() -> None:
    result = backtest_snapshot(
        pd.DataFrame({
            'predictions': [1, 0, 1, 0],
            'open': [100.0, 100.0, 100.0, 100.0],
            'close': [120.0, 100.0, 90.0, 100.0],
            'price_change': [20.0, 0.0, -10.0, 0.0],
        }),
        execution_lag_bars=0,
        fee_bps=0.0,
        slip_bps=0.0,
    ).iloc[0]

    assert result['mean_kelly_pct'] == 25.0


def test_backtest_snapshot_executes_on_next_bar() -> None:
    result = backtest_snapshot(
        pd.DataFrame({
            'predictions': [1, 0, 0],
            'open': [100.0, 100.0, 100.0],
            'close': [110.0, 90.0, 100.0],
            'price_change': [10.0, -10.0, 0.0],
        }),
        fee_bps=0.0,
        slip_bps=0.0,
    ).iloc[0]

    assert result['bars_total'] == 2
    assert result['trades_count'] == 1
    assert result['bars_in_market_pct'] == 50.0
    assert result['total_return_net_pct'] == -10.0


def test_backtest_snapshot_preserves_shifted_hold_while_one_continuation() -> None:
    result = backtest_snapshot(
        pd.DataFrame({
            'predictions': [1, 1, 0],
            'open': [100.0, 100.0, 110.0],
            'close': [100.0, 110.0, 121.0],
            'price_change': [0.0, 10.0, 11.0],
        }),
        fee_bps=0.0,
        slip_bps=0.0,
        trades_count_mode='runs',
    ).iloc[0]

    assert result['trades_count'] == 1
    assert result['total_return_gross_pct'] == 21.0
    assert result['total_return_net_pct'] == 21.0


def test_backtest_snapshot_drops_predictions_without_immediate_next_execution_bar() -> None:
    result = backtest_snapshot(
        pd.DataFrame({
            'predictions': [1, 0, 0],
            'open': [100.0, np.nan, 100.0],
            'close': [100.0, np.nan, 100.0],
            'price_change': [0.0, np.nan, 0.0],
        }),
        fee_bps=0.0,
        slip_bps=0.0,
    ).iloc[0]

    assert result['bars_total'] == 1
    assert result['trades_count'] == 0
    assert result['total_return_net_pct'] == 0.0


def test_completed_bar_signal_proves_next_bar_alignment() -> None:
    perf = _permutation_prediction_performance(_DummyCompletedBarSignal(), round_id=0)

    same_row = backtest_snapshot(
        perf,
        execution_lag_bars=0,
        fee_bps=0.0,
        slip_bps=0.0,
    ).iloc[0]
    next_bar = backtest_snapshot(
        perf,
        execution_lag_bars=1,
        fee_bps=0.0,
        slip_bps=0.0,
    ).iloc[0]

    assert same_row['total_return_net_pct'] == 21.0
    assert next_bar['total_return_net_pct'] == -19.0


def test_experiment_backtest_results_directionalizes_regression_predictions() -> None:
    result = _experiment_backtest_results(
        _DummyRegressionBacktestLog(),
        disable_progress_bar=True,
    ).iloc[0]

    expected = backtest_snapshot(
        pd.DataFrame({
            'predictions': [1, 0, 1],
            'open': [100.0, 100.0, 100.0],
            'close': [100.0, 110.0, 90.0],
            'price_change': [0.0, 10.0, -10.0],
        }),
        execution_lag_bars=1,
    ).iloc[0]

    assert result['total_return_net_pct'] == expected['total_return_net_pct']


def test_prepare_snapshot_backtest_input_rejects_multiclass() -> None:
    with pytest.raises(ValueError, match='snapshot backtest does not support multiclass'):
        _prepare_snapshot_backtest_input(
            pd.DataFrame({
                'predictions': [0, 1, 2],
                'actuals': [0, 1, 2],
            })
        )


def test_prepare_snapshot_backtest_input_rejects_non_numeric_logged_values() -> None:
    with pytest.raises(ValueError, match='snapshot backtest received non-numeric prediction values'):
        _prepare_snapshot_backtest_input(
            pd.DataFrame({
                'predictions': [1, 'bad', 0],
                'actuals': [1, 0, 0],
            })
        )


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
