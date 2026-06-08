import inspect
import re
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import ClassVar

from limen.backtest.backtest_snapshot import BACKTEST_SNAPSHOT_COLUMNS
from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.backtest.long_flat_strategy import ExecutionResult
from limen.backtest.long_flat_strategy import long_flat_strategy
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


class _DummyConfusionMetricsOutlierFiltered:

    def permutation_prediction_performance(self, round_id: int) -> pd.DataFrame:
        assert round_id == 0
        return pd.DataFrame({
            'predictions': [1, 0, 0, 0, 0],
            'actuals': [1, 1, 0, 0, 0],
            'open': [100.0, 100.0, 100.0, 100.0, 100.0],
            'close': [110.0, 90.0, 105.0, 100.0, 100.0],
            'price_change': [10.0, -10.0, 5.0, 0.0, 0.0],
            'x_metric': [1000.0, 0.0, 0.0, 0.0, 0.0],
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


def test_permutation_confusion_metrics_keeps_mean_returns_on_unfiltered_rows() -> None:
    result = _permutation_confusion_metrics(
        _DummyConfusionMetricsOutlierFiltered(),
        x='x_metric',
        round_id=0,
        outlier_quantiles=(0.25, 0.75),
    ).iloc[0]

    assert result['n_kept'] == 4
    assert np.isnan(result['tp_x_mean'])
    assert result['tp_mean_return_pct'] == -10.0
    assert result['fn_mean_return_pct'] == 5.0


def test_backtest_snapshot_emits_metric_ledger_columns() -> None:
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

    assert result.index.tolist() == BACKTEST_SNAPSHOT_COLUMNS
    assert len(BACKTEST_SNAPSHOT_COLUMNS) == 20
    assert result['wins_per_bar'] == 0.25
    assert result['pnl_per_bar_bps'] == 250.0
    assert result['avg_win_bps'] == 2000.0
    assert result['avg_loss_bps'] == -1000.0
    assert result['drawdown_bps_p50'] == -500.0
    assert result['trades_per_bar'] == 0.5


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

    assert result['wins_per_bar'] == 0.0
    assert result['avg_loss_bps'] == -1000.0
    assert result['pnl_per_bar_bps'] == -333.3
    assert result['drawdown_bps_p50'] == -1000.0


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
    ).iloc[0]

    assert result['edge_bps_p50'] == 1000.0
    assert result['pnl_bps_p50'] == 1000.0
    assert result['cost_bps_p50'] == 0.0
    assert result['inventory_per_bar'] == 0.6667
    assert result['pnl_per_bar_bps'] == 666.7


def test_backtest_snapshot_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match='backtest_snapshot requires at least one row'):
        backtest_snapshot(
            pd.DataFrame({
                'predictions': [],
                'open': [],
                'close': [],
                'price_change': [],
            })
        )


def test_backtest_snapshot_rejects_non_binary_predictions() -> None:
    with pytest.raises(ValueError, match='predictions must contain only 0 or 1'):
        backtest_snapshot(
            pd.DataFrame({
                'predictions': [1, 2, 0],
                'open': [100.0, 100.0, 100.0],
                'close': [101.0, 102.0, 100.0],
                'price_change': [1.0, 2.0, 0.0],
            })
        )


def test_backtest_snapshot_rejects_missing_predictions() -> None:
    with pytest.raises(ValueError, match='predictions must contain only 0 or 1'):
        backtest_snapshot(
            pd.DataFrame({
                'predictions': [1, np.nan, 0],
                'open': [100.0, 100.0, 100.0],
                'close': [101.0, 102.0, 100.0],
                'price_change': [1.0, 2.0, 0.0],
            })
        )


def test_backtest_snapshot_rejects_non_numeric_price_values() -> None:
    with pytest.raises(ValueError, match='open, close, and price_change must be numeric'):
        backtest_snapshot(
            pd.DataFrame({
                'predictions': [1, 0, 0],
                'open': [100.0, 'bad', 100.0],
                'close': [101.0, 102.0, 100.0],
                'price_change': [1.0, 2.0, 0.0],
            })
        )


def test_backtest_snapshot_rejects_inconsistent_price_change() -> None:
    with pytest.raises(ValueError, match='price_change must equal close - open'):
        backtest_snapshot(
            pd.DataFrame({
                'predictions': [1, 0, 0],
                'open': [100.0, 100.0, 100.0],
                'close': [101.0, 102.0, 100.0],
                'price_change': [1.0, 1.0, 0.0],
            })
        )


def test_backtest_snapshot_applies_costs_multiplicatively_per_fill() -> None:
    result = backtest_snapshot(
        pd.DataFrame({
            'predictions': [1],
            'open': [100.0],
            'close': [100.0],
            'price_change': [0.0],
        }),
        execution_lag_bars=0,
        fee_bps=50.0,
        slip_bps=50.0,
    ).iloc[0]

    assert result['cost_bps_p50'] == 198.3
    assert result['pnl_bps_p50'] == -198.3
    assert result['cost_per_bar_bps'] == 198.3


def test_backtest_snapshot_drawdown_includes_starting_equity_peak() -> None:
    result = backtest_snapshot(
        pd.DataFrame({
            'predictions': [1],
            'open': [100.0],
            'close': [90.0],
            'price_change': [-10.0],
        }),
        execution_lag_bars=0,
        fee_bps=0.0,
        slip_bps=0.0,
    ).iloc[0]

    assert result['drawdown_bps_p50'] == -1000.0


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

    assert result['inventory_per_bar'] == 0.0
    assert result['pnl_bps_p50'] == 0.0
    assert np.isnan(result['avg_win_bps'])
    assert np.isnan(result['avg_loss_bps'])


def test_backtest_snapshot_reports_all_bars_population() -> None:
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

    assert result['inventory_per_bar'] == 0.5
    assert result['wins_per_bar'] == 0.25
    assert result['edge_bps_p50'] == 0.0
    assert result['pnl_bps_p50'] == 0.0


def test_backtest_snapshot_cvar_floor_5pct_tail() -> None:
    close = [101.0] * 20
    close[7] = 80.0
    close[8] = 101.0
    result = backtest_snapshot(
        pd.DataFrame({
            'predictions': [1] * 20,
            'open': [100.0] * 20,
            'close': close,
            'price_change': [value - 100.0 for value in close],
        }),
        fee_bps=0.0,
        slip_bps=0.0,
    ).iloc[0]

    assert result['cvar_95_pnl_bps'] == -2079.2

    short = backtest_snapshot(
        pd.DataFrame({
            'predictions': [1, 0] * 5,
            'open': [100.0] * 10,
            'close': [101.0] * 10,
            'price_change': [1.0] * 10,
        }),
        fee_bps=0.0,
        slip_bps=0.0,
    ).iloc[0]

    assert np.isnan(short['cvar_95_pnl_bps'])


def test_backtest_snapshot_edge_case_sentinels() -> None:
    flat = backtest_snapshot(
        pd.DataFrame({
            'predictions': [0] * 25,
            'open': [100.0] * 25,
            'close': [101.0] * 25,
            'price_change': [1.0] * 25,
        }),
        fee_bps=0.0,
        slip_bps=0.0,
    ).iloc[0]

    assert np.isnan(flat['avg_win_bps'])
    assert np.isnan(flat['avg_loss_bps'])
    assert flat['cvar_95_pnl_bps'] == 0.0
    assert flat['wins_per_bar'] == 0.0
    assert flat['pnl_per_bar_bps'] == 0.0
    assert flat['trades_per_bar'] == 0.0

    no_losers = backtest_snapshot(
        pd.DataFrame({
            'predictions': [1] * 25,
            'open': [100.0] * 25,
            'close': [101.0] * 25,
            'price_change': [1.0] * 25,
        }),
        fee_bps=0.0,
        slip_bps=0.0,
    ).iloc[0]

    assert np.isnan(no_losers['avg_loss_bps'])
    assert not np.isnan(no_losers['avg_win_bps'])


def test_backtest_snapshot_scalars_invariant_under_window_doubling() -> None:
    prices = [100.0, 110.0, 99.0, 100.0, 105.0, 95.0, 100.0]
    base = pd.DataFrame({
        'predictions': [0, 1, 1, 0, 1, 1, 0],
        'open': [100.0] * 7,
        'close': prices,
        'price_change': [value - 100.0 for value in prices],
    })
    doubled = pd.concat([base, base], ignore_index=True)

    single = backtest_snapshot(base, execution_lag_bars=0, fee_bps=5.0, slip_bps=5.0).iloc[0]
    twice = backtest_snapshot(doubled, execution_lag_bars=0, fee_bps=5.0, slip_bps=5.0).iloc[0]

    intensive_scalars = [
        'wins_per_bar',
        'pnl_per_bar_bps',
        'avg_win_bps',
        'avg_loss_bps',
        'cost_per_bar_bps',
        'trades_per_bar',
        'inventory_per_bar',
    ]
    for column in intensive_scalars:
        assert single[column] == twice[column], column


def test_backtest_snapshot_is_time_free() -> None:
    import limen.backtest.backtest_snapshot as snapshot_module

    params = inspect.signature(backtest_snapshot).parameters
    assert 'datetime_col' not in params
    assert 'clock_window' not in params

    source = inspect.getsource(snapshot_module)
    assert not re.search(r'datetime|clock_window|Timedelta|to_datetime|dt\.floor|Timestamp|DatetimeIndex', source)


def test_backtest_snapshot_delegates_to_injected_strategy() -> None:
    def constant_long(predictions, *args, **kwargs) -> ExecutionResult:
        index = predictions.index
        return ExecutionResult(
            pos=pd.Series(1.0, index=index),
            gross=pd.Series(0.01, index=index),
            net=pd.Series(0.01, index=index),
        )

    result = backtest_snapshot(
        pd.DataFrame({
            'predictions': [0, 0, 0],
            'open': [100.0, 100.0, 100.0],
            'close': [100.0, 100.0, 100.0],
            'price_change': [0.0, 0.0, 0.0],
        }),
        strategy=constant_long,
    ).iloc[0]

    assert result['inventory_per_bar'] == 1.0
    assert result['wins_per_bar'] == 1.0
    assert result['pnl_per_bar_bps'] == 100.0
    assert result['edge_bps_p50'] == 100.0
    assert result['trades_per_bar'] == 0.33333


def test_long_flat_strategy_returns_execution_result() -> None:
    result = long_flat_strategy(
        pd.Series([1, 1, 0]),
        pd.Series([100.0, 100.0, 110.0]),
        pd.Series([100.0, 110.0, 121.0]),
        pd.Series([0.0, 10.0, 11.0]),
        execution_lag_bars=1,
        fee_bps=0.0,
        slip_bps=0.0,
    )

    assert isinstance(result, ExecutionResult)
    assert list(result.pos) == [0.0, 1.0, 1.0]
    assert result.net.tolist() == pytest.approx(result.gross.tolist())


def test_long_flat_strategy_rejects_negative_lag() -> None:
    with pytest.raises(ValueError, match='execution_lag_bars must be >= 0'):
        long_flat_strategy(
            pd.Series([1, 0]),
            pd.Series([100.0, 100.0]),
            pd.Series([100.0, 100.0]),
            pd.Series([0.0, 0.0]),
            execution_lag_bars=-1,
        )


def test_backtest_snapshot_notional_rate_scales_returns_not_structure() -> None:
    df = pd.DataFrame({
        'predictions': [1, 1, 0, 0],
        'open': [100.0, 100.0, 100.0, 100.0],
        'close': [110.0, 121.0, 100.0, 100.0],
        'price_change': [10.0, 21.0, 0.0, 0.0],
    })
    full = backtest_snapshot(df, execution_lag_bars=0, fee_bps=0.0, slip_bps=0.0, notional_rate=1.0).iloc[0]
    half = backtest_snapshot(df, execution_lag_bars=0, fee_bps=0.0, slip_bps=0.0, notional_rate=0.5).iloc[0]

    assert half['pnl_per_bar_bps'] == pytest.approx(0.5 * full['pnl_per_bar_bps'])
    assert half['edge_bps_p50'] == pytest.approx(0.5 * full['edge_bps_p50'])
    assert half['inventory_per_bar'] == pytest.approx(0.5 * full['inventory_per_bar'])
    assert half['wins_per_bar'] == full['wins_per_bar']
    assert half['trades_per_bar'] == full['trades_per_bar']


def test_long_flat_strategy_rejects_notional_rate_outside_unit_interval() -> None:
    prices = (
        pd.Series([1, 0]),
        pd.Series([100.0, 100.0]),
        pd.Series([100.0, 100.0]),
        pd.Series([0.0, 0.0]),
    )
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match='notional_rate must be in'):
            long_flat_strategy(*prices, notional_rate=bad)


def test_no_legacy_backtest_column_names() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    legacy = re.compile(
        r'edge_per_signal|trade_pnl_net|cost_drag|rolling_return|return_on_exposure'
        r'|drawdown_depth|drawdown_duration|cvar_95_return|bar_in_market|capital_in_market'
        r'|in_market_per_bar'
    )
    this_file = Path(__file__).resolve()
    offenders = [
        str(path.relative_to(repo_root))
        for base in ('limen', 'docs', 'tests')
        for path in (repo_root / base).rglob('*')
        if path.suffix in {'.py', '.md'}
        and path.resolve() != this_file
        and legacy.search(path.read_text(encoding='utf-8'))
    ]
    assert offenders == [], offenders


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

    assert same_row['pnl_bps_p50'] == 500.0
    assert next_bar['pnl_bps_p50'] == -500.0


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

    assert result['pnl_bps_p50'] == expected['pnl_bps_p50']


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
