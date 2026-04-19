from typing import Any
import numpy as np
import pandas as pd
import tqdm

from limen.backtest.backtest_snapshot import backtest_snapshot


def _prepare_snapshot_backtest_input(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Normalize logged prediction outputs to the binary contract expected by snapshot.

    Regression reference architectures directionalize inline with ``preds > 0`` before
    computing backtest metrics. Post-run snapshot summaries need to mirror that same
    contract so inline and logged backtest results stay identical.

    Supports:
    - binary 0/1 predictions directly
    - directional regression scores via sign, mapped to the current long/flat
      snapshot decision

    Does not support multiclass snapshot backtests. Non-negative integer labels
    with more than two classes raise explicitly instead of being collapsed.
    '''

    def _is_binary(series: pd.Series) -> bool:
        valid = series.dropna()
        return valid.empty or valid.isin([0, 1]).all()

    def _to_numeric_strict(series: pd.Series, name: str) -> pd.Series:
        numeric = pd.to_numeric(series, errors='coerce')
        invalid = numeric.isna() & ~series.isna()

        if invalid.any():
            raise ValueError(
                f'snapshot backtest received non-numeric {name} values'
            )

        return numeric

    def _is_unsupported_multiclass(series: pd.Series) -> bool:
        BINARY_CLASS_COUNT = 2
        valid = series.dropna()
        if valid.empty:
            return False

        integer_like = np.isclose(valid, np.round(valid)).all()
        unique_count = pd.Index(np.round(valid).astype(int)).nunique()

        return integer_like and (valid >= 0).all() and unique_count > BINARY_CLASS_COUNT

    result = df.copy()
    pred = _to_numeric_strict(result['predictions'], 'prediction')

    if 'actuals' in result:
        actual = _to_numeric_strict(result['actuals'], 'actual')
        if _is_unsupported_multiclass(actual):
            raise ValueError(
                'snapshot backtest does not support multiclass actuals; '
                'use binary predictions or directional regression scores'
            )

    if _is_binary(pred):
        return result

    if _is_unsupported_multiclass(pred):
        raise ValueError(
            'snapshot backtest does not support multiclass predictions; '
            'use binary predictions or directional regression scores'
        )

    result['predictions'] = (pred > 0).astype(int)

    return result


def _experiment_backtest_results(self: Any, disable_progress_bar: bool = False) -> pd.DataFrame:

    '''
    Compute backtest results for each round of an experiment.

    Args:
        disable_progress_bar (bool): Whether to disable the progress bar

    Returns:
        pd.DataFrame: One-row-per-round table with columns 'trade_win_rate_pct',
                      'trade_expectancy_pct', 'max_drawdown_pct',
                      'total_return_gross_pct', 'total_return_net_pct',
                      'trade_return_mean_win_pct', 'trade_return_mean_loss_pct',
                      'bars_total', 'sharpe_per_bar', 'bars_in_market_pct',
                      'trades_count', 'cost_round_trip_bps'
    '''

    all_rows = []

    for i in tqdm.tqdm(range(len(self.round_params)), disable=disable_progress_bar):
        perf = _prepare_snapshot_backtest_input(self.permutation_prediction_performance(i))

        result_df = backtest_snapshot(
            perf,
            execution_lag_bars=1,
        )

        all_rows.append(result_df)

    df_all = pd.concat(all_rows, ignore_index=True)

    return df_all
