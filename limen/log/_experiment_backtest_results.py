from typing import Any
import tqdm
import pandas as pd

from limen.backtest.backtest_snapshot import backtest_snapshot


def _prepare_snapshot_backtest_input(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Normalize experiment backtest predictions to the binary snapshot contract.

    Binary SFDs already log 0/1 predictions. Regression SFDs log continuous
    values, so for snapshot backtests we convert predictions to the same
    directional convention used by the inline reference-architecture path.
    '''

    predictions = pd.to_numeric(df['predictions'], errors='coerce')
    valid_predictions = predictions.dropna()
    if not valid_predictions.empty and valid_predictions.isin([0, 1]).all():
        return df

    normalized = df.copy()
    normalized['predictions'] = (predictions > 0).astype(int)

    return normalized


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
                      'bar_win_rate_pct', 'bar_expectancy_pct',
                      'bar_return_mean_win_pct', 'bar_return_mean_loss_pct',
                      'mean_kelly_pct', 'bars_total', 'sharpe_per_bar',
                      'bars_in_market_pct', 'bars_in_market_count',
                      'trades_count', 'trade_runs_count', 'cost_round_trip_bps',
                      'execution_lag_bars'
    '''

    all_rows = []

    for i in tqdm.tqdm(range(len(self.round_params)), disable=disable_progress_bar):

        perf_df = _prepare_snapshot_backtest_input(
            self.permutation_prediction_performance(i)
        )
        result_df = backtest_snapshot(
            perf_df,
            execution_lag_bars=1,
            trades_count_mode='runs',
        )

        all_rows.append(result_df)

    df_all = pd.concat(all_rows, ignore_index=True)

    return df_all
