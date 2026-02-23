from typing import Any
import pandas as pd

from limen.backtest.backtest_snapshot import backtest_snapshot


def _experiment_backtest_results(self: Any) -> pd.DataFrame:
    '''
    Compute backtest results for each round of an experiment.

    Returns:
        pd.DataFrame: One-row-per-round table with columns 'trade_win_rate_pct',
                      'trade_expectancy_pct', 'max_drawdown_pct',
                      'total_return_gross_pct', 'total_return_net_pct',
                      'trade_return_mean_win_pct', 'trade_return_mean_loss_pct',
                      'bars_total', 'sharpe_per_bar', 'bars_in_market_pct',
                      'trades_count', 'cost_round_trip_bps'
    '''

    all_rows = []

    for i in range(len(self.round_params)):

        result_df = backtest_snapshot(self.permutation_prediction_performance(i))

        all_rows.append(result_df)

    df_all = pd.concat(all_rows, ignore_index=True)

    return df_all
