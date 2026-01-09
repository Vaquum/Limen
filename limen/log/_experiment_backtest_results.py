import tqdm
import pandas as pd

from limen.backtest.backtest_snapshot import backtest_snapshot


def _experiment_backtest_results(self, disable_progress_bar: bool = False) -> pd.DataFrame:

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

        result_df = backtest_snapshot(self.permutation_prediction_performance(i))

        all_rows.append(result_df)

    df_all = pd.concat(all_rows, ignore_index=True)

    return df_all
