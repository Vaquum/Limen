from typing import Any
import pandas as pd
import tqdm

from limen.backtest.backtest_snapshot import backtest_snapshot


def _prepare_snapshot_backtest_input(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Normalize logged prediction outputs to the binary contract expected by snapshot.

    Regression reference architectures directionalize inline with ``preds > 0`` before
    computing backtest metrics. Post-run snapshot summaries need to mirror that same
    contract so inline and logged backtest results stay identical.
    '''

    result = df.copy()
    pred = pd.to_numeric(result['predictions'])

    if not pred.dropna().isin([0, 1]).all():
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
