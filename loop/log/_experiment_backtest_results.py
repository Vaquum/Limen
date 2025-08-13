import tqdm
import pandas as pd

from backtest_results import backtest_results


def _experiment_backtest_results(self, disable_progress_bar=False):

    '''
    Compute confusion metrics for each round of an experiment.

    Args:
        disable_progress_bar (bool): Whether to disable the progress bar.
    '''

    all_rows = []
    
    for i in tqdm.tqdm(range(len(self.round_params)), disable=disable_progress_bar):

        result_df = backtest_results(self.permutation_prediction_performance(i))

        all_rows.append(result_df)

    df_all = pd.concat(all_rows, ignore_index=True)

    return df_all
