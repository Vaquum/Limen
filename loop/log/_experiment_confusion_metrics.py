import tqdm
import pandas as pd


def _experiment_confusion_metrics(self, x: str, disable_progress_bar: bool = False) -> pd.DataFrame:

    '''
    Compute confusion metrics for each round of an experiment.
    
    Args:
        x (str): Column name to compute confusion metrics for
        disable_progress_bar (bool): Whether to disable the progress bar

    Returns:
        pd.DataFrame: One-row-per-round table with confusion and long-only summary columns (see permutation_confusion_metrics)
    '''

    all_rows = []
    
    for i in tqdm.tqdm(range(len(self.round_params)), disable=disable_progress_bar):

        result_df = self.permutation_confusion_metrics(x=x,
                                                       round_id=i,
                                                       id_cols=self.round_params[i])

        all_rows.append(result_df)

    df_all = pd.concat(all_rows, ignore_index=True)

    return df_all
