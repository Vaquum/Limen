import tqdm
import pandas as pd


def _experiment_confusion_metrics(self, x, disable_progress_bar=False):

    all_rows = []
    
    for i in tqdm.tqdm(range(len(self.round_params)), disable=disable_progress_bar):

        result_df = self.permutation_confusion_metrics(x=x,
                                                       round_id=i,
                                                       id_cols=self.round_params[i])

        all_rows.append(result_df)

    df_all = pd.concat(all_rows, ignore_index=True)

    return df_all
