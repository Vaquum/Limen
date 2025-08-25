import loop
from loop.sfm.transformer import binary_classifier
import polars as pl
import numpy as np

df = pl.read_csv('2025_Q1.csv')
print(df.describe())

uel = loop.UniversalExperimentLoop(
    data=df,
    single_file_model=binary_classifier,
)

uel.run(
    experiment_name='binary_classifier_test',
    n_permutations=10,
    prep_each_round=False,          # prep once for baseline runs
    random_search=True
)

# Post-run analysis via precomputed artifacts and internal Log
backtest_df = uel.experiment_backtest_results
confusion_df = uel.experiment_confusion_metrics
round0_perf = uel._log.permutation_prediction_performance(round_id=0)