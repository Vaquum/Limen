"""Test script for RDOP pipeline with backtesting functionality"""

import uuid
import sys
import traceback
import pandas as pd
import polars as pl

import loop
from loop import sfm
from loop.regime_diversified_opinion_pools import RegimeDiversifiedOpinionPools
from loop.tests.utils.cleanup import cleanup_csv_files
from loop.tests.utils.get_data import get_klines_data_fast, get_klines_data_large, get_klines_data_small_fast


def test_rdop():

    '''Test RDOP pipeline with multiple SFMs.'''

    tests = [
        # COLUMN ORDER: sfm, data_function, prep_each_round, uses_manifest
        (sfm.reference.xgboost, get_klines_data_fast, False, False),
        (sfm.reference.logreg, get_klines_data_fast, True, True),
        (sfm.logreg.regime_multiclass, get_klines_data_large, True, True),
        (sfm.logreg.breakout_regressor_ridge, get_klines_data_large, True, True),
        (sfm.reference.lightgbm, get_klines_data_large, True, True),
        (sfm.lightgbm.tradeable_regressor, get_klines_data_large, False, False),
        (sfm.lightgbm.tradeline_long_binary, get_klines_data_fast, True, False),
        (sfm.lightgbm.tradeline_multiclass, get_klines_data_fast, True, False),
        (sfm.rules_based.momentum_volatility_longonly,
         get_klines_data_small_fast, True, False),
        (sfm.rules_based.momentum_volatility,
         get_klines_data_small_fast, True, False),
        (sfm.ridge.ridge_classifier, get_klines_data_fast, True, True)
    ]

    for test in tests:

        try:
            confusion_metrics = []
            data = test[1]()
            n_permutations = 1

            for i in range(n_permutations):
                uel = loop.UniversalExperimentLoop(data=data, single_file_model=test[0])

                experiment_name = uuid.uuid4().hex[:8]

                if test[3]:
                    manifest = test[0].manifest()
                    uel.run(experiment_name=experiment_name,
                            n_permutations=1,
                            prep_each_round=test[2],
                            manifest=manifest)
                else:
                    uel.run(experiment_name=experiment_name,
                            n_permutations=1,
                            prep_each_round=test[2])

                confusion_df = uel.experiment_confusion_metrics
                confusion_metrics.append(confusion_df)

            confusion_metrics = pd.concat(confusion_metrics, ignore_index=True)

            rdop = RegimeDiversifiedOpinionPools(test[0])

            offline_result = rdop.offline_pipeline(
                confusion_metrics=confusion_metrics,
                perf_cols=None,
                iqr_multiplier=10.0,
                target_count=2,
                n_components=2,
                n_clusters=3,
                k=1
            )

            online_result = rdop.online_pipeline(
                data=data,
                aggregation_method='mean',
                aggregation_threshold=0.5
            )

            cleanup_csv_files()

            print(f'    ✅ {test[0].__name__}: PASSED')

        except Exception as e:

            print(f'    ❌ {test[0].__name__}: FAILED - {e}')

            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    test_rdop()
