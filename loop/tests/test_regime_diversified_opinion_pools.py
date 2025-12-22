"""Test script for RDOP pipeline with backtesting functionality"""

import uuid
import sys
import traceback
import pandas as pd

import loop
from loop import sfm
from loop import RegimeDiversifiedOpinionPools
from loop.tests.utils.cleanup import cleanup_csv_files


def test_rdop():
    '''Test RDOP pipeline with reference SFMs.'''

    reference_sfms = [
        sfm.reference.xgboost,
        sfm.reference.logreg,
        sfm.reference.lightgbm,
    ]

    for single_file_model in reference_sfms:

        try:
            confusion_metrics = []
            n_permutations = 1

            for i in range(n_permutations):
                uel = loop.UniversalExperimentLoop(single_file_model=single_file_model)
                experiment_name = uuid.uuid4().hex[:8]

                uel.run(
                    experiment_name=experiment_name,
                    n_permutations=1,
                    prep_each_round=True
                )

                confusion_df = uel.experiment_confusion_metrics
                confusion_metrics.append(confusion_df)

            confusion_metrics = pd.concat(confusion_metrics, ignore_index=True)

            rdop = RegimeDiversifiedOpinionPools(single_file_model)

            offline_result = rdop.offline_pipeline(
                confusion_metrics=confusion_metrics,
                perf_cols=None,
                iqr_multiplier=10.0,
                target_count=2,
                n_pca_components=2,
                n_pca_clusters=3,
                k_regimes=1
            )

            online_result = rdop.online_pipeline(
                data=uel.data,
                aggregation_method='mean',
                aggregation_threshold=0.5
            )

            cleanup_csv_files()

            print(f'    ✅ {single_file_model.__name__}: PASSED')

        except Exception as e:
            print(f'    ❌ {single_file_model.__name__}: FAILED - {e}')
            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    test_rdop()
