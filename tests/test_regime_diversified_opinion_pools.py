"""Test script for RDOP pipeline with backtesting functionality"""

import uuid
import sys
import traceback
import pandas as pd

import limen
from limen import sfd
from limen import RegimeDiversifiedOpinionPools
from tests.utils.cleanup import cleanup_csv_files


def test_rdop():
    '''Test RDOP pipeline with foundational SFDs.'''

    foundational_sfds = [
        sfd.foundational_sfd.xgboost_regressor,
        sfd.foundational_sfd.logreg_binary,
    ]

    for sfd_module in foundational_sfds:

        try:
            confusion_metrics = []
            n_permutations = 1

            for i in range(n_permutations):
                uel = limen.UniversalExperimentLoop(sfd=sfd_module)
                experiment_name = uuid.uuid4().hex[:8]

                uel.run(
                    experiment_name=experiment_name,
                    n_permutations=1,
                    prep_each_round=True
                )

                confusion_df = uel.experiment_confusion_metrics
                confusion_metrics.append(confusion_df)

            confusion_metrics = pd.concat(confusion_metrics, ignore_index=True)

            rdop = RegimeDiversifiedOpinionPools(sfd_module)

            _offline_result = rdop.offline_pipeline(
                confusion_metrics=confusion_metrics,
                perf_cols=None,
                iqr_multiplier=10.0,
                target_count=2,
                n_pca_components=2,
                n_pca_clusters=3,
                k_regimes=1
            )

            _online_result = rdop.online_pipeline(
                data=uel.data,
                aggregation_method='mean',
                aggregation_threshold=0.5
            )

            cleanup_csv_files()

            print(f'    ✅ {sfd_module.__name__}: PASSED')

        except Exception as e:
            print(f'    ❌ {sfd_module.__name__}: FAILED - {e}')
            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    test_rdop()
