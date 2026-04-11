"""Test script for RDOP pipeline with backtesting functionality"""

import sys
import traceback
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import limen
from limen import sfd
from limen import RegimeDiversifiedOpinionPools
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy
from tests.utils.cleanup import cleanup_csv_files

logger = logging.getLogger(__name__)


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

            for _i in range(n_permutations):
                with TemporaryDirectory() as tmpdir:
                    experiment_dir = Path(tmpdir) / 'experiment'
                    domain = ParamDomain(sfd_module.params())
                    strategy = RandomStrategy(domain, seed=42)

                    uel = limen.UniversalExperimentLoop(
                        sfd=sfd_module,
                        search_strategy=strategy,
                        experiment_dir=experiment_dir,
                    )

                    uel.run(
                        experiment_name=str(experiment_dir / 'test'),
                        n_permutations=1,
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
            logger.info('    ✅ %s: PASSED', sfd_module.__name__)

        except Exception as e:
            logger.error('    ❌ %s: FAILED - %s', sfd_module.__name__, str(e))
            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    test_rdop()
