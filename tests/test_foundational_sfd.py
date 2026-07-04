import limen
import sys
import traceback
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy
from limen.experiment.reducer import BudgetReducer
from limen.experiment.reducer import CorrelationReducer
from limen.experiment.reducer import FocusReducer
from limen.experiment.reducer import SanityReducer
from limen.experiment.reducer import SaturationReducer

from tests.utils.cleanup import cleanup_csv_files

logger = logging.getLogger(__name__)


def test_foundational_sfd():
    '''Test all foundational SFDs via MSQ path with RandomStrategy and pruning.'''

    foundational_sfds = [
        limen.sfd.foundational_sfd.dlinear_regressor,
        limen.sfd.foundational_sfd.lightgbm_binary,
        limen.sfd.foundational_sfd.random_binary,
        limen.sfd.foundational_sfd.xgboost_regressor,
        limen.sfd.foundational_sfd.logreg_binary,
        limen.sfd.foundational_sfd.rule_based,
    ]

    for sfd_module in foundational_sfds:

        try:
            with TemporaryDirectory() as tmpdir:
                experiment_dir = Path(tmpdir) / 'experiment'
                domain = ParamDomain(sfd_module.params())
                strategy = RandomStrategy(domain, seed=42)

                pruning_strategies = [
                    SanityReducer(metric='auc', min_observations=1),
                    BudgetReducer(max_permutations=10, check_after_pct=0.0),
                    CorrelationReducer(metric='auc', min_observations=1, n_boot=10),
                    FocusReducer(metric='auc', breakthrough_threshold=0.5, min_observations=1),
                    SaturationReducer(metric='auc', min_samples_per_value=1, window_size=2),
                ]

                uel = limen.UniversalExperimentLoop(
                    sfd=sfd_module,
                    search_strategy=strategy,
                    pruning_strategies=pruning_strategies,
                    feedback_interval=1,
                    experiment_dir=experiment_dir,
                    test_mode=True,
                )

                uel.run(
                    experiment_name=str(experiment_dir / 'test'),
                    n_permutations=2,
                )

                if sfd_module is limen.sfd.foundational_sfd.rule_based:
                    log = uel.experiment_log
                    assert log is not None, 'experiment_log is None'
                    assert 'num_trades_test' in log.columns, 'num_trades_test missing from log'
                    assert 'pnl_bps_p50_test' in log.columns, 'pnl_bps_p50_test missing from log'
                    assert 'is_stable' in log.columns, 'is_stable missing from log'
                    assert uel.experiment_confusion_metrics is None, 'experiment_confusion_metrics should be None for rule-based'
                    assert uel.experiment_backtest_results is None, 'experiment_backtest_results should be None for rule-based'
                else:
                    assert sfd_module.manifest().strict_mode is True, f'{sfd_module.__name__} manifest must have strict_mode=True'

            logger.info('    ✅ %s: PASSED', sfd_module.__name__)

        except Exception as e:
            logger.error('    ❌ %s: FAILED - %s', sfd_module.__name__, str(e))
            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    test_foundational_sfd()
