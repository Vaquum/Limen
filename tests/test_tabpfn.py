import limen
import sys
import traceback
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from limen.experiment.param_domain import ParamDomain
from limen.experiment.random_strategy import RandomStrategy

from tests.utils.cleanup import cleanup_csv_files

logger = logging.getLogger(__name__)


def test_tabpfn():
    '''Test TabPFN SFD via MSQ path with RandomStrategy.'''

    tabpfn_sfds = [
        limen.sfd.foundational_sfd.tabpfn_binary,
    ]

    for sfd_module in tabpfn_sfds:

        try:
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
                    n_permutations=2,
                )

            logger.info('✅ %s: PASSED', sfd_module.__name__)

        except Exception as e:
            logger.error('❌ %s: FAILED - %s', sfd_module.__name__, str(e))
            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    test_tabpfn()
