import limen
import uuid
import sys
import traceback
import logging

from tests.utils.cleanup import cleanup_csv_files

logger = logging.getLogger(__name__)


def test_foundational_sfd():
    '''Test all foundational SFDs.'''

    foundational_sfds = [
        limen.sfd.foundational_sfd.random_binary,
        limen.sfd.foundational_sfd.xgboost_regressor,
        limen.sfd.foundational_sfd.logreg_binary,
    ]

    for sfd_module in foundational_sfds:

        try:
            uel = limen.UniversalExperimentLoop(sfd=sfd_module)
            experiment_name = uuid.uuid4().hex[:8]

            uel.run(
                experiment_name=experiment_name,
                n_permutations=2,
                prep_each_round=True
            )

            logger.info('    ✅ %s: PASSED', sfd_module.__name__)

        except Exception as e:
            logger.error('    ❌ %s: FAILED - %s', sfd_module.__name__, str(e))
            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    test_foundational_sfd()
