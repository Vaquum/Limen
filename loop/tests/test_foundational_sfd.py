import uuid
import sys
import traceback

import loop
from loop import sfd
from loop.tests.utils.cleanup import cleanup_csv_files


def test_foundational_sfd():
    '''Test all foundational SFDs.'''

    foundational_sfds = [
        sfd.foundational_sfd.random_binary,
        sfd.foundational_sfd.xgboost_regressor,
        sfd.foundational_sfd.logreg_binary,
        sfd.foundational_sfd.lightgbm_binary,
    ]

    for single_file_decoder in foundational_sfds:

        try:
            uel = loop.UniversalExperimentLoop(single_file_decoder=single_file_decoder)
            experiment_name = uuid.uuid4().hex[:8]

            uel.run(
                experiment_name=experiment_name,
                n_permutations=2,
                prep_each_round=True
            )

            print(f'    ✅ {single_file_decoder.__name__}: PASSED')

        except Exception as e:
            print(f'    ❌ {single_file_decoder.__name__}: FAILED - {e}')
            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    test_foundational_sfd()
