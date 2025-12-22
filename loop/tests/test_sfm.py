import uuid
import sys
import traceback

import loop
from loop import sfm
from loop.tests.utils.cleanup import cleanup_csv_files


def test_sfm():
    '''Test all reference SFMs.'''

    reference_sfms = [
        sfm.reference.random,
        sfm.reference.xgboost,
        sfm.reference.logreg,
        sfm.reference.lightgbm,
    ]

    for single_file_model in reference_sfms:

        try:
            uel = loop.UniversalExperimentLoop(single_file_model=single_file_model)
            experiment_name = uuid.uuid4().hex[:8]

            uel.run(
                experiment_name=experiment_name,
                n_permutations=2,
                prep_each_round=True
            )

            print(f'    ✅ {single_file_model.__name__}: PASSED')

        except Exception as e:
            print(f'    ❌ {single_file_model.__name__}: FAILED - {e}')
            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    test_sfm()
