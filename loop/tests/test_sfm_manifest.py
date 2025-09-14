import uuid
import sys
import loop
import traceback

from loop import sfm
from loop.tests.utils.cleanup import cleanup_csv_files
from loop.tests.utils.get_data import get_klines_data


def test_sfm_manifest():

    manifest_tests = [
        (sfm.reference.logreg_manifest, get_klines_data, True),
    ]

    for test in manifest_tests:

        try:

            manifest = test[0].manifest()
            uel = loop.UniversalExperimentLoop(data=test[1](),
                                                single_file_model=test[0])

            uel.run(experiment_name=uuid.uuid4().hex[:8],
                    n_permutations=2,
                    prep_each_round=test[2],
                    manifest=manifest)

            print(f'    ✅ {test[0].__name__} (manifest): PASSED')

        except Exception as e:

            print(f'    ❌ {test[0].__name__} (manifest): FAILED - {e}')

            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":

    test_sfm_manifest()
