import limen
import uuid
import sys
import traceback

from tests.utils.cleanup import cleanup_csv_files


def test_tabpfn():
    '''Test TabPFN SFD.'''

    tabpfn_sfds = [
        limen.sfd.foundational_sfd.tabpfn_binary,
    ]

    for sfd_module in tabpfn_sfds:

        try:
            uel = limen.UniversalExperimentLoop(sfd=sfd_module)
            experiment_name = uuid.uuid4().hex[:8]

            uel.run(
                experiment_name=experiment_name,
                n_permutations=2,
                prep_each_round=True
            )

            print(f'    ✅ {sfd_module.__name__}: PASSED')

        except Exception as e:
            print(f'    ❌ {sfd_module.__name__}: FAILED - {e}')
            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    test_tabpfn()
