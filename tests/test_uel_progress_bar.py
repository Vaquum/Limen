import uuid

import limen


def test_uel_progress_bar() -> None:
    '''Smoke test for `enable_progress_bar` in `uel.run`.'''

    uel = limen.UniversalExperimentLoop(
        sfd=limen.sfd.foundational_sfd.random_binary)

    experiment_name = f"pb_{uuid.uuid4().hex[:8]}"

    uel.run(
        experiment_name=experiment_name,
        n_permutations=1,
        prep_each_round=True,
        enable_progress_bar=False,
    )

    assert len(uel.round_params) == 1


if __name__ == '__main__':

    test_uel_progress_bar()
