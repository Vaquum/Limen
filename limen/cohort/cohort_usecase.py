from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from limen.cohort.cohort import Cohort
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.trainer import Trainer
from limen.sfd.foundational_sfd import logreg_binary as logreg_sfd
from limen.sfd.foundational_sfd import xgboost_regressor as xgboost_sfd
from tests.stubs.stubs import StubStrategy


def _run_real_experiment(experiment_dir: Path,
                         *,
                         sfd_module,
                         experiment_name: str,
                         n_permutations: int) -> None:

    params = sfd_module.params()
    domain = ParamDomain(params)
    strategy = StubStrategy(domain)

    uel = UniversalExperimentLoop(
        sfd=sfd_module,
        search_strategy=strategy,
        experiment_dir=experiment_dir,
    )
    uel.run(experiment_name=experiment_name, n_permutations=n_permutations)


def _train_real_members_and_input(experiment_dir: Path,
                                  permutation_ids: list[int]) -> tuple[list, Any]:

    trainer = Trainer(experiment_dir)
    sensors = trainer.train(permutation_ids)
    data_dict = trainer._manifest.prepare_data(
        trainer._data, sensors[0].round_params)
    return sensors, data_dict['x_test']


def _select_schema_compatible_permutation_ids(experiment_dir: Path,
                                              *,
                                              min_members: int = 2) -> list[int]:

    trainer = Trainer(experiment_dir)
    ids_by_schema: dict[tuple[str, ...], list[int]] = {}

    for pid in sorted(trainer._round_data.keys()):
        round_params = dict(trainer._round_data[pid]['round_params'])
        data_dict = trainer._manifest.prepare_data(trainer._data, round_params)

        x_test = data_dict['x_test']
        if hasattr(x_test, 'columns'):
            schema = tuple(x_test.columns)
        else:
            schema = (f'array_shape_{np.asarray(x_test).shape[1:]}',)

        ids_by_schema.setdefault(schema, []).append(pid)

    for ids in ids_by_schema.values():
        if len(ids) >= min_members:
            return ids[:min_members]

    raise RuntimeError(
        f'Unable to find {min_members} schema-compatible permutations in experiment.'
    )


def run_probability_usecase() -> dict:

    with TemporaryDirectory() as tmp:
        exp_dir = Path(tmp) / 'probability_exp'
        _run_real_experiment(
            exp_dir,
            sfd_module=logreg_sfd,
            experiment_name='cohort_usecase_probability',
            n_permutations=8,
        )

        permutation_ids = _select_schema_compatible_permutation_ids(
            exp_dir,
            min_members=2,
        )
        members, x_test = _train_real_members_and_input(
            exp_dir, permutation_ids)

        cohort = Cohort(
            experiment_log_path=str(exp_dir),
            permutation_ids=permutation_ids,
        )
        cohort.set_members(members)

        predict_result = cohort.predict(x_test)
        decoder_result = cohort({'x_test': x_test})
        return {
            'predict_result': predict_result,
            'decoder_result': decoder_result,
            'mode': cohort.aggregation_mode,
            'n_rows': len(x_test),
        }


def run_fallback_usecase() -> dict:

    with TemporaryDirectory() as tmp:
        exp_dir = Path(tmp) / 'fallback_exp'
        _run_real_experiment(
            exp_dir,
            sfd_module=xgboost_sfd,
            experiment_name='cohort_usecase_fallback',
            n_permutations=2,
        )

        permutation_ids = [0, 1]
        members, x_test = _train_real_members_and_input(
            exp_dir, permutation_ids)

        cohort = Cohort(
            experiment_log_path=str(exp_dir),
            permutation_ids=permutation_ids,
        )
        cohort.set_members(members)

        predict_result = cohort.predict(x_test)
        decoder_result = cohort({'x_test': x_test})
        return {
            'predict_result': predict_result,
            'decoder_result': decoder_result,
            'mode': cohort.aggregation_mode,
            'n_rows': len(x_test),
        }


def test_real_usecase_probability_mode():

    payload = run_probability_usecase()
    predict_result = payload['predict_result']
    decoder_result = payload['decoder_result']

    assert payload['mode'] == 'probability_weighted'
    assert isinstance(predict_result, np.ndarray)
    assert isinstance(decoder_result, dict)
    assert '_preds' in decoder_result
    assert '_probs' in decoder_result
    assert len(predict_result) == payload['n_rows']
    assert len(decoder_result['_preds']) == payload['n_rows']
    assert len(decoder_result['_probs']) == payload['n_rows']
    assert np.array_equal(np.asarray(decoder_result['_preds']), predict_result)
    assert np.all((np.asarray(decoder_result['_probs']) >= 0.0) &
                  (np.asarray(decoder_result['_probs']) <= 1.0))


def test_real_usecase_fallback_mode():

    payload = run_fallback_usecase()
    predict_result = payload['predict_result']
    decoder_result = payload['decoder_result']

    assert payload['mode'] == 'majority_vote'
    assert isinstance(predict_result, np.ndarray)
    assert isinstance(decoder_result, dict)
    assert '_preds' in decoder_result
    assert '_probs' not in decoder_result
    assert len(predict_result) == payload['n_rows']
    assert len(decoder_result['_preds']) == payload['n_rows']
    assert np.array_equal(np.asarray(decoder_result['_preds']), predict_result)


if __name__ == '__main__':

    prob = run_probability_usecase()
    print('Probability mode:', prob['mode'], 'rows=', prob['n_rows'])

    fallback = run_fallback_usecase()
    print('Fallback mode:', fallback['mode'], 'rows=', fallback['n_rows'])
