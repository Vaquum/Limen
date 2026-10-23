import pytest

from limen.experiment._factorize import FactorizedRounds
from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
from tests.test_factorized_preparation import _config, _loop, _run
from tests.test_parallel_rounds import records
from tests.test_result_field_collisions import _assert_message, _rows


def test_factorized_cached_round_uses_same_collision_guard(tmp_path, monkeypatch):
    config = _config()
    config['sfd']['params']['n_estimators'] = [4]
    evaluated = []
    trained = []
    evaluate = FactorizedRounds.evaluate
    train = LightGBMBinary.train

    def collide(rounds, params):
        data, result, warnings, converged = evaluate(rounds, params)
        evaluated.append(params['fee'])
        if params['fee'] == 9.0:
            result['fee'] = params['fee']
        return data, result, warnings, converged

    def observe_training(model, data, **params):
        trained.append(params['n_estimators'])
        return train(model, data, **params)

    monkeypatch.setattr(FactorizedRounds, 'evaluate', collide)
    monkeypatch.setattr(LightGBMBinary, 'train', observe_training)
    loop = _loop(config, tmp_path)
    with pytest.raises(ValueError) as error:
        _run(loop, enabled=True, count=2)
    _assert_message(error, 'round parameters/result', ('fee',))
    assert evaluated == [4.0, 9.0]
    assert trained == [4]
    rows = _rows(tmp_path)
    assert len(rows) == 1 and float(rows[0]['fee']) == 4.0
    assert [record['round_id'] for record in records(tmp_path)] == [rows[0]['id']]
    assert loop.experiment_log is None
    assert not (tmp_path / 'audit.jsonl').exists()
