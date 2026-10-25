from __future__ import annotations

import json
import weakref

import pytest

from limen.experiment import _factorize
from limen.experiment._factorize import FactorizedRounds
from limen.experiment._factorized_key import parameter_key
from limen.experiment.errors import StrictModeError
from limen.experiment.manifest_core import MLManifest
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from tests.test_factorize_equivalence import _numeric_equal
from tests.test_factorized_preparation import (
    _config, _fitted, _instrument, _loop, _params, _rounds, _run, _science,
)


def _key(params, names):
    return json.dumps({name: params[name] for name in names}, sort_keys=True, default=str)


def _bounded_config():
    config = _config()
    config['sfd']['params']['lookback'] = [2, 3, 4]
    config['uel']['search_strategy'] = {'type': 'grid', 'shuffle': True, 'seed': 42}
    config['uel']['n_permutations'] = 12
    return config


def test_cache_eviction_equivalence(tmp_path, monkeypatch):
    assert _factorize._CACHE_LIMIT == 128
    monkeypatch.setattr(_factorize, '_CACHE_LIMIT', 2)
    config = _bounded_config()
    params = config['sfd']['params']
    candidates = list(GridStrategy(ParamDomain(params), shuffle=True, seed=42))
    ordinary = _loop(config, tmp_path / 'ordinary')
    with monkeypatch.context() as hooks:
        ordinary_counts = _instrument(hooks)
        _run(ordinary, enabled=False, count=len(candidates))
    assert ordinary_counts == {'target': 12, 'scaler': 12, 'model': 12, 'prepare': 12}

    prediction_order, preparation_order = [], []
    expected_counts = {'target': 0, 'scaler': 0, 'model': 0, 'prepare': 0}
    peaks = {'prediction': 0, 'preparation': 0}
    observed = []
    original = FactorizedRounds.evaluate

    def capture(rounds, candidate):
        assert rounds.preparation is not None
        signal_key = parameter_key({key: value for key, value in candidate.items() if key not in rounds.axes})
        prep_key = _key(candidate, rounds.preparation.names)
        if signal_key in prediction_order:
            prediction_order.remove(signal_key)
        else:
            expected_counts['model'] += 1
            expected_counts['prepare'] += 1
            if prep_key in preparation_order:
                preparation_order.remove(prep_key)
            else:
                expected_counts['target'] += 1
                expected_counts['scaler'] += 1
            preparation_order.append(prep_key)
            preparation_order[:] = preparation_order[-2:]
        prediction_order.append(signal_key)
        prediction_order[:] = prediction_order[-2:]
        output = original(rounds, candidate)
        observed.append(dict(candidate))
        assert list(rounds.cache) == prediction_order
        assert list(rounds.preparation.cache) == preparation_order
        assert counts == expected_counts
        peaks['prediction'] = max(peaks['prediction'], len(rounds.cache))
        peaks['preparation'] = max(peaks['preparation'], len(rounds.preparation.cache))
        return output

    factorized = _loop(config, tmp_path / 'factorized')
    with monkeypatch.context() as hooks:
        counts = _instrument(hooks)
        hooks.setattr(FactorizedRounds, 'evaluate', capture)
        _run(factorized, enabled=True, count=len(candidates))
    assert observed == [{key: value for key, value in candidate.items() if key != '_id'}
                        for candidate in candidates]
    assert peaks == {'prediction': 2, 'preparation': 2}
    assert counts['target'] <= counts['model'] <= ordinary_counts['model']
    assert factorized.experiment_log['_round_index'].to_list() == list(range(12))
    assert factorized.experiment_log['_id'].to_list() == [candidate['_id'] for candidate in candidates]
    _numeric_equal(_science(tmp_path / 'ordinary'), _science(tmp_path / 'factorized'))


def test_cache_recency_release_and_failed_misses(monkeypatch):
    monkeypatch.setattr(_factorize, '_CACHE_LIMIT', 2)
    config = _bounded_config()
    rounds = _rounds(config)
    assert rounds.preparation is not None
    params = _params(config)
    counts = _instrument(monkeypatch)

    def evaluate(**updates):
        data, result, _, _ = rounds.evaluate({**params, **updates})
        assert len(rounds.cache) <= 2 and len(rounds.preparation.cache) <= 2
        assert all(set(entry) == {'_target_cls_up_next', '_scaler'}
                   for entry in rounds.preparation.cache.values())
        assert all('_model' not in entry.result and 'x_train' not in entry.data
                   for entry in rounds.cache.values())
        return data, result

    data, result = evaluate()
    expected = _fitted(data)
    first_key = next(iter(rounds.cache))
    first_prep = next(iter(rounds.preparation.cache))
    prep_ref = weakref.ref(rounds.preparation.cache[first_prep]['_scaler'])
    model_ref, frame_ref = weakref.ref(result['_model']), weakref.ref(data['x_train'])
    data['_scaler'].medians.clear()
    data['_target_cls_up_next'].cutoff = -1.
    del data, result
    assert model_ref() is None and frame_ref() is None

    data, result = evaluate(lookback=3)
    second_key = next(reversed(rounds.cache))
    prediction_ref = weakref.ref(rounds.cache[second_key].result['_preds'])
    price_ref = weakref.ref(rounds.cache[second_key].data['price_data_for_backtest'])
    del data, result
    data, result = evaluate(fee=9.)
    assert _fitted(data) == expected
    assert list(rounds.cache) == [second_key, first_key]
    assert next(iter(rounds.preparation.cache)) == first_prep
    del data, result
    data, result = evaluate(lookback=4)
    del data, result
    assert second_key not in rounds.cache and first_key in rounds.cache
    assert first_prep not in rounds.preparation.cache
    assert prediction_ref() is None and price_ref() is None and prep_ref() is None
    assert counts == {'target': 3, 'scaler': 3, 'model': 3, 'prepare': 3}

    data, result = evaluate(n_estimators=8)
    _numeric_equal(_fitted(data), expected)
    del data, result
    data, result = evaluate(lookback=4, n_estimators=8)
    del data, result
    assert counts == {'target': 4, 'scaler': 4, 'model': 5, 'prepare': 5}
    assert json.loads(next(reversed(rounds.preparation.cache)))['lookback'] == 4
    before = (list(rounds.cache), list(rounds.preparation.cache))
    original_model = MLManifest.run_model

    def fail(*args, **kwargs):
        raise StrictModeError('Recorded cache miss fails before retention')

    monkeypatch.setattr(MLManifest, 'run_model', fail)
    rounds.model = rounds.manifest.run_model
    with pytest.raises(StrictModeError, match='before retention'):
        rounds.evaluate({**params, 'lookback': 3})
    assert (list(rounds.cache), list(rounds.preparation.cache)) == before
    assert counts == {'target': 5, 'scaler': 5, 'model': 5, 'prepare': 6}
    monkeypatch.setattr(MLManifest, 'run_model', original_model)
    rounds.model = rounds.manifest.run_model
    data, result = evaluate(lookback=3)
    del data, result
    assert counts == {'target': 6, 'scaler': 6, 'model': 6, 'prepare': 7}
