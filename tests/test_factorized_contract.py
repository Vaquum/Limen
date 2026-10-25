import copy
import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from limen.experiment import ReuseContract, declare_reuse
from limen.experiment._factorized_key import parameter_key
from limen.sfd.reference_architecture.ridge_regressor import ridge_regressor
from tests.test_factorize_equivalence import _numeric_equal
from tests.test_factorized_preparation import _config, _loop, _run, _science, _rounds, _params


def _configuration(case):
    config = _config()
    manifest = config['sfd']['manifest']
    if case.startswith('ridge'):
        manifest['reference_architecture'] = 'limen.sfd.reference_architecture.ridge_regressor'
        manifest['target'] = {'name': 'next_return', 'class': 'limen.targets.NextReturnTarget', 'transform_params': {'periods': 1, 'scale': 100}}
        config['sfd']['params'] = {'lookback': [2], 'fee': [4., 9.], 'alpha': [1., 2.], 'solver': ['cholesky'], 'positive': [False], 'copy_X': [True]}
    if case == 'ridge_rank':
        manifest['scaler'] = {'class': 'limen.scalers.RankGaussScaler'}
    elif case == 'ridge_pca':
        manifest['pca_compression'] = {}
        config['sfd']['params'].update(auto_pca=[True], pca_k=[1])
    elif case == 'registry':
        manifest['scaler'] = {'from_params': 'scaler_type'}
        config['sfd']['params']['scaler_type'] = ['rank_gauss']
    elif case == 'target':
        manifest['target'] = {'name': 'up', 'class': 'limen.targets.NextBarUpTarget'}
    elif case == 'logreg':
        manifest['reference_architecture'] = 'limen.sfd.reference_architecture.logreg_binary'
        config['sfd']['params'] = {'lookback': [2], 'quantile': [0.5], 'fee': [4., 9.], 'C': [0.1, 1.], 'solver': ['lbfgs'], 'n_jobs': [1], 'max_iter': [1000]}
    return config


@pytest.mark.parametrize('case', ['ridge', 'ridge_rank', 'ridge_pca', 'registry', 'target', 'logreg'])
def test_recorded_factorized_manifest_contract(case, tmp_path, monkeypatch):
    config = _configuration(case)
    ordinary = _loop(copy.deepcopy(config), tmp_path / 'ordinary')
    _run(ordinary, enabled=False)
    reused = _loop(copy.deepcopy(config), tmp_path / 'reused')
    calls = []
    # Count the original execution boundary without replacing its declaration.
    from limen.experiment import _factorize
    execute = _factorize.FactorizedRounds.evaluate

    def evaluate(rounds, params):
        model = rounds.model
        def count_model(*args, **kwargs):
            calls.append(params.copy())
            return model(*args, **kwargs)
        rounds.model = count_model
        try:
            return execute(rounds, params)
        finally:
            rounds.model = model

    monkeypatch.setattr(_factorize.FactorizedRounds, 'evaluate', evaluate)
    _run(reused, enabled=True)
    assert len(calls) == 2
    _numeric_equal(_science(tmp_path / 'ordinary'), _science(tmp_path / 'reused'))
    assert len(set(reused.experiment_log['backtest_pnl_per_bar_bps'].to_list())) > 1


def test_factorized_declared_local_architecture(tmp_path):
    config = _configuration('ridge')
    rounds = _rounds(config)
    baseline = rounds.manifest.architecture_function
    calls = []
    def local(data, **params):
        calls.append(params['alpha'])
        return baseline(data, **{key: value for key, value in params.items() if key in ('alpha', 'solver', 'positive', 'copy_X')})
    declare_reuse(local, ReuseContract(lambda settings: None, consumes=('alpha', 'solver', 'positive', 'copy_X'), prediction_kind='continuous'))
    rounds.manifest.with_reference_architecture(local)
    plain = copy.deepcopy(rounds.manifest).with_reference_architecture(ridge_regressor)
    for fee in (4., 9.):
        params = _params(config) | {'fee': fee}
        _data, result, _, _ = rounds.evaluate(params)
        original_data = plain.prepare_data(rounds.data, params)
        expected = plain.run_model(original_data, params)
        _numeric_equal({key: value for key, value in expected.items() if key not in ('_model', '_preds')},
                       {key: value for key, value in result.items() if key not in ('_model', '_preds')})
        np.testing.assert_array_equal(result['_preds'], expected['_preds'])
    assert calls == [1.]


@pytest.mark.parametrize('case', ['ridge', 'seed', 'economic', 'opaque'])
def test_factorized_contract_refusals(case, tmp_path):
    config = _configuration('ridge')
    if case == 'ridge':
        config['sfd']['params']['solver'] = ['cholesky', 'auto']
    elif case == 'seed':
        config = _config()
        config['sfd']['params']['random_state'] = [42, None]
    elif case == 'economic':
        config['sfd']['manifest']['features'][0]['params']['end'] = '{fee}'
    else:
        config['sfd']['params']['opaque'] = [SimpleNamespace(value=1)]
    with pytest.raises(ValueError, match='factorize'):
        if case == 'opaque':
            _rounds(config)
        else:
            loop = _loop(config, tmp_path / 'refused')
            _run(loop, enabled=True)
    assert not (tmp_path / 'refused/metadata.json').exists()


def test_factorized_cache_key_retains_types():
    values = [None, True, 1, 1., '1', (1, 2), [1, 2], {'value': 1}, np.int64(1), np.float64(1.), -0., 0.]
    assert len({parameter_key({'value': value}) for value in values}) == len(values)


def test_factorized_preflight_projects_only_eligibility_inputs(monkeypatch):
    from limen.experiment._factorized_native import contract_for
    from limen.sfd.reference_architecture.lightgbm_binary import lightgbm_binary

    declaration = contract_for(lightgbm_binary)
    calls = []
    def validate(settings):
        calls.append(settings)
        declaration.validate(settings)
    monkeypatch.setattr(lightgbm_binary, '__limen_reuse__', replace(declaration, validate=validate), raising=False)
    config = _config()
    config['sfd']['params'].update({f'unrelated_{index}': list(range(10)) for index in range(10)})
    config['sfd']['params'].update({f'unexposed_resource_alias_{index}': list(range(10)) for index in range(5)})
    _rounds(config)
    assert 0 < len(calls) <= 500


@pytest.mark.parametrize('fees', [[4., 9.], [9., 4.]])
def test_factorized_warning_ownership(fees, tmp_path, monkeypatch):
    from limen.targets import QuantileBinaryTarget
    from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
    from limen.sfd.reference_architecture import base, _backtest_evaluation

    target, model, economic = QuantileBinaryTarget.__init__, LightGBMBinary.train, _backtest_evaluation.compute_backtest
    counts = {'fit': 0, 'model': 0}
    def fit(*args, **kwargs):
        counts['fit'] += 1
        warnings.warn('recorded-fit-warning', RuntimeWarning, stacklevel=2)
        return target(*args, **kwargs)
    def train(*args, **kwargs):
        counts['model'] += 1
        warnings.warn(f"recorded-model-{kwargs['n_estimators']}", RuntimeWarning, stacklevel=2)
        return model(*args, **kwargs)
    def execute(predictions, data):
        if data['backtest_fee_bps'] == 4.:
            warnings.warn('recorded-fee-warning', ConvergenceWarning, stacklevel=2)
        return economic(predictions, data)
    monkeypatch.setattr(QuantileBinaryTarget, '__init__', fit)
    monkeypatch.setattr(LightGBMBinary, 'train', train)
    monkeypatch.setattr(base, '_compute_backtest', execute)
    monkeypatch.setattr(_backtest_evaluation, 'compute_backtest', execute)
    observed = []
    for enabled in (False, True):
        counts.update(fit=0, model=0)
        config = _config()
        config['sfd']['params']['fee'] = fees
        loop = _loop(config, tmp_path / str(enabled))
        _run(loop, enabled=enabled)
        assert counts == ({'fit': 1, 'model': 2} if enabled else {'fit': 4, 'model': 4})
        observed.append((_science(tmp_path / str(enabled)), loop.convergence_report))
    _numeric_equal(observed[0][0], observed[1][0])
    assert observed[0][1] == observed[1][1]


def test_factorized_declared_economic_replay_owns_payload():
    from limen.experiment._factorize import FactorizedRounds
    from limen.experiment.param_domain import ParamDomain
    from limen.experiment.param_search import GridStrategy
    from limen.sfd.reference_architecture._backtest_evaluation import compute_backtest

    config = _configuration('ridge')
    config['sfd']['params'].update(alpha=[1.], fee=[4., 9., 14.])
    original = _rounds(config)
    manifest, calls, replayed = original.manifest, [], []
    def extension(data_dict, **_kwargs):
        data_dict['_auxiliary'] = {'value': data_dict['x_val']['ret_1'].mean()}
        return data_dict
    def local(data, **params):
        calls.append(params['alpha'])
        result = ridge_regressor(data, alpha=params['alpha'], solver=params['solver'])
        result['custom_economic_value'] = result['backtest_pnl_per_bar_bps'] + params['fee'] * data['_auxiliary']['value']
        data['_auxiliary']['value'] = None
        return result
    def replay(scientific, data, params):
        replayed.append(data['_auxiliary']['value'])
        fresh = scientific | compute_backtest((np.asarray(scientific['_preds']) > 0).astype(int), data)
        fresh['custom_economic_value'] = fresh['backtest_pnl_per_bar_bps'] + params['fee'] * data['_auxiliary']['value']
        data['_auxiliary']['value'] = None
        return fresh
    declare_reuse(extension, ReuseContract(lambda settings: None, validation_keys=(), consumes=()))
    declare_reuse(local, ReuseContract(lambda settings: None, validation_keys=(), consumes=('alpha', 'solver'),
                                      prediction_kind='continuous', replay=replay, replay_inputs=('_auxiliary',)))
    manifest.with_reference_architecture(local)
    manifest.data_dict_extension = extension
    domain = config['sfd']['params']
    rounds = FactorizedRounds(manifest=manifest, strategy=GridStrategy(ParamDomain(domain)), domain=domain,
                              pruning=False, callback=False, context=None, prep=manifest.prepare_data,
                              model=manifest.run_model, data=original.data, record_execution=True, record_model_outputs=True)
    plain = copy.deepcopy(manifest).with_reference_architecture(ridge_regressor)
    for fee in domain['fee']:
        params = _params(config) | {'fee': fee}
        oracle = plain.prepare_data(rounds.data, params)
        expected = plain.run_model(oracle, params)
        _data, result, _, _ = rounds.evaluate(params)
        assert result['custom_economic_value'] == expected['backtest_pnl_per_bar_bps'] + fee * oracle['_auxiliary']['value']
    assert len(calls) == 1 and len(replayed) == 3
    assert all(value == replayed[0] and value is not None for value in replayed)
