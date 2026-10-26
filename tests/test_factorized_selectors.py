from itertools import product
from math import prod

import pytest

from limen.experiment import ReuseContract, declare_reuse
from limen.experiment import _factorize, _factorized_admission as admission
from limen.experiment._factorized_projection import projections
from limen.experiment.param_search import GridStrategy
from limen.sfd.reference_architecture.ridge_regressor import ridge_regressor
from tests.test_factorized_contract import _configuration
from tests.test_factorized_preparation import _config, _instrument, _loop, _run


def _switches(config=None, *, order=(False, True)):
    config = _config() if config is None else config
    for index in range(30):
        config['sfd']['manifest']['indicators'].append({
            'func': 'limen.indicators.window_return', 'params': {'period': index + 2},
            'include_if': f'switch_{index}'})
        config['sfd']['params'][f'switch_{index}'] = list(order)
    return config


def _guard(monkeypatch, *, limit=136):
    counts = {'tuples': 0, 'validated': 0, 'draws': 0}
    validate = admission.validate_candidate
    draw = GridStrategy.__next__

    def bounded_product(*pools):
        for values in product(*pools):
            counts['tuples'] += 1
            assert counts['tuples'] <= limit, 'Preflight enumerated unrelated selector products'
            yield values

    def bounded_validate(*args):
        counts['validated'] += 1
        assert counts['validated'] <= limit, 'Preflight validation exceeded the linear bound'
        return validate(*args)

    def counted_draw(strategy):
        value = draw(strategy)
        counts['draws'] += 1
        return value

    monkeypatch.setattr(admission, 'product', bounded_product)
    monkeypatch.setattr(admission, 'validate_candidate', bounded_validate)
    monkeypatch.setattr(_factorize, 'validate_candidate', bounded_validate)
    monkeypatch.setattr(GridStrategy, '__next__', counted_draw)
    return counts


def _assert_linear(loop):
    domain = loop._search_strategy.domain.params
    axes = _factorize._axes(loop.manifest, domain)
    assert sum(prod(len(domain[name]) for name in group - axes) for group in projections(loop.manifest, domain)) <= 136


@pytest.mark.parametrize('order', ((False, True), (True, False)))
def test_thirty_independent_switches_complete_one_recorded_trial(order, tmp_path, monkeypatch):
    loop = _loop(_switches(order=order), tmp_path / 'run')
    _assert_linear(loop)
    work = _guard(monkeypatch)
    fitted = _instrument(monkeypatch)
    _run(loop, enabled=True, count=1)
    assert 0 < work['tuples'] <= 136
    assert 0 < work['validated'] <= 136
    assert work['draws'] == fitted['model'] == 1
    assert loop.experiment_log.height == 1
    assert loop.experiment_log['strict_mode_error'].null_count() == 1
    assert all(loop.experiment_log[f'switch_{index}'][0] is order[0] for index in range(30))


@pytest.mark.parametrize('grouped', (False, True))
def test_last_switch_and_joint_group_refuse_undeclared_work(grouped, tmp_path, monkeypatch):
    config = _switches()
    if grouped:
        config['sfd']['params']['feature_groups'] = ['other', 'danger']
    loop = _loop(config, tmp_path / 'run')

    def undeclared(frame):
        raise AssertionError('Undeclared transform executed before refusal')

    loop.manifest.add_feature(undeclared, include_if='switch_29', group='danger' if grouped else None)
    _assert_linear(loop)
    work = _guard(monkeypatch)
    fitted = _instrument(monkeypatch)
    with pytest.raises(ValueError, match='missing deterministic reuse declaration'):
        _run(loop, enabled=True, count=1)
    assert work['draws'] == fitted['model'] == fitted['target'] == fitted['scaler'] == 0
    assert not (tmp_path / 'run/metadata.json').exists()


@pytest.mark.parametrize('inactive', ('flag', 'group'))
def test_inactive_undeclared_component_stays_admitted(inactive, tmp_path, monkeypatch):
    config = _switches()
    config['sfd']['params']['feature_groups'] = ['other'] if inactive == 'group' else ['danger']
    if inactive == 'flag':
        config['sfd']['params']['switch_29'] = [False]
    loop = _loop(config, tmp_path / 'run')

    def undeclared(frame):
        raise AssertionError('Inactive transform executed')

    loop.manifest.add_feature(undeclared, include_if='switch_29', group='danger')
    _assert_linear(loop)
    _guard(monkeypatch)
    _run(loop, enabled=True, count=1)
    assert loop.experiment_log.height == 1


def test_native_joint_eligibility_is_not_split_into_single_axes(tmp_path, monkeypatch):
    config = _configuration('logreg')
    config['sfd']['params'].update(solver=['lbfgs', 'sag'], random_state=[42, None])
    loop = _loop(config, tmp_path / 'run')
    work = _guard(monkeypatch)
    with pytest.raises(ValueError, match='integer random_state'):
        _run(loop, enabled=True, count=1)
    assert work['draws'] == 0
    assert not (tmp_path / 'run/metadata.json').exists()


def test_activation_and_aliased_joint_reads_share_one_projection(tmp_path, monkeypatch):
    config = _switches()
    config['sfd']['params'].update(feature_groups=['other', 'danger'], left_alias=[1, 2], right_alias=[3, 4])
    loop = _loop(config, tmp_path / 'run')

    def transform(frame, left, right):
        raise AssertionError('Invalid joint eligibility reached execution')

    def validate(settings):
        if settings['left'] == 2 and settings['right'] == 4:
            raise ValueError('joint alias refusal')

    declare_reuse(transform, ReuseContract(validate, validation_keys=('left', 'right'), parameter_keys=('left', 'right')))
    loop.manifest.add_feature(transform, include_if='switch_29', group='danger', left='{left_alias}', right='right_alias')
    _guard(monkeypatch)
    with pytest.raises(ValueError, match='joint alias refusal'):
        _run(loop, enabled=True, count=1)
    assert not (tmp_path / 'run/metadata.json').exists()


def test_fixed_signature_default_contract_excludes_undelivered_switches(tmp_path, monkeypatch):
    config = _switches(_configuration('ridge'))
    loop = _loop(config, tmp_path / 'run')
    calls = []

    def local(data, alpha=1.0, solver='cholesky', positive=False, copy_X=True):
        calls.append(alpha)
        return ridge_regressor(data, alpha=alpha, solver=solver, positive=positive, copy_X=copy_X)

    declare_reuse(local, ReuseContract(lambda settings: None, prediction_kind='continuous'))
    loop.manifest.with_reference_architecture(local)
    _assert_linear(loop)
    _guard(monkeypatch)
    _run(loop, enabled=True, count=1)
    assert calls == [1.0]
    assert loop.experiment_log.height == 1


def test_default_kwargs_reads_remain_joint_and_distinct_from_empty(tmp_path):
    loop = _loop(_switches(), tmp_path / 'run')
    domain = loop._search_strategy.domain.params

    def local(data, **settings):
        raise AssertionError('Analytic projection control must not execute')

    declare_reuse(local, ReuseContract(lambda settings: None, prediction_kind='binary'))
    loop.manifest.with_reference_architecture(local)
    group = projections(loop.manifest, domain)[0]
    assert set(domain) <= group
    assert prod(len(domain[name]) for name in group - {'fee'}) >= 2 ** 30
    declare_reuse(local, ReuseContract(lambda settings: None, validation_keys=(), prediction_kind='binary'))
    assert projections(loop.manifest, domain)[0] == frozenset()
