import copy
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.experiment.param_domain import ParamDomain
from limen.experiment.msq import MSQ, FilterExhaustedError
from limen.experiment.param_search import GridStrategy, RandomStrategy


_INVALID_CAPS = (True, False, 0, -1, 1.5, 2.0, float('nan'), float('inf'), '3', np.int64(3))
_CAP_IDS = ('true', 'false', 'zero', 'negative', 'fractional', 'integral-float', 'nan', 'infinity', 'string', 'numpy-integer')


class _IntCap(int):
    __slots__ = ()


@pytest.fixture(scope='module')
def recorded_choices():
    return pl.read_parquet(Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet', n_rows=32)['close'].unique(maintain_order=True).head(4).to_list()


@pytest.mark.parametrize('cap', _INVALID_CAPS, ids=_CAP_IDS)
def test_invalid_cap_refused_before_iteration(cap, recorded_choices):
    domain = ParamDomain({'threshold': recorded_choices})
    strategy = GridStrategy(domain)
    before = copy.deepcopy(strategy.get_state())
    with pytest.raises(ValueError, match='n_permutations must be a positive non-bool int or None'):
        MSQ(strategy, domain, n_permutations=cap)
    assert strategy.get_state() == before


@pytest.mark.parametrize('cap', _INVALID_CAPS, ids=_CAP_IDS)
def test_invalid_restored_cap_leaves_queue_unchanged(cap, recorded_choices):
    queue, _ = _make_msq({'threshold': recorded_choices}, n_permutations=3)
    next(queue)
    before = copy.deepcopy(queue.get_state())
    invalid = copy.deepcopy(before)
    invalid.update(n_permutations=cap, yielded_count=99)
    with pytest.raises(ValueError, match='n_permutations must be a positive non-bool int or None'):
        queue.set_state(invalid)
    assert queue.get_state() == before


@pytest.mark.parametrize('cap', (None, 1, 3, 100, _IntCap(2)))
def test_valid_caps_preserve_natural_grid_exhaustion(cap, recorded_choices):
    queue, _ = _make_msq({'threshold': recorded_choices}, n_permutations=cap)
    rows = list(queue)
    assert len(rows) == (len(recorded_choices) if cap is None else min(cap, len(recorded_choices)))


def test_legacy_state_without_cap_restores_none(recorded_choices):
    queue, _ = _make_msq({'threshold': recorded_choices}, n_permutations=1)
    state = queue.get_state()
    state.pop('n_permutations')
    queue.set_state(state)
    assert queue.get_state()['n_permutations'] is None
    assert len(list(queue)) == len(recorded_choices)


@pytest.mark.parametrize('strategy_type', (GridStrategy, RandomStrategy))
def test_cap_above_initial_domain_binds_after_growth(strategy_type, recorded_choices):
    domain = ParamDomain({'threshold': recorded_choices[:1]})
    queue = MSQ(strategy_type(domain, seed=42), domain, n_permutations=3)
    rows = [next(queue)]
    for value in recorded_choices[1:]:
        assert queue.inject_value('threshold', value)
    rows.extend(queue)
    assert len(rows) == 3
    assert len({row['threshold'] for row in rows}) == 3
    assert [row['_round_index'] for row in rows] == [0, 1, 2]
    queue.inject({'threshold': recorded_choices[-1]}, prioritize=True)
    assert queue.priority_queue_size == 1
    with pytest.raises(StopIteration):
        next(queue)
    assert queue.yielded_count == 3


@pytest.mark.parametrize('cap', (True, 0, 1.5))
def test_invalid_uel_cap_creates_no_experiment_artifacts(cap, tmp_path):
    from tests.test_experiment_core_msq import _make_uel

    directory = tmp_path / 'run'
    loop, strategy, _ = _make_uel(experiment_dir=directory)
    before = copy.deepcopy(strategy.get_state())
    with pytest.raises(ValueError, match='n_permutations must be a positive non-bool int or None'):
        loop.run('invalid_cap', n_permutations=cap, progress_bar=False)
    assert not directory.exists()
    assert strategy.get_state() == before


def _make_msq(params=None, n_permutations=None):

    if params is None:
        params = {'a': [1, 2], 'b': ['x', 'y']}
    domain = ParamDomain(params)
    strategy = GridStrategy(domain)
    return MSQ(strategy, domain, n_permutations=n_permutations), domain


def _param_keys(combo):

    return {k: v for k, v in combo.items() if not k.startswith('_')}


def test_msq_basic_iteration():

    msq, _ = _make_msq()
    combos = list(msq)

    assert len(combos) == 4

    for i, c in enumerate(combos):
        assert c['_round_index'] == i
        assert isinstance(c['_id'], str)
        assert '_param_hash' not in c
        assert c['_injected'] is False

    param_combos = {frozenset(_param_keys(c).items()) for c in combos}
    expected = {
        frozenset({('a', 1), ('b', 'x')}),
        frozenset({('a', 1), ('b', 'y')}),
        frozenset({('a', 2), ('b', 'x')}),
        frozenset({('a', 2), ('b', 'y')}),
    }
    assert param_combos == expected


def test_msq_yielded_count():

    msq, _ = _make_msq()
    assert msq.yielded_count == 0

    next(msq)
    assert msq.yielded_count == 1

    next(msq)
    assert msq.yielded_count == 2


def test_priority_queue():

    msq, _ = _make_msq()

    msq.inject({'a': 10, 'b': 'back'})
    msq.inject({'a': 20, 'b': 'front'}, prioritize=True)
    assert msq.priority_queue_size == 2

    first = next(msq)
    assert first['a'] == 20 and first['_injected'] is True
    assert msq.priority_queue_size == 1

    second = next(msq)
    assert second['a'] == 10 and second['_injected'] is True

    # Next comes from strategy
    third = next(msq)
    assert third['_injected'] is False

    assert first['_round_index'] < second['_round_index'] < third['_round_index']


def test_inject_validates_keys():

    msq, _ = _make_msq()

    try:
        msq.inject({'a': 1})
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'missing' in str(e).lower()

    try:
        msq.inject({'a': 1, 'b': 'x', 'c': 99})
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'extra' in str(e).lower()


def test_intervention_routing():

    msq, domain = _make_msq({'lr': [0.001, 0.01, 0.05, 0.1, 1.0]})

    msq.remove_is('lr', 1.0)
    assert 1.0 not in domain.values_for('lr')

    removed = msq.remove_ge('lr', 0.1)
    assert removed == 1
    assert domain.values_for('lr') == [0.001, 0.01, 0.05]

    removed = msq.remove_le('lr', 0.001)
    assert removed == 1
    assert domain.values_for('lr') == [0.01, 0.05]

    assert msq.inject_value('lr', 0.03) is True
    assert 0.03 in domain.values_for('lr')

    msq.keep_between('lr', 0.01, 0.04)
    assert domain.values_for('lr') == [0.01, 0.03]

    msq.keep_is('lr', 0.01)
    assert domain.values_for('lr') == [0.01]


def test_remove_custom():

    msq, _ = _make_msq()

    msq.remove_custom(lambda c: c['a'] == 1 and c['b'] == 'x')

    combos = list(msq)
    assert len(combos) == 3
    for c in combos:
        p = _param_keys(c)
        assert not (p['a'] == 1 and p['b'] == 'x')


def test_filter_exhausted_error():

    domain = ParamDomain({'a': list(range(10)), 'b': list(range(10))})
    strategy = GridStrategy(domain)
    msq = MSQ(strategy, domain, max_filter_retries=5)

    msq.remove_custom(lambda c: True)

    try:
        next(msq)
        assert False, 'Should have raised FilterExhaustedError'
    except FilterExhaustedError:
        pass


def test_trim():

    msq, _ = _make_msq()

    next(msq)
    msq.trim(2)
    assert msq.remaining_count() == 1

    next(msq)

    try:
        next(msq)
        assert False, 'Should have raised StopIteration'
    except StopIteration:
        pass


def test_trim_with_priority_queue():

    msq, _ = _make_msq()
    msq.trim(2)
    msq.inject({'a': 10, 'b': 'z'})

    assert msq.remaining_count() == 2

    next(msq)
    assert msq.remaining_count() == 1


def test_remaining_count():

    msq, _ = _make_msq()
    assert msq.remaining_count() == 4
    next(msq)
    assert msq.remaining_count() == 3

    msq2, _ = _make_msq(n_permutations=100)
    assert msq2.remaining_count() == 100
    next(msq2)
    assert msq2.remaining_count() == 99


def test_distribution():

    # With n_permutations
    msq, _ = _make_msq(n_permutations=100)
    assert msq.distribution('a') == {1: 50, 2: 50}
    assert msq.distribution() == {'a': {1: 50, 2: 50}, 'b': {'x': 50, 'y': 50}}

    # After yielding
    next(msq)
    next(msq)
    assert msq.distribution('a') == {1: 49, 2: 49}

    # Without n_permutations, uses total_combinations for finite
    msq2, _ = _make_msq()
    assert msq2.distribution('a') == {1: 2, 2: 2}


def test_inject_does_not_mutate_caller_dict():

    msq, _ = _make_msq()
    original = {'a': 1, 'b': 'x'}
    msq.inject(original)

    next(msq)
    assert '_id' not in original
    assert '_injected' not in original


def test_n_permutations_stops_iteration():

    msq, _ = _make_msq(n_permutations=3)

    combos = list(msq)
    assert len(combos) == 3


def test_mismatched_domain_raises():

    domain1 = ParamDomain({'a': [1, 2]})
    domain2 = ParamDomain({'a': [1, 2]})
    strategy = GridStrategy(domain1)

    try:
        MSQ(strategy, domain2)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'same ParamDomain' in str(e)


def test_intervention_log():

    msq, _ = _make_msq()

    msq.remove_is('a', 2)
    msq.inject_value('b', 'z')

    log = msq.intervention_log
    assert len(log) == 2
    assert log[0]['operation'] == 'remove_is'
    assert log[1]['operation'] == 'inject_value'


def test_set_state_missing_key():

    msq, _ = _make_msq()
    state = msq.get_state()

    for key in ('yielded_count', 'trim_budget', 'priority_queue', 'intervention_log', 'strategy_state'):
        incomplete = {k: v for k, v in state.items() if k != key}
        try:
            msq.set_state(incomplete)
            assert False, f'Should have raised ValueError for missing {key!r}'
        except ValueError as e:
            assert key in str(e)


def test_get_set_state():

    msq, _ = _make_msq(n_permutations=100)

    next(msq)
    next(msq)
    msq.inject({'a': 10, 'b': 'z'})

    state = msq.get_state()

    assert state['yielded_count'] == 2
    assert state['n_permutations'] == 100
    assert len(state['priority_queue']) == 1
    assert state['trim_budget'] is None

    msq2, _ = _make_msq()
    msq2.set_state(state)

    assert msq2.yielded_count == 2
    assert msq2.priority_queue_size == 1
    assert msq2.remaining_count() == 99  # 98 remaining + 1 in queue


def test_set_filter():

    msq, _ = _make_msq(params={'a': [1, 2, 3], 'b': ['x']})
    msq.set_filter('test', lambda c: c['a'] == 3)
    combos = [_param_keys(c) for c in msq]

    assert len(combos) == 2
    assert all(c['a'] != 3 for c in combos)


def test_clear_filter():

    msq, _ = _make_msq(params={'a': [1, 2, 3], 'b': ['x']})
    msq.set_filter('test', lambda c: c['a'] == 3)

    next(msq)
    msq.clear_filter('test')

    remaining = [_param_keys(c) for c in msq]
    a_values = {c['a'] for c in remaining}
    assert 3 in a_values


def test_set_filter_replaces():

    msq, _ = _make_msq(params={'a': [1, 2, 3], 'b': ['x']})
    msq.set_filter('test', lambda c: c['a'] == 1)
    msq.set_filter('test', lambda c: c['a'] == 2)
    combos = [_param_keys(c) for c in msq]

    assert len(combos) == 2
    a_values = {c['a'] for c in combos}
    assert 1 in a_values
    assert 2 not in a_values


def test_named_filter_descriptor_roundtrip():

    from limen.experiment.reducer.filter_types import FILTER_BUILDERS
    from limen.experiment.reducer.filter_types import FILTER_KEEP_BETWEEN
    from limen.experiment.reducer.filter_types import FILTER_KEEP_VALUES

    msq, _ = _make_msq(params={'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z']})

    between_params = {'param': 'a', 'lower': 2, 'upper': 4}
    msq.set_filter(
        'focus_a',
        FILTER_BUILDERS[FILTER_KEEP_BETWEEN](between_params),
        filter_type=FILTER_KEEP_BETWEEN,
        filter_params=between_params,
    )

    values_params = {'param': 'b', 'values': ['x', 'y']}
    msq.set_filter(
        'focus_b',
        FILTER_BUILDERS[FILTER_KEEP_VALUES](values_params),
        filter_type=FILTER_KEEP_VALUES,
        filter_params=values_params,
    )

    next(msq)
    state = msq.get_state()

    assert 'named_filter_descriptors' in state
    assert 'focus_a' in state['named_filter_descriptors']
    assert 'focus_b' in state['named_filter_descriptors']

    msq2, _ = _make_msq(params={'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z']})
    msq2.set_state(state)

    combos = [_param_keys(c) for c in msq2]
    for c in combos:
        assert 2 <= c['a'] <= 4
        assert c['b'] in ('x', 'y')


def test_clear_filter_removes_descriptor():

    from limen.experiment.reducer.filter_types import FILTER_BUILDERS
    from limen.experiment.reducer.filter_types import FILTER_KEEP_VALUES

    msq, _ = _make_msq(params={'a': [1, 2, 3], 'b': ['x']})

    params = {'param': 'a', 'values': [1, 2]}
    msq.set_filter(
        'test',
        FILTER_BUILDERS[FILTER_KEEP_VALUES](params),
        filter_type=FILTER_KEEP_VALUES,
        filter_params=params,
    )

    state1 = msq.get_state()
    assert 'test' in state1['named_filter_descriptors']

    msq.clear_filter('test')
    state2 = msq.get_state()
    assert 'test' not in state2['named_filter_descriptors']


def test_non_descriptor_filter_warns_on_restore():

    import warnings

    msq, _ = _make_msq(params={'a': [1, 2], 'b': ['x']})
    msq.set_filter('bare', lambda c: c['a'] == 1)

    state = msq.get_state()
    assert 'bare' in state['named_filter_keys']
    assert 'bare' not in state['named_filter_descriptors']

    msq2, _ = _make_msq(params={'a': [1, 2], 'b': ['x']})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        msq2.set_state(state)
        assert len(w) == 1
        assert 'non-restorable' in str(w[0].message).lower()
