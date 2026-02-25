from limen.experiment.param_domain import ParamDomain
from limen.experiment.msq import MSQ, FilterExhaustedError
from tests.stubs.stubs import StubStrategy


def _make_msq(params=None, n_permutations=None):

    if params is None:
        params = {'a': [1, 2], 'b': ['x', 'y']}
    domain = ParamDomain(params)
    strategy = StubStrategy(domain)
    return MSQ(strategy, domain, n_permutations=n_permutations), domain


def _param_keys(combo):

    return {k: v for k, v in combo.items() if not k.startswith('_')}


def test_msq_basic_iteration():

    msq, _ = _make_msq()
    combos = list(msq)

    assert len(combos) == 4

    for i, c in enumerate(combos):
        assert c['_id'] == i
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

    assert first['_id'] < second['_id'] < third['_id']


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
    strategy = StubStrategy(domain)
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
    strategy = StubStrategy(domain1)

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
