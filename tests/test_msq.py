from limen.experiment.param_domain import ParamDomain
from limen.experiment.search_strategy import SearchStrategy
from limen.experiment.msq import MSQ, FilterExhaustedError


class StubStrategy(SearchStrategy):

    '''Deterministic finite strategy for testing. Cycles through all combos.'''

    def __init__(self, domain, *, seed=None):
        super().__init__(domain, seed=seed)
        self._combos = self._build_combos()
        self._index = 0

    @property
    def is_finite(self):
        return True

    def _build_combos(self):
        import itertools
        params = self._domain.params
        keys = sorted(params.keys())
        return [
            dict(zip(keys, vals, strict=True))
            for vals in itertools.product(*(params[k] for k in keys))
        ]

    def __next__(self):
        if self._index >= len(self._combos):
            raise StopIteration
        combo = self._combos[self._index]
        self._index += 1
        self._generated_count += 1
        return combo

    def on_domain_changed(self, domain, changed_params):
        self._combos = self._build_combos()
        self._index = 0

    def get_state(self):
        return {'index': self._index, 'generated_count': self._generated_count}

    def set_state(self, state):
        self._index = state['index']
        self._generated_count = state['generated_count']


class _InfiniteStubStrategy(SearchStrategy):

    '''Strategy that cycles forever. For testing filter exhaustion.'''

    def __init__(self, domain, *, seed=None):
        super().__init__(domain, seed=seed)

    def __next__(self):
        import random
        params = self._domain._params
        combo = {k: random.choice(v) for k, v in params.items()}
        self._generated_count += 1
        return combo

    def get_state(self):
        return {'generated_count': self._generated_count}

    def set_state(self, state):
        self._generated_count = state['generated_count']


def _make_msq(params=None, n_permutations=None):

    if params is None:
        params = {'a': [1, 2], 'b': ['x', 'y']}
    domain = ParamDomain(params)
    strategy = StubStrategy(domain)
    return MSQ(strategy, domain, n_permutations=n_permutations), domain


def _param_keys(combo):

    return {k: v for k, v in combo.items() if not k.startswith('_')}


def test_msq_basic_iteration():

    '''Yields all combinations with sequential _id and _injected metadata.'''

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

    '''Injected combos yield before strategy; prioritize controls ordering.'''

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

    # IDs are sequential regardless of source
    assert first['_id'] < second['_id'] < third['_id']


def test_inject_validates_keys():

    msq, _ = _make_msq()

    try:
        msq.inject({'a': 1})  # missing 'b'
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'missing' in str(e).lower()


def test_intervention_routing():

    '''All single-param MSQ methods route to ParamDomain correctly.'''

    msq, domain = _make_msq({'lr': [0.001, 0.01, 0.05, 0.1, 1.0]})

    # remove_is
    msq.remove_is('lr', 1.0)
    assert 1.0 not in domain.values_for('lr')

    # remove_ge
    removed = msq.remove_ge('lr', 0.1)
    assert removed == 1
    assert domain.values_for('lr') == [0.001, 0.01, 0.05]

    # remove_le
    removed = msq.remove_le('lr', 0.001)
    assert removed == 1
    assert domain.values_for('lr') == [0.01, 0.05]

    # inject_value
    assert msq.inject_value('lr', 0.03) is True
    assert 0.03 in domain.values_for('lr')

    # keep_between
    msq.keep_between('lr', 0.01, 0.04)
    assert domain.values_for('lr') == [0.01, 0.03]

    # keep_is
    msq.keep_is('lr', 0.01)
    assert domain.values_for('lr') == [0.01]


def test_remove_custom():

    msq, _ = _make_msq()

    msq.remove_custom(lambda c: not (c['a'] == 1 and c['b'] == 'x'))

    combos = list(msq)
    assert len(combos) == 3
    for c in combos:
        p = _param_keys(c)
        assert not (p['a'] == 1 and p['b'] == 'x')


def test_filter_exhausted_error():

    '''Overly aggressive filters raise FilterExhaustedError, not infinite loop.'''

    domain = ParamDomain({'a': [1, 2], 'b': ['x', 'y']})
    strategy = _InfiniteStubStrategy(domain)
    msq = MSQ(strategy, domain, max_filter_retries=5)

    msq.remove_custom(lambda c: False)

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

    '''Priority queue items count against trim budget.'''

    msq, _ = _make_msq()
    msq.trim(2)
    msq.inject({'a': 10, 'b': 'z'})

    assert msq.remaining_count() == 2 + 1

    next(msq)
    assert msq.remaining_count() == 1 + 0


def test_remaining_count_known():

    '''Finite strategy, n_permutations, and trim all produce known counts.'''

    # Finite strategy: uses total_combinations
    msq, _ = _make_msq()
    assert msq.remaining_count() == 4
    next(msq)
    assert msq.remaining_count() == 3

    # n_permutations takes precedence over total_combinations
    msq2, _ = _make_msq(n_permutations=100)
    assert msq2.remaining_count() == 100
    next(msq2)
    assert msq2.remaining_count() == 99

    # Infinite strategy with n_permutations
    domain = ParamDomain({'a': [1, 2], 'b': ['x', 'y']})
    strategy = _InfiniteStubStrategy(domain)
    msq3 = MSQ(strategy, domain, n_permutations=50)
    assert msq3.remaining_count() == 50
    next(msq3)
    assert msq3.remaining_count() == 49


def test_remaining_count_unknown():

    '''Infinite strategy without budget returns None.'''

    domain = ParamDomain({'a': [1, 2], 'b': ['x', 'y']})
    strategy = _InfiniteStubStrategy(domain)
    msq = MSQ(strategy, domain)

    assert msq.remaining_count() is None


def test_distribution():

    '''Uniform estimate of per-value counts from remaining runs.'''

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

    # Infinite without budget returns empty
    domain = ParamDomain({'a': [1, 2], 'b': ['x', 'y']})
    strategy = _InfiniteStubStrategy(domain)
    msq3 = MSQ(strategy, domain)
    assert msq3.distribution() == {}


def test_intervention_log():

    msq, _ = _make_msq()

    msq.remove_is('a', 2)
    msq.inject_value('b', 'z')

    log = msq.intervention_log
    assert len(log) == 2
    assert log[0]['operation'] == 'remove_is'
    assert log[1]['operation'] == 'inject_value'


def test_get_set_state():

    '''Checkpoint round-trip preserves yielded_count, n_permutations, priority queue.'''

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
    assert msq2.remaining_count() == 98 + 1
