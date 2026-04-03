from limen.experiment.grid_strategy import GridStrategy
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.random_strategy import RandomStrategy
from limen.experiment.strategy_registry import STRATEGY_REGISTRY


def _small_domain():
    return ParamDomain({'a': [1, 2], 'b': ['x', 'y']})


def _medium_domain():
    return ParamDomain({'a': list(range(100)), 'b': list(range(100))})


# --- RandomStrategy ---


def test_random_basic_iteration():

    domain = _small_domain()
    strategy = RandomStrategy(domain, seed=42)
    combo = next(strategy)
    assert 'a' in combo
    assert 'b' in combo
    assert combo['a'] in [1, 2]
    assert combo['b'] in ['x', 'y']


def test_random_is_infinite():

    domain = _medium_domain()
    strategy = RandomStrategy(domain, seed=42)
    assert strategy.is_finite is False
    for _ in range(50):
        next(strategy)


def test_random_seeded_reproducibility():

    domain = _medium_domain()
    s1 = RandomStrategy(domain, seed=99)
    s2 = RandomStrategy(domain, seed=99)
    seq1 = [next(s1) for _ in range(10)]
    seq2 = [next(s2) for _ in range(10)]
    assert seq1 == seq2


def test_random_different_seeds():

    domain = _medium_domain()
    s1 = RandomStrategy(domain, seed=1)
    s2 = RandomStrategy(domain, seed=2)
    seq1 = [next(s1) for _ in range(20)]
    seq2 = [next(s2) for _ in range(20)]
    assert seq1 != seq2


def test_random_param_hash():

    domain = _small_domain()
    strategy = RandomStrategy(domain, seed=42)
    assert strategy.last_param_hash is None
    next(strategy)
    h = strategy.last_param_hash
    assert h is not None
    assert len(h) == 16
    assert all(c in '0123456789abcdef' for c in h)


def test_random_dedup_skips_seen():

    domain = ParamDomain({'a': [1], 'b': [1]})
    strategy = RandomStrategy(domain, seed=42)
    next(strategy)
    h = strategy.last_param_hash
    assert h is not None

    try:
        next(strategy)
        assert False, 'Expected RuntimeError from dedup exhaustion'
    except RuntimeError as e:
        assert 'novel' in str(e).lower()


def test_random_get_set_state():

    domain = _medium_domain()
    strategy = RandomStrategy(domain, seed=42)
    for _ in range(5):
        next(strategy)

    state = strategy.get_state()
    combo_after_save = next(strategy)

    strategy2 = RandomStrategy(domain, seed=0)
    strategy2.set_state(state)
    combo_after_restore = next(strategy2)
    assert combo_after_save == combo_after_restore


def test_random_rebuild_seen():

    domain = _small_domain()
    strategy = RandomStrategy(domain, seed=42)
    combo = next(strategy)
    h = strategy._compute_param_hash(combo)

    strategy2 = RandomStrategy(domain, seed=42)
    strategy2.rebuild_seen([h])
    assert h in strategy2._seen


def test_random_with_msq():

    domain = _medium_domain()
    strategy = RandomStrategy(domain, seed=42)
    msq = MSQ(strategy, domain, n_permutations=3)

    combos = list(msq)
    assert len(combos) == 3
    for combo in combos:
        assert '_id' in combo
        assert '_injected' in combo
        assert '_param_hash' in combo
        assert '_generation_index' in combo
        assert '_search_strategy' in combo
        assert combo['_search_strategy'] == 'RandomStrategy'


def test_random_large_domain():

    from limen.sfd.foundational_sfd import logreg_binary as sfd
    domain = ParamDomain(sfd.params())
    strategy = RandomStrategy(domain, seed=42)
    combos = [next(strategy) for _ in range(100)]
    assert len(combos) == 100


# --- GridStrategy ---


def test_grid_exhaustive_small_space():

    domain = _small_domain()
    strategy = GridStrategy(domain)
    combos = list(strategy)
    assert len(combos) == 4


def test_grid_is_finite():

    domain = _small_domain()
    strategy = GridStrategy(domain)
    assert strategy.is_finite is True


def test_grid_no_duplicates():

    domain = _small_domain()
    strategy = GridStrategy(domain)
    combos = list(strategy)
    combo_tuples = [tuple(sorted(c.items())) for c in combos]
    assert len(set(combo_tuples)) == len(combos)


def test_grid_covers_full_space():

    domain = _small_domain()
    strategy = GridStrategy(domain)
    combos = list(strategy)
    combo_set = {tuple(sorted(c.items())) for c in combos}
    expected = {
        (('a', 1), ('b', 'x')),
        (('a', 1), ('b', 'y')),
        (('a', 2), ('b', 'x')),
        (('a', 2), ('b', 'y')),
    }
    assert combo_set == expected


def test_grid_deterministic_order():

    domain = _small_domain()
    s1 = GridStrategy(domain)
    s2 = GridStrategy(domain)
    assert list(s1) == list(s2)


def test_grid_shuffle_covers_full_space():

    domain = _small_domain()
    strategy = GridStrategy(domain, shuffle=True, seed=42)
    combos = list(strategy)
    combo_set = {tuple(sorted(c.items())) for c in combos}
    assert len(combo_set) == 4


def test_grid_shuffle_different_order():

    domain = _small_domain()
    sequential = [tuple(sorted(c.items())) for c in GridStrategy(domain)]
    shuffled = [tuple(sorted(c.items())) for c in GridStrategy(domain, shuffle=True, seed=42)]
    assert set(sequential) == set(shuffled)
    assert sequential != shuffled


def test_grid_shuffle_seeded_reproducibility():

    domain = _small_domain()
    s1 = GridStrategy(domain, shuffle=True, seed=42)
    s2 = GridStrategy(domain, shuffle=True, seed=42)
    assert list(s1) == list(s2)


def test_grid_get_set_state():

    domain = _small_domain()
    strategy = GridStrategy(domain)
    next(strategy)
    next(strategy)

    state = strategy.get_state()
    combo_after_save = next(strategy)

    strategy2 = GridStrategy(domain)
    strategy2.set_state(state)
    combo_after_restore = next(strategy2)
    assert combo_after_save == combo_after_restore


def test_grid_on_domain_changed():

    domain = ParamDomain({'a': [1, 2, 3], 'b': ['x', 'y']})
    strategy = GridStrategy(domain)
    next(strategy)
    next(strategy)
    assert strategy.generated_count == 2

    domain.remove_value('a', 3)
    combos = list(strategy)
    combo_set = {tuple(sorted(c.items())) for c in combos}
    expected = {
        (('a', 1), ('b', 'x')),
        (('a', 1), ('b', 'y')),
        (('a', 2), ('b', 'x')),
        (('a', 2), ('b', 'y')),
    }
    assert combo_set == expected


def test_grid_large_space_shuffle_no_oom():

    from limen.sfd.foundational_sfd import logreg_binary as sfd
    domain = ParamDomain(sfd.params())
    strategy = GridStrategy(domain, shuffle=True, seed=42)
    combos = [next(strategy) for _ in range(100)]
    assert len(combos) == 100
    combo_tuples = [tuple(sorted(c.items())) for c in combos]
    assert len(set(combo_tuples)) == 100


def test_grid_with_msq():

    domain = _small_domain()
    strategy = GridStrategy(domain)
    msq = MSQ(strategy, domain, n_permutations=4)
    combos = list(msq)
    assert len(combos) == 4
    for combo in combos:
        assert combo['_search_strategy'] == 'GridStrategy'
        assert combo['_param_hash'] is None


# --- STRATEGY_REGISTRY ---


def test_strategy_registry_contains_all():

    assert 'random' in STRATEGY_REGISTRY
    assert 'grid' in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY['random'] is RandomStrategy
    assert STRATEGY_REGISTRY['grid'] is GridStrategy


def test_strategy_registry_instantiation():

    domain = _small_domain()
    strategy = STRATEGY_REGISTRY['random'](domain, seed=42)
    combo = next(strategy)
    assert 'a' in combo


# --- _compute_param_hash ---


def test_param_hash_deterministic():

    domain = _small_domain()
    strategy = RandomStrategy(domain, seed=42)
    combo = {'a': 1, 'b': 'x'}
    h1 = strategy._compute_param_hash(combo)
    h2 = strategy._compute_param_hash(combo)
    assert h1 == h2


def test_param_hash_key_order_independent():

    domain = _small_domain()
    strategy = RandomStrategy(domain, seed=42)
    h1 = strategy._compute_param_hash({'a': 1, 'b': 'x'})
    h2 = strategy._compute_param_hash({'b': 'x', 'a': 1})
    assert h1 == h2


def test_param_hash_different_combos_differ():

    domain = _small_domain()
    strategy = RandomStrategy(domain, seed=42)
    h1 = strategy._compute_param_hash({'a': 1, 'b': 'x'})
    h2 = strategy._compute_param_hash({'a': 2, 'b': 'x'})
    assert h1 != h2
