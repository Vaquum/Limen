"""Finite seeded index traversal and immutable resume identity."""
import copy
import hashlib
import itertools
import json
import math
import random
import statistics
import tracemalloc
from unittest.mock import patch

import pytest

from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.experiment.param_search._grid_permutation import (
    lcg_map, lcg_params, permuted_index, domain_digest,
)


def _stream(params, count):
    domain = ParamDomain(params)
    return list(MSQ(GridStrategy(domain, shuffle=True, seed=42), domain,
                    n_permutations=count))


@pytest.mark.parametrize('size', [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                                  63, 64, 65, 120, 255, 256, 257, 385, 1440])
def test_seeded_permutation_contract(size):
    for seed in (0, 7, 42, -123, 2**65):
        order = [permuted_index(index, size, seed) for index in range(size)]
        assert sorted(order) == list(range(size))
        assert order == [permuted_index(index, size, seed) for index in range(size)]
    strategy = GridStrategy(ParamDomain({'candidate': list(range(size))}), seed=42, shuffle=True)
    assert len(list(strategy)) == size
    with pytest.raises(StopIteration):
        next(strategy)


def test_run_budget_prefix_and_existing_candidate_identity():
    params = {'z': [0.1, 0.3, 1.0], 'alpha': [1, 2, 3, 4, 5], '费用': [0.0, 0.001]}
    short, complete = _stream(params, 7), _stream(params, 30)
    assert short == complete[:7]
    assert complete == _stream(dict(reversed(list(params.items()))), 30)
    ordinary = list(GridStrategy(ParamDomain(params)))
    assert {row['_id'] for row in complete} == {row['_id'] for row in ordinary}
    assert [row['_round_index'] for row in complete] == list(range(30))
    assert [row['_generation_index'] for row in complete] == list(range(30))
    assert len({row['_id'] for row in complete}) == 30
    assert list(GridStrategy(ParamDomain(params))._index_to_combo(0)) == sorted(params)
    assert GridStrategy(ParamDomain(params))._index_to_combo(0) == {'alpha': 1, 'z': 0.1, '费用': 0.0}


def test_ordered_domain_digest_retains_existing_json_identity():
    params = {'z': [1.0, -0.0], 'a': [(1, 2), None, {'label': '观察'}]}
    strategy = GridStrategy(ParamDomain(params), seed=42, shuffle=True)
    assert domain_digest(params) == strategy.compute_param_hash(params)
    assert domain_digest(params) == hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()
    assert domain_digest(params) == domain_digest(json.loads(json.dumps(params)))
    assert domain_digest(params) != domain_digest({**params, 'z': [-0.0, 1.0]})
    assert domain_digest({'a': [1]}) != domain_digest({'a': [1.0]})
    assert strategy.get_state()['domain_digest'] == domain_digest(params)


def test_versioned_resume_preserves_stream_and_native_level_types():
    params = {'alpha': [0.1, 0.3, 1.0], 'window': [(1, 2), (3, 4)], 'fee': [0, 1, 2]}
    original = GridStrategy(ParamDomain(params), shuffle=True, seed=-7)
    for _ in range(7):
        original.mark_seen(next(original))
    state = json.loads(json.dumps(original.get_state()))
    restored = GridStrategy(ParamDomain(params), shuffle=True, seed=-7)
    restored.set_state(state)
    assert restored.get_state() == state
    actual, expected = list(restored), list(original)
    assert actual == expected
    assert all(type(row['window']) is tuple for row in actual)
    assert len(actual) == 11
    assert state['permutation_version'] == 2


@pytest.mark.parametrize('versionless', [False, True])
def test_seedless_in_memory_restore_keeps_existing_permutation(versionless):
    strategy = GridStrategy(ParamDomain({'candidate': list(range(385))}), shuffle=True)
    for _ in range(11):
        next(strategy)
    state = copy.deepcopy(strategy.get_state())
    if versionless:
        del state['permutation_version']
        del state['domain_digest']
    expected = list(strategy)
    with patch('limen.experiment.param_search.grid_strategy.lcg_params',
               side_effect=AssertionError('in-memory restore regenerated the seedless permutation')) as regenerate:
        strategy.set_state(state)
        assert list(strategy) == expected
    regenerate.assert_not_called()


def test_versionless_checkpoint_replays_original_affine_order():
    domain = ParamDomain({'candidate': list(range(385))})
    strategy = GridStrategy(domain, shuffle=True, seed=42)
    strategy.set_state({'current_index': 150, 'generated_count': 150,
                        'shuffle': True, 'seed': 42})
    assert strategy.get_state()['permutation_version'] == 1
    assert [row['candidate'] for row in strategy] == [(237 * index + 327) % 385 for index in range(150, 385)]
    saved = strategy.get_state()
    again = GridStrategy(domain, shuffle=True, seed=42)
    again.set_state(saved)
    assert list(again) == []
    domain.remove_value('candidate', 384)
    assert strategy.get_state()['permutation_version'] == 1


@pytest.mark.parametrize('change', [
    {'permutation_version': 3}, {'permutation_version': True}, {'seed': 43},
    {'seed': None}, {'seed': True}, {'shuffle': False}, {'shuffle': None},
    {'domain_digest': None}, {'domain_digest': 'changed'},
    {'current_index': -1}, {'current_index': 31}, {'current_index': True},
    {'generated_count': -1}, {'generated_count': 2}, {'generated_count': 1.0},
])
def test_rejected_state_does_not_mutate_strategy(change):
    strategy = GridStrategy(ParamDomain({'candidate': list(range(30))}), seed=42, shuffle=True)
    next(strategy)
    before = strategy.get_state()
    with pytest.raises(ValueError, match=r'version|seed|shuffle|domain|counters'):
        strategy.set_state({**before, **change})
    assert strategy.get_state() == before


def test_changed_declared_level_order_rejects_restore():
    original = GridStrategy(ParamDomain({'candidate': [1, 2, 3]}), shuffle=True, seed=42)
    reversed_levels = GridStrategy(ParamDomain({'candidate': [3, 2, 1]}), shuffle=True, seed=42)
    before = copy.deepcopy(reversed_levels.get_state())
    with pytest.raises(ValueError, match='ordered domain'):
        reversed_levels.set_state(original.get_state())
    assert reversed_levels.get_state() == before
    with pytest.raises(ValueError, match='shuffle'):
        GridStrategy(ParamDomain({'candidate': [1, 2, 3]})).set_state(original.get_state())


@pytest.mark.parametrize('options', [{'shuffle': 1}, {'shuffle': 'true'}, {'shuffle': None},
                                    {'seed': True}, {'seed': 1.5}, {'seed': '42'}])
def test_native_options_are_not_coerced(options):
    with pytest.raises(ValueError, match=r'shuffle|seed'):
        GridStrategy(ParamDomain({'candidate': [1]}), **options)


def test_legacy_seedless_and_unshuffled_routes_remain_available():
    domain = ParamDomain({'a': [1, 2], 'b': ['x', 'y']})
    assert list(GridStrategy(domain)) == list(GridStrategy(domain, seed=42))
    legacy = GridStrategy(domain, shuffle=True)
    assert legacy.get_state()['permutation_version'] == 1
    assert {row['_id'] for row in legacy} == {row['_id'] for row in GridStrategy(domain)}
    old = GridStrategy(domain)
    old.set_state({'current_index': 1, 'generated_count': 1, 'shuffle': False, 'seed': None})
    assert len(list(old)) == 3
    with pytest.raises(ValueError, match='choices 0 and 1 collide under the execution identity'):
        ParamDomain({'candidate': [1, 1, 2]})


def test_large_domain_startup_is_lazy():
    params = {f'parameter_{index:03}': [0, 1] for index in range(96)}
    tracemalloc.start()
    try:
        with patch.object(GridStrategy, '_index_to_combo', side_effect=AssertionError('eager candidate materialization')):
            strategy = GridStrategy(ParamDomain(params), seed=42, shuffle=True)
        assert strategy._total == 2**96
        assert strategy._seen == set()
        prefix = [next(strategy) for _ in range(8)]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len({row['_id'] for row in prefix}) == 8
    assert peak < 2 * 1024**2


def _digits(index, sizes):
    result = []
    for size in sizes:
        index, offset = divmod(index, size)
        result.append(offset)
    return result


def _coverage(order, sizes):
    rows = [_digits(index, sizes) for index in order]
    marginal = [len({row[axis] for row in rows}) / size for axis, size in enumerate(sizes)]
    joint = [len({(row[left], row[right]) for row in rows}) / (sizes[left] * sizes[right])
             for left, right in itertools.combinations(range(len(sizes)), 2)]
    return marginal, joint


def _nested_levels(sizes, seed, count):
    rng = random.Random(seed)
    levels = [rng.sample(range(size), size) for size in sizes]
    strides = [math.prod(sizes[:axis]) for axis in range(len(sizes))]
    return [sum(levels[axis][offset] * strides[axis]
                for axis, offset in enumerate(_digits(index, sizes))) for index in range(count)]


def test_measured_early_joint_coverage_with_negative_controls():
    affine_failures = []
    for sizes in ((6, 3, 5, 4, 4), (3, 5, 2, 7, 5)):
        count, total = 2 * max(a * b for a, b in itertools.combinations(sizes, 2)), math.prod(sizes)
        measured, affine, nested = [], [], []
        for seed in range(8):
            order = [permuted_index(index, total, seed) for index in range(count)]
            marginal, joint = _coverage(order, sizes)
            assert min(marginal) == 1.0
            measured.append(joint)
            affine.append(_coverage([lcg_map(index, total, *lcg_params(total, seed))
                                     for index in range(count)], sizes)[1])
            nested.append(_coverage(_nested_levels(sizes, seed, count), sizes)[1])
        means = [statistics.mean(projection) for projection in zip(*measured, strict=True)]
        assert min(itertools.chain.from_iterable(measured)) >= .65
        assert min(means) >= .8
        affine_failures.append(any(statistics.mean(projection) < .8
                                   for projection in zip(*affine, strict=True)))
        assert any(statistics.mean(projection) < .8 for projection in zip(*nested, strict=True))
    assert any(affine_failures)


def test_seed_orders_are_not_rotations_of_one_cycle():
    first = [permuted_index(index, 385, 1) for index in range(385)]
    second = [permuted_index(index, 385, 2) for index in range(385)]
    start = first.index(second[0])
    assert second != first[start:] + first[:start]
    assert [permuted_index(index, 385, 42) for index in range(16)] == [
        379, 98, 54, 7, 346, 155, 79, 87, 375, 112, 364, 382, 225, 177, 216, 146,
    ]
