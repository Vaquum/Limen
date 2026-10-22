import copy
import csv

import pytest

from limen.experiment._filter_resume import validate_filter_resume
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.experiment.reducer.filter_types import FILTER_BUILDERS, FILTER_SAMPLE


def _queue(*, cap=None):
    domain = ParamDomain({'C': [0.1, 0.3, 0.5]})
    strategy = GridStrategy(domain)
    return MSQ(strategy, domain, n_permutations=cap), domain, strategy


def _checkpoint(queue, domain):
    return {
        'metadata': {'experiment_round': queue.yielded_count - 1},
        'msq_state': copy.deepcopy(queue.get_state()),
        'domain_state': domain.get_state(),
    }


def _prefix(tmp_path, rows):
    path = tmp_path / 'results.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['id', '_id', '_param_hash'])
        writer.writeheader()
        for row in rows:
            writer.writerow({'id': row['_id'], '_id': rows[0]['_id'],
                             '_param_hash': row['_id']})
    return path


@pytest.mark.parametrize(('patch', 'message'), [
    ({'custom_filters_count': -1}, 'custom filter count'),
    ({'named_filter_keys': [None]}, 'named filter keys'),
    ({'named_filter_descriptors': []}, 'named filter descriptors'),
    ({'named_filter_descriptors': {'bad': ['sample']}}, 'named filter descriptor'),
    ({'named_filter_descriptors': {'bad': [None, {}]}}, 'filter type'),
    ({'named_filter_descriptors': {'bad': ['sample', []]}}, 'filter parameters'),
])
def test_malformed_filter_metadata_fails_with_value_error(patch, message):
    queue, _, _ = _queue()
    state = queue.get_state() | patch
    with pytest.raises(ValueError, match=message):
        queue.set_state(state)


@pytest.mark.parametrize('descriptor', [['unsupported', {}], ['exclude_value', {}]])
def test_unrestorable_descriptor_refuses_continuation(tmp_path, descriptor):
    queue, domain, strategy = _queue()
    rows = [next(queue)]
    queue.set_filter('invalid', lambda combo: combo['C'] == 0.3)
    checkpoint = _checkpoint(queue, domain)
    checkpoint['msq_state']['named_filter_descriptors']['invalid'] = descriptor
    with pytest.raises(ValueError, match='invalid'):
        validate_filter_resume(checkpoint, strategy, _prefix(tmp_path, rows), parallel=False)


@pytest.mark.parametrize('named', [False, True])
def test_wiped_filter_metadata_still_refuses_from_history(tmp_path, named):
    queue, domain, strategy = _queue()
    rows = [next(queue)]

    def condition(combo):
        return combo['C'] == 0.3

    if named:
        queue.set_filter('no_middle', condition)
    else:
        queue.remove_custom(condition)
    checkpoint = _checkpoint(queue, domain)
    checkpoint['msq_state'].update(custom_filters_count=0, named_filter_keys=[],
                                   named_filter_descriptors={})
    before = copy.deepcopy(checkpoint)
    with pytest.raises(ValueError, match='no_middle' if named else '1 custom filter'):
        validate_filter_resume(checkpoint, strategy, _prefix(tmp_path, rows), parallel=False)
    assert checkpoint == before


def test_cleared_named_filter_is_not_resurrected_from_history(tmp_path):
    queue, domain, strategy = _queue()
    queue.set_filter('no_middle', lambda combo: combo['C'] == 0.3)
    queue.clear_filter('no_middle')
    validate_filter_resume(_checkpoint(queue, domain), strategy,
                           _prefix(tmp_path, []), parallel=False)


@pytest.mark.parametrize('completion', ['cap', 'trim'])
def test_completed_cap_or_trim_precedes_pending_injections(tmp_path, completion):
    queue, domain, strategy = _queue(cap=1 if completion == 'cap' else None)
    queue.remove_custom(lambda combo: combo['C'] == 0.3)
    if completion == 'trim':
        queue.trim(1)
    rows = [next(queue)]
    queue.inject({'C': 0.5})
    checkpoint = _checkpoint(queue, domain)
    path = _prefix(tmp_path, rows)
    validate_filter_resume(checkpoint, strategy, path, parallel=False)
    checkpoint['msq_state']['n_permutations' if completion == 'cap' else 'trim_budget'] = 2
    with pytest.raises(ValueError, match='1 custom filter'):
        validate_filter_resume(checkpoint, strategy, path, parallel=False)


@pytest.mark.parametrize('parallel', [False, True])
def test_seen_grid_exhaustion_is_checked_without_mutating_live_queue(tmp_path, parallel):
    queue, domain, strategy = _queue()
    rows = list(queue)
    queue.remove_custom(lambda combo: combo['C'] == 0.3)
    queue.remove_is('C', 0.5)
    checkpoint = _checkpoint(queue, domain)
    before = copy.deepcopy(queue.get_state())
    validate_filter_resume(checkpoint, strategy, _prefix(tmp_path, rows), parallel=parallel)
    assert queue.get_state() == before
    assert checkpoint['msq_state'] == before
    queue.inject({'C': 0.3})
    with pytest.raises(ValueError, match='1 custom filter'):
        validate_filter_resume(_checkpoint(queue, domain), strategy,
                               _prefix(tmp_path, rows), parallel=parallel)


def test_sampling_hash_error_precedes_completed_filter_loss(tmp_path):
    queue, domain, strategy = _queue(cap=1)
    rows = [next(queue)]
    queue.remove_custom(lambda combo: combo['C'] == 0.3)
    params = {'param': 'C', 'value': 0.3, 'fraction': 0.5}
    queue.set_filter('sample_C', FILTER_BUILDERS[FILTER_SAMPLE](params),
                     filter_type=FILTER_SAMPLE, filter_params=params)
    checkpoint = _checkpoint(queue, domain)
    checkpoint['msq_state']['named_filter_descriptors']['sample_C'][1].pop('hash_rule')
    with pytest.raises(ValueError, match='sampling hash rule'):
        validate_filter_resume(checkpoint, strategy, _prefix(tmp_path, rows), parallel=False)


def test_standalone_restore_allows_reregistering_custom_filter():
    original, _, _ = _queue()
    original.remove_custom(lambda combo: combo['C'] == 0.3)
    assert next(original)['C'] == 0.1
    restored, _, _ = _queue()
    with pytest.warns(UserWarning, match='Checkpoint had non-restorable filters'):
        restored.set_state(original.get_state())
    restored.remove_custom(lambda combo: combo['C'] == 0.3)
    assert next(restored)['C'] == 0.5
