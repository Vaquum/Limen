import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from limen.experiment import UniversalExperimentLoop, _round_storage
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy, RandomStrategy
from limen.yaml import CompiledSFD
from tests.test_parameter_identity_admission import Observer
from tests.test_record_model_outputs import _config, recorded_source


@pytest.mark.parametrize('value', [1, 1.0, True], ids=['int', 'float', 'bool'])
@pytest.mark.parametrize('operation', ['remove', 'keep'])
def test_exact_choice_mutations_preserve_other_execution_identities(value, operation):
    domain = ParamDomain({'choice': [1, 1.0, True]})
    observer = Observer()
    domain.add_observer(observer)
    if operation == 'remove':
        assert domain.remove_value('choice', value)
        expected = [kind for kind in (int, float, bool) if kind is not type(value)]
    else:
        assert domain.keep_values('choice', [value]) == 2
        expected = [type(value)]
    assert [type(item) for item in domain.values_for('choice')] == expected
    assert domain.version == 1 and len(observer.calls) == 1
    for candidate in (1, 1.0, True):
        assert domain.is_valid_combination({'choice': candidate}) is (type(candidate) in expected)


def test_distinct_equal_numeric_injection_and_absent_removal_preserve_notifications():
    domain = ParamDomain({'choice': [1]})
    observer = Observer()
    domain.add_observer(observer)
    assert not domain.is_valid_combination({'choice': 1.0})
    assert not domain.is_valid_combination({'choice': True})
    assert domain.remove_value('choice', True) is False
    assert domain.version == 0 and not observer.calls
    assert domain.inject_value('choice', 1.0)
    assert domain.inject_value('choice', True)
    assert not domain.inject_value('choice', 1.0)
    assert [type(value) for value in domain.values_for('choice')] == [int, float, bool]
    assert domain.version == 2 and len(observer.calls) == 2


@pytest.mark.parametrize('operation', ['remove', 'keep', 'inject', 'membership'])
def test_representational_identity_uses_the_original_choice(operation):
    same, other = Path('same'), Path('other')
    domain = ParamDomain({'choice': [same, other]})
    observer = Observer()
    domain.add_observer(observer)
    if operation == 'remove':
        assert domain.remove_value('choice', 'same')
        assert domain.values_for('choice') == [other]
    elif operation == 'keep':
        assert domain.keep_values('choice', ['same']) == 1
        assert domain.values_for('choice')[0] is same
    elif operation == 'inject':
        assert domain.inject_value('choice', 'same') is False
    else:
        assert domain.is_valid_combination({'choice': 'same'})
    changed = operation in ('remove', 'keep')
    assert domain.version == int(changed) and len(observer.calls) == int(changed)


@pytest.mark.parametrize('operation', ['remove_values_ge', 'remove_values_le', 'keep_between'])
def test_numeric_ranges_keep_their_existing_comparison_semantics(operation):
    domain = ParamDomain({'choice': [0, 1, 1.0, True, 2]})
    args = (1, 1) if operation == 'keep_between' else (1,)
    getattr(domain, operation)('choice', *args)
    expected = [1, 1.0, True] if operation == 'keep_between' else [0] if operation == 'remove_values_ge' else [2]
    assert domain.values_for('choice') == expected
    assert [type(value) for value in domain.values_for('choice')] == [type(value) for value in expected]


def test_priority_combination_injection_retains_explicit_queue_intent():
    domain = ParamDomain({'choice': [1]})
    queue = MSQ(GridStrategy(domain), domain, n_permutations=2)
    queue.inject({'choice': True}, prioritize=True)
    first, second = next(queue), next(queue)
    assert first['_injected'] and type(first['choice']) is bool
    assert not second['_injected'] and type(second['choice']) is int
    assert first['_id'] != second['_id'] and domain.version == 0


def test_noop_injection_preserves_feedback_source_isolation(tmp_path):
    domain = ParamDomain({'choice': [1]})
    observer = Observer()
    domain.add_observer(observer)
    strategy = GridStrategy(domain)
    queue = MSQ(strategy, domain, n_permutations=3)

    def callback(_log, msq):
        assert msq.inject_value('choice', 1) is False
        raise RuntimeError('Recorded feedback isolation control')

    path = tmp_path / 'interventions.json'
    path.write_text(json.dumps([{'op': 'inject_value', 'param': 'choice', 'value': value} for value in (1.0, True)]))
    audit = tmp_path / 'audit.jsonl'
    controller = FeedbackController(intra_callback=callback, intervention_path=path, audit_log_path=audit)
    controller.trigger(recorded_source().head(32), queue, strategy, 1)
    assert [type(value) for value in domain.values_for('choice')] == [int, float, bool]
    assert domain.version == 2 and len(observer.calls) == 2
    assert [type(row['choice']) for row in queue] == [int, float, bool]
    entry = json.loads(audit.read_text())
    assert len(entry['errors']) == 1 and entry['errors'][0]['source'] == 'intra_callback'


@pytest.mark.parametrize('strategy_type', [GridStrategy, RandomStrategy])
def test_recorded_feedback_growth_preserves_typed_candidates_and_resume(tmp_path, strategy_type):
    config = _config('logreg')
    manifest = CompiledSFD(config).manifest()
    params = {'C': [0.1], 'max_iter': [1000], 'choice': [1, True]}
    recording = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)

    def execute(directory, *, resume=False, stop=False):
        domain = ParamDomain(params)
        strategy = strategy_type(domain, seed=42)
        observed = []

        def model(data, round_params):
            penalty = {int: 0.1, float: 1.0, bool: 10.0}[type(round_params['choice'])]
            result = manifest.run_model(data, round_params | {'C': penalty})
            assert penalty == result['_model'].model.C and len(result['_preds']) > 0
            observed.append((type(round_params['choice']), penalty))
            return result

        def feedback(log, queue):
            if log.height == 2:
                assert queue.inject_value('choice', 1.0)
                if stop:
                    loop._shutdown_requested = True

        sfd = SimpleNamespace(__name__=__name__, params=lambda: params, prep=manifest.prepare_data, model=model)
        loop = UniversalExperimentLoop(sfd=sfd, data=recorded_source(), experiment_dir=directory,
            search_strategy=strategy, checkpoint_interval=1, feedback_interval=1, intra_callback=feedback)
        loop.run('typed_feedback', n_permutations=3, prep_each_round=True, resume=resume,
            progress_bar=False, recording=recording)
        with _round_storage.open_text(_round_storage.round_path(directory)) as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        return loop, observed, records

    full, expected, full_records = execute(tmp_path / 'full')
    assert full.experiment_log.height == 3
    assert {kind for kind, _ in expected} == {int, float, bool}
    first, prefix, _ = execute(tmp_path / 'resumed', stop=True)
    assert first.interrupted and first.experiment_log.height == 2
    checkpoint = json.loads((tmp_path / 'resumed/checkpoint.json').read_text())
    assert [type(value) for value in checkpoint['domain_state']['choice']] == [int, bool, float]
    resumed, suffix, actual_records = execute(tmp_path / 'resumed', resume=True)
    assert resumed.experiment_log.height == 3 and prefix + suffix == expected
    assert [type(row['round_params']['choice']) for row in actual_records] == [kind for kind, _ in expected]
    fields = ('round_id', 'round_params', 'preds', 'alignment')
    assert [{key: row[key] for key in fields} for row in actual_records] == [{key: row[key] for key in fields} for row in full_records]
    assert pl.read_csv(tmp_path / 'resumed/results.csv').height == 3
