import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest

from limen.experiment.feedback_controller import FeedbackController
from tests.stubs.stubs import StubPruningStrategy, make_msq


@pytest.fixture(scope='module')
def recorded_log():
    path = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
    return pl.read_parquet(path, n_rows=32)


def assert_audit_error(actual, source, message, exception):
    assert set(actual) == {'source', 'error', 'traceback'}
    assert actual['source'] == source
    assert actual['error'] == message
    assert exception in actual['traceback']


def test_should_trigger_interval():

    fc = FeedbackController(feedback_interval=5)

    assert fc.should_trigger(0) is False
    assert fc.should_trigger(1) is False
    assert fc.should_trigger(4) is False
    assert fc.should_trigger(5) is True
    assert fc.should_trigger(10) is True
    assert fc.should_trigger(7) is False


def test_trigger_pruning_strategy():

    msq, strategy, domain = make_msq()
    ps = StubPruningStrategy(interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3},
    ])
    fc = FeedbackController(
        feedback_interval=5,
        pruning_strategies=[ps],
    )

    result = fc.trigger(None, msq, strategy, 5)

    assert len(result) == 1
    assert result[0]['op'] == 'remove_is'
    assert result[0]['source'] == 'StubPruningStrategy'
    assert 3 not in domain.values_for('a')


def test_trigger_inactive_pruning_strategy():

    msq, strategy, domain = make_msq()
    ps = StubPruningStrategy(active=False, interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3},
    ])
    fc = FeedbackController(
        feedback_interval=5,
        pruning_strategies=[ps],
    )

    result = fc.trigger(None, msq, strategy, 5)

    assert result == []
    assert 3 in domain.values_for('a')


def test_trigger_intra_callback():

    msq, strategy, domain = make_msq()

    def callback(_log, msq):
        msq.remove_is('a', 3)

    fc = FeedbackController(
        feedback_interval=5,
        intra_callback=callback,
    )

    result = fc.trigger(None, msq, strategy, 5)

    assert len(result) == 1
    assert result[0]['source'] == 'intra_callback'
    assert 3 not in domain.values_for('a')


def test_apply_intervention_dispatch():

    msq, _, domain = make_msq()

    FeedbackController._apply_intervention(
        msq, {'op': 'remove_is', 'param': 'a', 'value': 3},
    )
    assert 3 not in domain.values_for('a')

    FeedbackController._apply_intervention(
        msq, {'op': 'inject_value', 'param': 'a', 'value': 99},
    )
    assert 99 in domain.values_for('a')

    FeedbackController._apply_intervention(
        msq, {'op': 'trim', 'target_count': 2},
    )
    assert msq.remaining_count() is not None


def test_apply_intervention_unknown_op():

    msq, _, _ = make_msq()

    try:
        FeedbackController._apply_intervention(
            msq, {'op': 'unknown_op'},
        )
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'unknown_op' in str(e)


def test_strategy_update_called():

    msq, strategy, domain = make_msq()
    ps = StubPruningStrategy(interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3},
    ])
    fc = FeedbackController(
        feedback_interval=5,
        pruning_strategies=[ps],
    )

    fc.trigger(None, msq, strategy, 5)

    assert any(i['operation'] == 'remove_is' for i in msq.intervention_log)
    assert 3 not in domain.params['a']


def test_trigger_intervention_file():

    msq, strategy, domain = make_msq()

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'interventions.json'
        path.write_text(json.dumps([
            {'op': 'remove_is', 'param': 'a', 'value': 3},
        ]))

        fc = FeedbackController(
            feedback_interval=5,
            intervention_path=path,
        )

        result = fc.trigger(None, msq, strategy, 5)

    assert len(result) == 1
    assert result[0]['source'] == 'intervention_file'
    assert 3 not in domain.values_for('a')


def test_file_polling_skips_unchanged():

    msq, strategy, _ = make_msq()

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'interventions.json'
        path.write_text(json.dumps([
            {'op': 'remove_is', 'param': 'a', 'value': 3},
        ]))

        fc = FeedbackController(
            feedback_interval=5,
            intervention_path=path,
        )

        result1 = fc.trigger(None, msq, strategy, 5)
        assert len(result1) == 1

        result2 = fc.trigger(None, msq, strategy, 10)
        assert result2 == []


def test_file_polling_detects_change():

    msq, strategy, domain = make_msq()

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'interventions.json'
        path.write_text(json.dumps([
            {'op': 'remove_is', 'param': 'a', 'value': 3},
        ]))

        fc = FeedbackController(
            feedback_interval=5,
            intervention_path=path,
        )

        result1 = fc.trigger(None, msq, strategy, 5)
        assert len(result1) == 1

        # Ensure mtime changes (filesystem granularity)
        time.sleep(0.05)
        path.write_text(json.dumps([
            {'op': 'inject_value', 'param': 'a', 'value': 99},
        ]))

        result2 = fc.trigger(None, msq, strategy, 10)
        assert len(result2) == 1
        assert result2[0]['op'] == 'inject_value'
        assert 99 in domain.values_for('a')


def test_file_missing_no_error():

    msq, strategy, _ = make_msq()

    with TemporaryDirectory() as tmpdir:
        missing = Path(tmpdir) / 'nonexistent.json'

        fc = FeedbackController(
            feedback_interval=5,
            intervention_path=missing,
        )

        result = fc.trigger(None, msq, strategy, 5)
        assert result == []


def test_file_invalid_json_no_error(recorded_log):
    msq, strategy, _ = make_msq()

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'bad.json'
        path.write_text('not valid json {{{')
        audit_path = Path(tmpdir) / 'audit.jsonl'
        fc = FeedbackController(feedback_interval=5, intervention_path=path, audit_log_path=audit_path)
        result = fc.trigger(recorded_log, msq, strategy, 5)
        entry = json.loads(audit_path.read_text())

    assert result == []
    assert len(entry['errors']) == 1
    assert_audit_error(entry['errors'][0], 'intervention_file',
                       'Expecting value: line 1 column 1 (char 0)', 'JSONDecodeError')


@pytest.mark.parametrize('mutate_first', [False, True])
def test_source_isolation(recorded_log, caplog, mutate_first):
    msq, strategy, domain = make_msq()
    calls = []

    def bad_callback(_log, queue):
        calls.append(1)
        if mutate_first:
            queue.remove_is('a', 2)
        raise RuntimeError('callback crashed')

    ps = StubPruningStrategy(interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3},
    ])
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'interventions.json'
        path.write_text(json.dumps([{'op': 'inject_value', 'param': 'a', 'value': 99}]))
        audit_path = Path(tmpdir) / 'audit.jsonl'
        fc = FeedbackController(
            feedback_interval=5, pruning_strategies=[ps],
            intra_callback=bad_callback, intervention_path=path, audit_log_path=audit_path,
        )
        result = fc.trigger(recorded_log, msq, strategy, 5)
        entry = json.loads(audit_path.read_text())

    assert calls == [1]
    assert 'Intra callback failed: callback crashed' in caplog.text
    assert 3 not in domain.values_for('a') and 99 in domain.values_for('a')
    assert (2 not in domain.values_for('a')) is mutate_first
    sources = ['StubPruningStrategy']
    if mutate_first:
        sources.append('intra_callback')
    sources.append('intervention_file')
    assert [item['source'] for item in result] == sources
    assert [item['source'] for item in entry['interventions']] == sources
    assert len(entry['errors']) == 1
    assert_audit_error(entry['errors'][0], 'intra_callback', 'callback crashed', 'RuntimeError')


def test_pruning_strategy_isolation(recorded_log, caplog):
    msq, strategy, domain = make_msq()
    calls = []

    class FailingStrategy(StubPruningStrategy):
        def analyze_and_intervene(self, _log, _msq):
            calls.append(1)
            raise RuntimeError('strategy crashed')

    ps_bad = FailingStrategy()
    ps_good = StubPruningStrategy(interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3},
    ])
    with TemporaryDirectory() as tmpdir:
        audit_path = Path(tmpdir) / 'audit.jsonl'
        fc = FeedbackController(
            feedback_interval=5, pruning_strategies=[ps_bad, ps_good], audit_log_path=audit_path,
        )
        result = fc.trigger(recorded_log, msq, strategy, 5)
        entry = json.loads(audit_path.read_text())

    assert calls == [1]
    assert 'PruningStrategy FailingStrategy failed: strategy crashed' in caplog.text
    assert [item['source'] for item in result] == ['StubPruningStrategy']
    assert 3 not in domain.values_for('a')
    assert len(entry['errors']) == 1
    assert_audit_error(entry['errors'][0], 'FailingStrategy', 'strategy crashed', 'RuntimeError')


def test_distinct_source_errors_keep_execution_order(recorded_log):
    msq, strategy, _ = make_msq()

    class FailingStrategy(StubPruningStrategy):
        def analyze_and_intervene(self, _log, _msq):
            raise ValueError('shared failure')

    def bad_callback(_log, _msq):
        raise ValueError('shared failure')

    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / 'interventions.json'
        file_path.write_text('not valid json {{{')
        audit_path = Path(tmpdir) / 'audit.jsonl'
        fc = FeedbackController(
            pruning_strategies=[FailingStrategy()], intra_callback=bad_callback,
            intervention_path=file_path, audit_log_path=audit_path,
        )
        assert fc.trigger(recorded_log, msq, strategy, 5) == []
        entry = json.loads(audit_path.read_text())

    assert len(entry['errors']) == 3
    assert_audit_error(entry['errors'][0], 'FailingStrategy', 'shared failure', 'ValueError')
    assert_audit_error(entry['errors'][1], 'intra_callback', 'shared failure', 'ValueError')
    assert_audit_error(entry['errors'][2], 'intervention_file',
                       'Expecting value: line 1 column 1 (char 0)', 'JSONDecodeError')


def test_out_of_boundary_errors_still_propagate(recorded_log):
    msq, strategy, _ = make_msq()

    def bad_callback(_log, _msq):
        raise AttributeError('outside feedback boundary')

    with TemporaryDirectory() as tmpdir:
        audit_path = Path(tmpdir) / 'audit.jsonl'
        fc = FeedbackController(intra_callback=bad_callback, audit_log_path=audit_path)
        with pytest.raises(AttributeError, match='outside feedback boundary'):
            fc.trigger(recorded_log, msq, strategy, 5)
        assert not audit_path.exists()


def test_audit_log_written():

    msq, strategy, _ = make_msq()
    ps = StubPruningStrategy(interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3},
    ])

    with TemporaryDirectory() as tmpdir:
        audit_path = Path(tmpdir) / 'audit.jsonl'

        fc = FeedbackController(
            feedback_interval=5,
            pruning_strategies=[ps],
            audit_log_path=audit_path,
        )

        fc.trigger(None, msq, strategy, 5)

        assert audit_path.exists()
        lines = audit_path.read_text().strip().split('\n')
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry['round'] == 5
        assert entry['trigger_number'] == 0
        assert 'timestamp' in entry
        assert len(entry['interventions']) == 1
        assert 'msq_state_after' in entry
        assert entry['errors'] == []


def test_set_state_missing_key():

    fc = FeedbackController(feedback_interval=10)
    state = fc.get_state()

    for key in ('trigger_count', 'intervention_last_mtime'):
        incomplete = {k: v for k, v in state.items() if k != key}
        try:
            fc.set_state(incomplete)
            assert False, f'Should have raised ValueError for missing {key!r}'
        except ValueError as e:
            assert key in str(e)


def test_get_set_state():

    fc = FeedbackController(feedback_interval=10)

    fc._trigger_count = 3
    fc._intervention_last_mtime = 12345.67

    state = fc.get_state()
    assert state['trigger_count'] == 3
    assert state['intervention_last_mtime'] == 12345.67

    fc2 = FeedbackController(feedback_interval=10)
    fc2.set_state(state)
    assert fc2._trigger_count == 3
    assert fc2._intervention_last_mtime == 12345.67


def test_suggestion_not_dispatched():

    msq, strategy, domain = make_msq()
    ps = StubPruningStrategy(interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3, 'action': 'suggest'},
    ])
    fc = FeedbackController(
        feedback_interval=5,
        pruning_strategies=[ps],
    )

    result = fc.trigger(None, msq, strategy, 5)

    assert result == []
    assert 3 in domain.values_for('a')


def test_suggestion_in_audit_log():

    msq, strategy, _ = make_msq()
    ps = StubPruningStrategy(interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3, 'action': 'suggest'},
    ])

    with TemporaryDirectory() as tmpdir:
        audit_path = Path(tmpdir) / 'audit.jsonl'

        fc = FeedbackController(
            feedback_interval=5,
            pruning_strategies=[ps],
            audit_log_path=audit_path,
        )

        fc.trigger(None, msq, strategy, 5)

        entry = json.loads(audit_path.read_text().strip())
        assert entry['interventions'] == []
        assert len(entry['suggestions']) == 1
        assert entry['suggestions'][0]['action'] == 'suggest'


def test_mixed_interventions_and_suggestions():

    msq, strategy, domain = make_msq()
    ps = StubPruningStrategy(interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3},
        {'op': 'remove_is', 'param': 'a', 'value': 2, 'action': 'suggest'},
    ])
    fc = FeedbackController(
        feedback_interval=5,
        pruning_strategies=[ps],
    )

    result = fc.trigger(None, msq, strategy, 5)

    assert len(result) == 1
    assert result[0]['value'] == 3
    assert 3 not in domain.values_for('a')
    assert 2 in domain.values_for('a')


def test_set_filter_dispatch():

    from limen.experiment.feedback_controller import FeedbackController
    from limen.experiment.reducer.filter_types import FILTER_EXCLUDE_VALUE

    msq, _, domain = make_msq()

    FeedbackController._apply_intervention(msq, {
        'op': 'set_filter',
        'key': 'test_filter',
        'filter_type': FILTER_EXCLUDE_VALUE,
        'filter_params': {'param': 'a', 'value': 3},
    })

    combos = list(msq)
    a_values = {c['a'] for c in combos}
    assert 3 not in a_values
    assert 3 in domain.values_for('a')


def test_clear_filter_dispatch():

    from limen.experiment.feedback_controller import FeedbackController
    from limen.experiment.reducer.filter_types import FILTER_EXCLUDE_VALUE

    msq, _, _ = make_msq()

    FeedbackController._apply_intervention(msq, {
        'op': 'set_filter',
        'key': 'test_filter',
        'filter_type': FILTER_EXCLUDE_VALUE,
        'filter_params': {'param': 'a', 'value': 3},
    })

    FeedbackController._apply_intervention(msq, {
        'op': 'clear_filter',
        'key': 'test_filter',
    })

    combos = list(msq)
    a_values = {c['a'] for c in combos}
    assert 3 in a_values
