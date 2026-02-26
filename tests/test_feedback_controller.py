import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from limen.experiment.feedback_controller import FeedbackController
from tests.stubs.stubs import StubPruningStrategy, make_msq


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

    msq, strategy, _ = make_msq()
    ps = StubPruningStrategy(interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3},
    ])
    fc = FeedbackController(
        feedback_interval=5,
        pruning_strategies=[ps],
    )

    fc.trigger(None, msq, strategy, 5)

    assert len(strategy._feedback_calls) == 1
    assert len(strategy._feedback_calls[0]) == 1
    assert strategy._feedback_calls[0][0]['op'] == 'remove_is'


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

        # Second trigger without file change — no new interventions
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


def test_file_invalid_json_no_error():

    msq, strategy, _ = make_msq()

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'bad.json'
        path.write_text('not valid json {{{')

        fc = FeedbackController(
            feedback_interval=5,
            intervention_path=path,
        )

        # Should not raise — error captured internally
        result = fc.trigger(None, msq, strategy, 5)
        assert result == []


def test_source_isolation():

    msq, strategy, domain = make_msq()

    def bad_callback(_log, _msq):
        raise RuntimeError('callback crashed')

    ps = StubPruningStrategy(interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3},
    ])

    fc = FeedbackController(
        feedback_interval=5,
        pruning_strategies=[ps],
        intra_callback=bad_callback,
    )

    result = fc.trigger(None, msq, strategy, 5)

    # Pruning strategy ran before callback, so its intervention applied
    assert any(i['op'] == 'remove_is' for i in result)
    assert 3 not in domain.values_for('a')


def test_pruning_strategy_isolation():

    msq, strategy, domain = make_msq()

    class FailingStrategy(StubPruningStrategy):
        def analyze_and_intervene(self, _log, _msq):
            raise RuntimeError('strategy crashed')

    ps_bad = FailingStrategy()
    ps_good = StubPruningStrategy(interventions=[
        {'op': 'remove_is', 'param': 'a', 'value': 3},
    ])

    fc = FeedbackController(
        feedback_interval=5,
        pruning_strategies=[ps_bad, ps_good],
    )

    result = fc.trigger(None, msq, strategy, 5)

    assert any(i['op'] == 'remove_is' for i in result)
    assert 3 not in domain.values_for('a')


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
