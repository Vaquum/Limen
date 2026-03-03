from pathlib import Path
from tempfile import TemporaryDirectory

from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.sfd.foundational_sfd import random_binary as sfd_module
from tests.stubs.stubs import make_msq
from tests.stubs.stubs import StubStrategy
from tests.stubs.stubs import StubPruningStrategy


def _make_uel(**kwargs):

    params = sfd_module.params()
    domain = ParamDomain(params)
    strategy = StubStrategy(domain)

    uel = UniversalExperimentLoop(
        sfd=sfd_module,
        search_strategy=strategy,
        feedback_interval=kwargs.get('feedback_interval', 100),
        checkpoint_interval=kwargs.get('checkpoint_interval', 1000),
        experiment_dir=kwargs.get('experiment_dir'),
        intra_callback=kwargs.get('intra_callback'),
    )
    return uel, strategy, domain


def test_run_with_msq_basic_flow():

    uel, _, _ = _make_uel()

    with TemporaryDirectory() as tmpdir:
        uel._run_with_msq(
            experiment_name=str(Path(tmpdir) / 'test'),
            n_permutations=6,

            context_params=None,
            resume=False,
        )

    assert uel.experiment_log.shape[0] == 6
    assert 'id' in uel.experiment_log.columns
    assert '_id' in uel.experiment_log.columns
    assert 'execution_time' in uel.experiment_log.columns
    assert 'random_weights' in uel.experiment_log.columns
    assert 'breakout_threshold' in uel.experiment_log.columns
    assert 'shift' in uel.experiment_log.columns

    assert len(uel.round_params) == 6
    assert len(uel._alignment) == 6
    assert len(uel.preds) == 6

    for rp in uel.round_params:
        assert 'random_weights' in rp
        assert 'breakout_threshold' in rp
        assert 'shift' in rp

    for p in uel.preds:
        assert len(p) > 0

    for a in uel._alignment:
        assert 'first_test_datetime' in a
        assert 'last_test_datetime' in a
        assert 'missing_datetimes' in a

    # _finalize produced valid results
    assert uel._log is not None
    assert uel.experiment_confusion_metrics is not None
    assert len(uel.experiment_confusion_metrics) > 0
    assert uel.experiment_backtest_results is not None
    assert len(uel.experiment_backtest_results) > 0
    corr = uel.experiment_parameter_correlation('auc', min_n=1)
    assert len(corr) > 0


def test_run_with_msq_context_params():

    received_params = []
    uel, _, _ = _make_uel()
    original_model = uel.model

    def capturing_model(data, round_params):
        received_params.append(dict(round_params))
        return original_model(data, round_params)

    uel.model = capturing_model

    with TemporaryDirectory() as tmpdir:
        uel._run_with_msq(
            experiment_name=str(Path(tmpdir) / 'test'),
            n_permutations=1,

            context_params={'env': 'test', 'version': 3},
            resume=False,
        )

    assert received_params[0]['env'] == 'test'
    assert received_params[0]['version'] == 3
    assert 'random_weights' in received_params[0]


def test_run_with_msq_feedback_trigger():

    trigger_rounds = []
    original_trigger = FeedbackController.trigger

    def mock_trigger(self, log, msq, strategy, current_round):
        trigger_rounds.append(current_round)
        return []

    FeedbackController.trigger = mock_trigger

    try:
        uel, _, _ = _make_uel(feedback_interval=2)

        with TemporaryDirectory() as tmpdir:
            uel._run_with_msq(
                experiment_name=str(Path(tmpdir) / 'test'),
                n_permutations=6,
                context_params=None,
                resume=False,
            )

        assert 2 in trigger_rounds
        assert 4 in trigger_rounds
        assert 0 not in trigger_rounds
        assert 1 not in trigger_rounds
    finally:
        FeedbackController.trigger = original_trigger


def test_run_with_msq_checkpoint_trigger():

    checkpoint_rounds = []
    original_save = CheckpointManager.save
    original_trigger = FeedbackController.trigger

    def mock_save(self, checkpoint_dir, msq, domain,
                  current_round, target_permutations, **kwargs):
        checkpoint_rounds.append(current_round)

    def mock_trigger(self, log, msq, strategy, current_round):
        return [{'op': 'remove', 'param': 'shift', 'value': 99}]

    CheckpointManager.save = mock_save
    FeedbackController.trigger = mock_trigger

    try:
        with TemporaryDirectory() as tmpdir:
            uel, _, _ = _make_uel(
                checkpoint_interval=5,
                feedback_interval=2,
                experiment_dir=Path(tmpdir),
            )

            uel._run_with_msq(
                experiment_name='test_ckpt',
                n_permutations=6,
                context_params=None,
                resume=False,
            )

            # Periodic checkpoint at round 5
            assert 5 in checkpoint_rounds
            # Feedback interventions at rounds 2, 4 trigger checkpoints
            assert 2 in checkpoint_rounds
            assert 4 in checkpoint_rounds
            # No checkpoint at rounds without triggers
            assert 0 not in checkpoint_rounds
            assert 1 not in checkpoint_rounds
    finally:
        CheckpointManager.save = original_save
        FeedbackController.trigger = original_trigger


def test_checkpoint_saves_feedback_and_pruning_state():

    ps = StubPruningStrategy(active=True)
    msq, _, domain = make_msq()
    next(msq)

    fc = FeedbackController(
        feedback_interval=100,
        pruning_strategies=[ps],
    )
    fc._trigger_count = 5

    with TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / 'ckpt'
        ckpt_dir.mkdir()
        cm = CheckpointManager()

        cm.save(ckpt_dir, msq, domain, 10, 100,
                strategy_type='StubStrategy', content_hash='a' * 64,
                feedback_controller=fc, pruning_strategies=[ps])

        data = cm.load(ckpt_dir)

    assert 'feedback_controller_state' in data
    assert data['feedback_controller_state']['trigger_count'] == 5
    assert 'pruning_strategy_states' in data
    assert len(data['pruning_strategy_states']) == 1
    assert data['pruning_strategy_states'][0]['active'] is True


def test_run_with_msq_shutdown_resume_full_data():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()

        # Phase 1: run with shutdown after 2 rounds
        uel, _, _ = _make_uel(experiment_dir=exp_dir)
        original_model = uel.model
        round_count = 0

        def shutdown_after_2(data, round_params):
            nonlocal round_count
            round_count += 1
            result = original_model(data, round_params)
            if round_count >= 2:
                uel._shutdown_requested = True
            return result

        uel.model = shutdown_after_2

        uel._run_with_msq(
            experiment_name='test_full_data',
            n_permutations=6,

            context_params=None,
            resume=False,
        )

        # Verify round_data.jsonl exists with 2 lines
        round_data_path = exp_dir / 'round_data.jsonl'
        assert round_data_path.exists()
        with round_data_path.open('r') as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 2

        # Verify results.csv exists with 2 data rows
        csv_path = exp_dir / 'results.csv'
        assert csv_path.exists()

        # Phase 2: resume and complete
        uel2, _, _ = _make_uel(experiment_dir=exp_dir)

        uel2._run_with_msq(
            experiment_name='test_full_data',
            n_permutations=6,

            context_params=None,
            resume=True,
        )

        # Full experiment data integrity — counts
        assert uel2.experiment_log.shape[0] == 6
        assert uel2.experiment_log['id'].to_list() == [0, 1, 2, 3, 4, 5]
        assert len(uel2.preds) == 6
        assert len(uel2._alignment) == 6
        assert len(uel2.round_params) == 6

        # round_params content across shutdown boundary
        for rp in uel2.round_params:
            assert 'random_weights' in rp
            assert 'breakout_threshold' in rp
            assert 'shift' in rp

        # preds content across shutdown boundary
        for p in uel2.preds:
            assert len(p) > 0

        # _alignment content across shutdown boundary
        for a in uel2._alignment:
            assert 'first_test_datetime' in a
            assert 'last_test_datetime' in a
            assert 'missing_datetimes' in a

        # JSONL has all 6 rounds after resume completion
        with round_data_path.open('r') as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 6

        # CSV has header + 6 data rows
        with csv_path.open('r') as f:
            csv_lines = [line for line in f if line.strip()]
        assert len(csv_lines) == 7

        # _finalize ran successfully with full data
        assert uel2._log is not None
        assert uel2.experiment_confusion_metrics is not None
        assert len(uel2.experiment_confusion_metrics) > 0
        assert uel2.experiment_backtest_results is not None
        assert len(uel2.experiment_backtest_results) > 0
        corr = uel2.experiment_parameter_correlation('auc', min_n=1)
        assert len(corr) > 0


def test_resume_fails_without_round_data():

    params = sfd_module.params()
    domain = ParamDomain(params)
    strategy = StubStrategy(domain)
    msq = MSQ(strategy, domain, n_permutations=6)

    # Advance MSQ 3 rounds
    for _ in range(3):
        next(msq)

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()

        # Save checkpoint but no round_data.jsonl
        content_hash = CheckpointManager.compute_content_hash(params)
        cm = CheckpointManager(checkpoint_interval=1000)
        cm.save(exp_dir, msq, domain, 2, 6,
                strategy_type='StubStrategy', content_hash=content_hash)

        uel, _, _ = _make_uel(experiment_dir=exp_dir)

        raised = False
        try:
            uel._run_with_msq(
                experiment_name='test_missing_data',
                n_permutations=6,
                context_params=None,
                resume=True,
            )
        except ValueError as e:
            raised = True
            assert 'round_data.jsonl not found' in str(e)

        assert raised
