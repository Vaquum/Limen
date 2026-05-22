import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import polars as pl

import pandas as pd
import pytest

from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.sfd.foundational_sfd import random_binary as sfd_module
from limen.experiment.param_search import GridStrategy
from limen.experiment.param_search import RandomStrategy
from limen.log._experiment_parameter_correlation import _experiment_parameter_correlation
from tests.stubs.stubs import make_msq
from tests.stubs.stubs import StubPruningStrategy


def _make_uel(strategy_cls=RandomStrategy, test_mode=True, **kwargs):

    params = sfd_module.params()
    domain = ParamDomain(params)
    strategy = strategy_cls(domain, seed=42)

    uel = UniversalExperimentLoop(
        sfd=sfd_module,
        search_strategy=strategy,
        feedback_interval=kwargs.get('feedback_interval', 100),
        checkpoint_interval=kwargs.get('checkpoint_interval', 1000),
        experiment_dir=kwargs.get('experiment_dir'),
        intra_callback=kwargs.get('intra_callback'),
        test_mode=test_mode,
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
            post_processing=True,
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


def test_run_with_msq_skips_post_processing_by_default():

    uel, _, _ = _make_uel()
    finalize_calls = []

    def fake_finalize():
        finalize_calls.append(True)

    uel._finalize = fake_finalize

    with TemporaryDirectory() as tmpdir:
        uel.run(
            experiment_name=str(Path(tmpdir) / 'test'),
            n_permutations=1,
        )

    assert finalize_calls == []
    assert uel.experiment_log.shape[0] == 1
    assert uel._log is None
    assert uel.experiment_confusion_metrics is None
    assert uel.experiment_backtest_results is None
    assert uel.experiment_parameter_correlation is None


def test_run_with_msq_post_processing_opt_in_reaches_finalize():

    uel, _, _ = _make_uel()
    finalize_calls = []

    def fake_finalize():
        finalize_calls.append(True)

    uel._finalize = fake_finalize

    with TemporaryDirectory() as tmpdir:
        uel.run(
            experiment_name=str(Path(tmpdir) / 'test'),
            n_permutations=1,
            post_processing=True,
        )

    assert finalize_calls == [True]


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
                strategy_type='GridStrategy', content_hash='a' * 64,
                feedback_controller=fc, pruning_strategies=[ps])

        data = cm.load(ckpt_dir)

    assert 'feedback_controller_state' in data
    assert data['feedback_controller_state']['trigger_count'] == 5
    assert 'pruning_strategy_states' in data
    assert len(data['pruning_strategy_states']) == 1
    assert data['pruning_strategy_states'][0]['active'] is True


def _shutdown_resume_full_data(strategy_cls):

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()

        uel, _, _ = _make_uel(strategy_cls=strategy_cls, experiment_dir=exp_dir)
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

        round_data_path = exp_dir / 'round_data.jsonl'
        assert round_data_path.exists()
        with round_data_path.open('r') as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 2

        csv_path = exp_dir / 'results.csv'
        assert csv_path.exists()

        uel2, _, _ = _make_uel(strategy_cls=strategy_cls, experiment_dir=exp_dir)

        uel2._run_with_msq(
            experiment_name='test_full_data',
            n_permutations=6,
            context_params=None,
            resume=True,
            post_processing=True,
        )

        assert uel2.experiment_log.shape[0] == 6
        assert uel2.experiment_log['id'].to_list() == [0, 1, 2, 3, 4, 5]
        assert len(uel2.preds) == 6
        assert len(uel2._alignment) == 6
        assert len(uel2.round_params) == 6

        for rp in uel2.round_params:
            assert 'random_weights' in rp
            assert 'breakout_threshold' in rp
            assert 'shift' in rp

        for p in uel2.preds:
            assert len(p) > 0

        for a in uel2._alignment:
            assert 'first_test_datetime' in a
            assert 'last_test_datetime' in a
            assert 'missing_datetimes' in a

        with round_data_path.open('r') as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 6

        with csv_path.open('r') as f:
            csv_lines = [line for line in f if line.strip()]
        assert len(csv_lines) == 7

        assert uel2._log is not None
        assert uel2.experiment_confusion_metrics is not None
        assert len(uel2.experiment_confusion_metrics) > 0
        assert uel2.experiment_backtest_results is not None
        assert len(uel2.experiment_backtest_results) > 0
        corr = uel2.experiment_parameter_correlation('auc', min_n=1)
        assert len(corr) > 0


def test_run_with_msq_shutdown_resume_full_data():

    _shutdown_resume_full_data(RandomStrategy)


def test_run_with_msq_shutdown_resume_grid():

    _shutdown_resume_full_data(GridStrategy)


def test_shutdown_before_any_round_completes():

    uel, _, _ = _make_uel()
    uel._shutdown_requested = True

    with TemporaryDirectory() as tmpdir:
        uel._run_with_msq(
            experiment_name=str(Path(tmpdir) / 'test'),
            n_permutations=6,
            context_params=None,
            resume=False,
        )

    assert uel.experiment_log is None


def test_resume_fails_without_round_data():

    params = sfd_module.params()
    domain = ParamDomain(params)
    strategy = RandomStrategy(domain, seed=42)
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
                strategy_type='RandomStrategy', content_hash=content_hash)

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


def test_preds_none_does_not_crash():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()
        round_data_path = exp_dir / 'round_data.jsonl'

        uel, _, _ = _make_uel(experiment_dir=exp_dir)
        sfd_params = {'random_weights': True, 'breakout_threshold': 1.2, 'shift': 1}
        data_dict = uel.prep(uel.data, round_params=sfd_params)
        round_results = uel.model(data=data_dict, round_params=sfd_params)

        round_results['_preds'] = None

        current_preds = round_results.pop('_preds', None)
        if current_preds is None:
            current_preds = []

        uel._append_round_data(
            round_data_path, 0, sfd_params, current_preds,
            data_dict['_alignment'],
        )

        assert round_data_path.exists()
        with round_data_path.open('r') as f:
            entry = json.loads(f.readline())
        assert entry['preds'] == []


def test_resume_truncates_stale_rounds():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()

        uel, _, _ = _make_uel(
            experiment_dir=exp_dir,
            checkpoint_interval=2,
        )
        original_model = uel.model
        round_count = 0

        def shutdown_after_4(data, round_params):
            nonlocal round_count
            round_count += 1
            result = original_model(data, round_params)
            if round_count >= 4:
                uel._shutdown_requested = True
            return result

        uel.model = shutdown_after_4

        uel._run_with_msq(
            experiment_name='test_truncation',
            n_permutations=6,
            context_params=None,
            resume=False,
        )

        round_data_path = exp_dir / 'round_data.jsonl'
        csv_path = exp_dir / 'results.csv'
        assert round_data_path.exists()
        assert csv_path.exists()

        with round_data_path.open('r') as f:
            existing_lines = [raw.strip() for raw in f if raw.strip()]

        # Append stale entries beyond the checkpoint to simulate crash
        checkpoint_round = json.loads(existing_lines[-1])['round_id']
        for i in range(1, 3):
            stale_entry = json.loads(existing_lines[-1])
            stale_entry['round_id'] = checkpoint_round + i
            with round_data_path.open('a') as f:
                f.write(json.dumps(stale_entry) + '\n')

        with csv_path.open('r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        id_idx = header.index('id')
        with csv_path.open('a', newline='') as f:
            writer = csv.writer(f)
            for offset in range(1, 3):
                stale_row = list(rows[-1])
                stale_row[id_idx] = str(int(stale_row[id_idx]) + offset)
                writer.writerow(stale_row)

        with round_data_path.open('r') as f:
            pre_resume_jsonl = [raw for raw in f if raw.strip()]
        assert len(pre_resume_jsonl) == len(existing_lines) + 2

        uel2, _, _ = _make_uel(
            experiment_dir=exp_dir,
            checkpoint_interval=2,
        )

        uel2._run_with_msq(
            experiment_name='test_truncation',
            n_permutations=6,
            context_params=None,
            resume=True,
        )

        assert uel2.experiment_log.shape[0] == 6
        assert uel2.experiment_log['id'].to_list() == [0, 1, 2, 3, 4, 5]

        with round_data_path.open('r') as f:
            final_jsonl = [raw for raw in f if raw.strip()]
        assert len(final_jsonl) == 6
        for i, line in enumerate(final_jsonl):
            assert json.loads(line)['round_id'] == i

        with csv_path.open('r') as f:
            final_csv = [raw for raw in f if raw.strip()]
        assert len(final_csv) == 7


def test_truncate_round_data_unit():

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'round_data.jsonl'

        entries = [
            {'round_id': 0, 'data': 'a'},
            {'round_id': 1, 'data': 'b'},
            {'round_id': 2, 'data': 'c'},
            {'round_id': 3, 'data': 'd'},
            {'round_id': 4, 'data': 'e'},
        ]
        with path.open('w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        UniversalExperimentLoop._truncate_round_data(path, 3)

        with path.open('r') as f:
            remaining = [json.loads(raw) for raw in f if raw.strip()]

        assert len(remaining) == 3
        assert [e['round_id'] for e in remaining] == [0, 1, 2]


def test_experiment_parameter_correlation_keeps_numeric_signal_columns_and_orders_cohorts():

    experiment_log = pd.DataFrame({
        'auc': [0.1, 0.2, 0.3, 0.4, 0.5],
        'param_up': [1, 2, 3, 4, 5],
        'param_down': [5, 4, 3, 2, 1],
        'constant': [7, 7, 7, 7, 7],
        'label': ['a', 'b', 'c', 'd', 'e'],
    })

    corr = _experiment_parameter_correlation(
        SimpleNamespace(experiment_log=experiment_log),
        'auc',
        heads=(1.0, 0.6),
        min_n=2,
        n_boot=20,
        random_state=0,
    )

    assert corr.index.names == ['cohort_pct', 'feature']
    assert list(corr.columns) == [
        'n_rows',
        'corr',
        'corr_med',
        'ci_lo',
        'ci_hi',
        'sign_stability',
    ]
    assert set(corr.xs(100, level='cohort_pct').index.tolist()) == {
        'param_down',
        'param_up',
    }
    assert 'constant' not in corr.index.get_level_values('feature')
    assert 'label' not in corr.index.get_level_values('feature')
    assert corr.loc[(100, 'param_up'), 'corr_med'] == pytest.approx(1.0)
    assert corr.loc[(100, 'param_down'), 'corr_med'] == pytest.approx(-1.0)
    assert corr.loc[(60, 'param_up'), 'n_rows'] == 3
    assert corr['sign_stability'].between(0.0, 1.0).all()


def test_experiment_parameter_correlation_validates_metric_and_sort_key_presence():

    experiment_log = pd.DataFrame({
        'auc': [0.1, 0.2, 0.3],
        'param': [1, 2, 3],
    })
    shim = SimpleNamespace(experiment_log=experiment_log)

    with pytest.raises(ValueError, match='metric "loss" not found'):
        _experiment_parameter_correlation(shim, 'loss')

    with pytest.raises(ValueError, match='sort_key "missing" not found'):
        _experiment_parameter_correlation(shim, 'auc', sort_key='missing')


def test_experiment_parameter_correlation_requires_numeric_features_after_cleaning():

    experiment_log = pd.DataFrame({
        'auc': [0.1, 0.2],
        'constant': [1, 1],
        'label': ['low', 'high'],
    })

    with pytest.raises(
        ValueError,
        match='No numeric features available',
    ):
        _experiment_parameter_correlation(
            SimpleNamespace(experiment_log=experiment_log),
            'auc',
            min_n=1,
            n_boot=5,
        )


def test_experiment_parameter_correlation_rejects_logs_without_metric_rows():

    experiment_log = pd.DataFrame({
        'auc': [None, None],
        'param': [1, 2],
    })

    with pytest.raises(
        ValueError,
        match='No rows remain after dropping NaNs in the metric column',
    ):
        _experiment_parameter_correlation(
            SimpleNamespace(experiment_log=experiment_log),
            'auc',
            min_n=1,
            n_boot=5,
        )


def test_uel_test_mode_routes_to_test_data_source():

    _stub_data = pl.DataFrame({'x': [1, 2, 3]})
    fetched = []

    class _FakeManifest:
        data_source_config = object()
        test_data_source_config = object()

        def architecture_function(self, data, **kw):
            return {}

        def fetch_data(self):
            fetched.append('fetch_data')
            return _stub_data

        def fetch_test_data(self):
            fetched.append('fetch_test_data')
            return _stub_data

    class _FakeSfd:
        def params(self):
            return {'x': [1, 2]}

        def manifest(self):
            return _FakeManifest()

    params = _FakeSfd().params()
    domain = ParamDomain(params)
    strategy = RandomStrategy(domain, seed=42)

    UniversalExperimentLoop(
        sfd=_FakeSfd(),
        search_strategy=strategy,
        test_mode=True,
    )
    assert fetched == ['fetch_test_data']

    fetched.clear()
    UniversalExperimentLoop(
        sfd=_FakeSfd(),
        search_strategy=strategy,
        test_mode=False,
    )
    assert fetched == ['fetch_data']

    class _FakeManifestNoTestSource:
        data_source_config = object()
        test_data_source_config = None

        def architecture_function(self, data, **kw):
            return {}

        def fetch_data(self):
            fetched.append('fetch_data')
            return _stub_data

        def fetch_test_data(self):
            fetched.append('fetch_test_data')
            return _stub_data

    class _FakeSfdNoTestSource:
        def params(self):
            return {'x': [1, 2]}

        def manifest(self):
            return _FakeManifestNoTestSource()

    fetched.clear()
    UniversalExperimentLoop(
        sfd=_FakeSfdNoTestSource(),
        search_strategy=strategy,
        test_mode=True,
    )
    assert fetched == ['fetch_data']
