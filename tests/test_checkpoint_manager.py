import json
import signal
from pathlib import Path
from tempfile import TemporaryDirectory

from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from tests.stubs.stubs import make_msq


def _make_uel():
    uel = object.__new__(UniversalExperimentLoop)
    uel._shutdown_requested = False
    uel._pause_requested = False
    return uel


def test_checkpoint_interval_validation():

    try:
        CheckpointManager(checkpoint_interval=0)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'checkpoint_interval' in str(e)


def test_should_checkpoint_interval():

    cm = CheckpointManager(checkpoint_interval=5)

    assert cm.should_checkpoint(0) is False
    assert cm.should_checkpoint(1) is False
    assert cm.should_checkpoint(4) is False
    assert cm.should_checkpoint(5) is True
    assert cm.should_checkpoint(10) is True
    assert cm.should_checkpoint(7) is False


def test_compute_content_hash_dict():

    h1 = CheckpointManager.compute_content_hash({'b': 2, 'a': 1})
    h2 = CheckpointManager.compute_content_hash({'a': 1, 'b': 2})

    assert h1 == h2
    assert len(h1) == 64


def test_initialize_fresh_creates_directory():

    with TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / 'my_experiment'
        cm = CheckpointManager()

        result = cm.initialize_fresh(ckpt_dir)

        assert result.is_dir()
        assert result == ckpt_dir


def test_save_writes_three_files():

    msq, _, domain = make_msq()

    with TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / 'ckpt'
        ckpt_dir.mkdir()
        cm = CheckpointManager()

        cm.save(ckpt_dir, msq, domain, 100, 1000,
                strategy_type='StubStrategy', content_hash='a' * 64)

        assert (ckpt_dir / 'checkpoint_metadata.json').exists()
        assert (ckpt_dir / 'msq_state.json').exists()
        assert (ckpt_dir / 'domain_state.json').exists()


def test_save_metadata_content():

    msq, _, domain = make_msq()

    with TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / 'ckpt'
        ckpt_dir.mkdir()
        cm = CheckpointManager()

        cm.save(ckpt_dir, msq, domain, 42, 500,
                strategy_type='GridStrategy', content_hash='b' * 64)

        with (ckpt_dir / 'checkpoint_metadata.json').open() as f:
            metadata = json.load(f)

        assert metadata['experiment_round'] == 42
        assert metadata['target_permutations'] == 500
        assert metadata['strategy_type'] == 'GridStrategy'
        assert metadata['content_hash'] == 'b' * 64
        assert 'saved_at' in metadata


def test_validate_passes_when_all_match():

    msq, _, domain = make_msq()
    content_hash = CheckpointManager.compute_content_hash({'params': 'test'})

    with TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / 'ckpt'
        ckpt_dir.mkdir()
        cm = CheckpointManager()

        cm.save(ckpt_dir, msq, domain, 10, 100,
                strategy_type='StubStrategy', content_hash=content_hash)

        cm.validate(ckpt_dir, content_hash=content_hash, strategy_type='StubStrategy')


def test_validate_raises_on_hash_mismatch():

    msq, _, domain = make_msq()

    with TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / 'ckpt'
        ckpt_dir.mkdir()
        cm = CheckpointManager()

        cm.save(ckpt_dir, msq, domain, 10, 100,
                strategy_type='StubStrategy', content_hash='a' * 64)

        try:
            cm.validate(ckpt_dir, content_hash='b' * 64, strategy_type='StubStrategy')
            assert False, 'Should have raised ValueError'
        except ValueError as e:
            assert 'Content hash mismatch' in str(e)


def test_validate_raises_on_strategy_mismatch():

    msq, _, domain = make_msq()
    content_hash = 'a' * 64

    with TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / 'ckpt'
        ckpt_dir.mkdir()
        cm = CheckpointManager()

        cm.save(ckpt_dir, msq, domain, 10, 100,
                strategy_type='RandomStrategy', content_hash=content_hash)

        try:
            cm.validate(ckpt_dir, content_hash=content_hash, strategy_type='GridStrategy')
            assert False, 'Should have raised ValueError'
        except ValueError as e:
            assert 'Strategy type mismatch' in str(e)


def test_validate_raises_on_missing_metadata():

    with TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / 'ckpt'
        ckpt_dir.mkdir()
        cm = CheckpointManager()

        try:
            cm.validate(ckpt_dir, content_hash='a' * 64, strategy_type='StubStrategy')
            assert False, 'Should have raised ValueError'
        except ValueError as e:
            assert 'incomplete' in str(e)


def test_validate_raises_on_corrupt_metadata():

    with TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / 'ckpt'
        ckpt_dir.mkdir()
        cm = CheckpointManager()

        (ckpt_dir / 'checkpoint_metadata.json').write_text('not valid json')

        try:
            cm.validate(ckpt_dir, content_hash='a' * 64, strategy_type='StubStrategy')
            assert False, 'Should have raised ValueError'
        except ValueError as e:
            assert 'Corrupt' in str(e)


def test_param_domain_set_state_invalid_leaves_state_unchanged():

    domain = ParamDomain({'lr': [0.001, 0.01], 'n': [10, 20]})
    original_lr = domain.values_for('lr')

    try:
        domain.set_state({'lr': [0.001], 'n': []})
        assert False, 'Should have raised ValueError'
    except ValueError:
        pass

    assert domain.values_for('lr') == original_lr
    assert domain.values_for('n') == [10, 20]


def test_param_domain_get_set_state():

    domain = ParamDomain({'lr': [0.001, 0.01], 'n': [10, 20, 30]})

    state = domain.get_state()

    assert state == {'lr': [0.001, 0.01], 'n': [10, 20, 30]}

    domain.remove_value('lr', 0.01)
    assert domain.values_for('lr') == [0.001]

    domain2 = ParamDomain({'lr': [0.001], 'n': [10]})
    domain2.set_state(state)

    assert domain2.values_for('lr') == [0.001, 0.01]
    assert domain2.values_for('n') == [10, 20, 30]
    assert domain2.version == 0


def test_uel_shutdown_flag():

    uel = _make_uel()

    assert uel._shutdown_requested is False
    assert uel._pause_requested is False

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)

    try:
        uel._register_shutdown_handler()
        signal.raise_signal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)

    assert uel._shutdown_requested is True
    assert uel._pause_requested is False


def test_uel_checkpoint_and_resume():

    uel = _make_uel()
    msq, _, domain = make_msq()
    next(msq)
    content_hash = 'a' * 64

    with TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / 'ckpt'
        cm = CheckpointManager()

        uel._initialize_fresh(ckpt_dir, cm)
        uel._checkpoint(msq, domain, ckpt_dir, cm, 10, 100,
                        strategy_type='StubStrategy', content_hash=content_hash)

        state = uel._resume_from_checkpoint(ckpt_dir, cm,
                                             content_hash=content_hash,
                                             strategy_type='StubStrategy')

    assert state['metadata']['experiment_round'] == 10
    assert state['msq_state']['yielded_count'] == 1
