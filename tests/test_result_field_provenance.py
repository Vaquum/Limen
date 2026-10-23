import pytest

from limen.experiment import UniversalExperimentLoop
from limen.experiment import _holdout_history as history
from limen.experiment import _holdout_research as research
from limen.experiment.holdout import reserve_holdout
from limen.yaml import CompiledSFD
from limen.yaml.compiler import build_search_strategy
from tests.test_holdout_research import _configuration, _project


def _research_rows(root):
    with history.connection(root / 'manifests/holdout.sqlite3') as db:
        return {row['id']: dict(row) for row in db.execute('SELECT * FROM research')}


def _artifact_snapshot(directory):
    return {path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


def test_collision_admission_preserves_protected_provenance(tmp_path, monkeypatch):
    root = _project(tmp_path / 'project')
    config = _configuration(root)
    reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    directory = root / 'results'
    first = UniversalExperimentLoop(
        sfd=CompiledSFD(config, project_root=root), project_root=root,
        experiment_dir=directory, search_strategy=build_search_strategy(config),
        yaml_reference=config,
    )
    first.run('guarded', n_permutations=1, progress_bar=False, post_processing=True)
    rejected_directory = root / 'rejected'
    rejected = UniversalExperimentLoop(
        sfd=CompiledSFD(config, project_root=root), project_root=root,
        experiment_dir=rejected_directory, search_strategy=build_search_strategy(config),
        yaml_reference=config,
    )
    before = _research_rows(root)
    assert before and all(row['state'] == 'completed' for row in before.values())
    artifacts = _artifact_snapshot(directory)
    started = []
    original_start = research.start_research

    def observe_start(binding, role, manifest_id=None):
        record_id = original_start(binding, role, manifest_id)
        started.append(dict(history.research_record(binding, record_id)))
        return record_id

    monkeypatch.setattr(research, 'start_research', observe_start)
    with pytest.raises(ValueError, match='strict_mode_error'):
        rejected.run('rejected', n_permutations=1, progress_bar=False,
                     context_params={'strict_mode_error': 'forged'})

    after = _research_rows(root)
    assert len(started) == 1
    attempt = started[0]
    assert attempt['state'] == 'started' and attempt['finished_ns'] is None
    assert set(after) - set(before) == {attempt['id']}
    assert {key: after[key] for key in before} == before
    failed = after[attempt['id']]
    assert failed['state'] == 'failed' and failed['finished_ns'] is not None
    assert failed['role'] == 'experiment' and failed['round_id'] is None
    assert failed['error'].startswith('ValueError:') and 'strict_mode_error' in failed['error']
    assert not rejected_directory.exists()
    assert _artifact_snapshot(directory) == artifacts
