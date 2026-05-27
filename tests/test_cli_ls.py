import json
import tempfile
from pathlib import Path

from limen.cli.commands.ls import run_ls
from limen.yaml.config import _STORE_RELATIVE


def _make_project(root: Path) -> Path:
    (root / 'limen.toml').write_text('')
    store = root / _STORE_RELATIVE
    store.mkdir(parents=True)
    return store


def test_run_ls_fails_when_no_project_root() -> None:
    with tempfile.TemporaryDirectory() as d:
        assert run_ls(Path(d)) is False


def test_run_ls_returns_true_when_no_index() -> None:
    with tempfile.TemporaryDirectory() as d:
        _make_project(Path(d))
        assert run_ls(Path(d)) is True


def test_run_ls_returns_true_when_empty_manifests() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_project(Path(d))
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': []}))
        assert run_ls(Path(d)) is True


def test_run_ls_lists_committed_manifests() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_project(Path(d))
        entry = {
            'id': 'sha256:' + 'a' * 64,
            'name': 'my_experiment',
            'committed_at': '2026-05-27T12:00:00',
            'parent_id': None,
        }
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': [entry]}))
        assert run_ls(Path(d)) is True


def test_run_ls_returns_true_when_index_is_corrupt() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_project(Path(d))
        (store / 'index.json').write_text('not valid json')
        assert run_ls(Path(d)) is True


def test_run_ls_shows_parent_id_when_present() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_project(Path(d))
        entry = {
            'id': 'sha256:' + 'b' * 64,
            'name': 'child_experiment',
            'committed_at': '2026-05-27T13:00:00',
            'parent_id': 'sha256:' + 'a' * 64,
        }
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': [entry]}))
        assert run_ls(Path(d)) is True
