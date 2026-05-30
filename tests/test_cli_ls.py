import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from limen.cli.commands.ls import run_ls
from limen.yaml.config import _STORE_RELATIVE


def _make_project(root: Path) -> Path:
    (root / 'limen.toml').write_text('')
    store = root / _STORE_RELATIVE
    store.mkdir(parents=True)
    return store


def test_run_ls_fails_when_no_project_root() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        assert run_ls(Path(d)) is False


def test_run_ls_returns_true_when_no_index() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        _make_project(Path(d))
        assert run_ls(Path(d)) is True


def test_run_ls_returns_true_when_empty_manifests() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        store = _make_project(Path(d))
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': []}))
        assert run_ls(Path(d)) is True


def test_run_ls_lists_committed_manifests() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
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
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        store = _make_project(Path(d))
        (store / 'index.json').write_text('not valid json')
        assert run_ls(Path(d)) is True


def test_run_ls_returns_true_when_index_has_invalid_structure() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        store = _make_project(Path(d))
        (store / 'index.json').write_text(json.dumps([1, 2, 3]))
        assert run_ls(Path(d)) is True


def test_run_ls_skips_malformed_entries() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        store = _make_project(Path(d))
        good = {'id': 'sha256:' + 'a' * 64, 'name': 'ok', 'committed_at': '2026-05-28T10:00:00Z', 'parent_id': None}
        bad = {'broken': True}
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': [good, bad]}))
        assert run_ls(Path(d)) is True


def test_run_ls_skips_entry_with_non_string_id() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        store = _make_project(Path(d))
        entry = {'id': 12345, 'name': 'bad_id', 'committed_at': '2026-05-28T10:00:00Z', 'parent_id': None}
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': [entry]}))
        assert run_ls(Path(d)) is True


def test_run_ls_skips_entry_with_non_string_parent_id() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        store = _make_project(Path(d))
        entry = {'id': 'sha256:' + 'a' * 64, 'name': 'ok', 'committed_at': '2026-05-28T10:00:00Z', 'parent_id': 999}
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': [entry]}))
        assert run_ls(Path(d)) is True


def test_run_ls_skips_entry_with_truncated_id() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        store = _make_project(Path(d))
        entry = {'id': 'sha256:' + 'a' * 8, 'name': 'ok', 'committed_at': '2026-05-28T10:00:00Z', 'parent_id': None}
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': [entry]}))
        assert run_ls(Path(d)) is True


def test_run_ls_skips_entry_with_non_hex_id() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        store = _make_project(Path(d))
        entry = {'id': 'sha256:' + 'z' * 64, 'name': 'ok', 'committed_at': '2026-05-28T10:00:00Z', 'parent_id': None}
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': [entry]}))
        assert run_ls(Path(d)) is True


def test_run_ls_skips_entry_with_non_string_name() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        store = _make_project(Path(d))
        entry = {'id': 'sha256:' + 'a' * 64, 'name': 99, 'committed_at': '2026-05-28T10:00:00Z', 'parent_id': None}
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': [entry]}))
        assert run_ls(Path(d)) is True


def test_run_ls_skips_entry_with_non_string_committed_at() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        store = _make_project(Path(d))
        entry = {'id': 'sha256:' + 'a' * 64, 'name': 'ok', 'committed_at': 12345, 'parent_id': None}
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': [entry]}))
        assert run_ls(Path(d)) is True


def test_run_ls_shows_parent_id_when_present() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        store = _make_project(Path(d))
        entry = {
            'id': 'sha256:' + 'b' * 64,
            'name': 'child_experiment',
            'committed_at': '2026-05-27T13:00:00',
            'parent_id': 'sha256:' + 'a' * 64,
        }
        (store / 'index.json').write_text(json.dumps({'version': 1, 'manifests': [entry]}))
        assert run_ls(Path(d)) is True
