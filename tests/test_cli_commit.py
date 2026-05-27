import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from limen.cli.commands.commit import run_commit
from limen.cli.git_utils import git_executable


_DEV_DICT = {'metadata': {'name': 'test', 'mode': 'development'}}
_PROD_DICT = {'metadata': {'name': 'test', 'mode': 'production'}}

_MINIMAL_YAML = 'schema_version: "1.0"\nmetadata:\n  name: test\n  mode: production\n'


def _make_project(root: Path) -> None:
    git = git_executable()
    (root / 'limen.toml').write_text('')
    subprocess.run([git, 'init'], cwd=root, capture_output=True, check=False)
    subprocess.run([git, 'config', 'user.email', 'test@test.com'], cwd=root, capture_output=True, check=False)
    subprocess.run([git, 'config', 'user.name', 'Test'], cwd=root, capture_output=True, check=False)
    (root / 'manifests' / 'committed').mkdir(parents=True)


def test_run_commit_fails_when_no_project_root() -> None:
    with tempfile.TemporaryDirectory() as d:
        yaml_path = Path(d) / 'test.yaml'
        yaml_path.write_text(_MINIMAL_YAML)
        assert run_commit(yaml_path, None, None) is False


def test_run_commit_fails_on_invalid_yaml() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_project(root)
        yaml_path = root / 'test.yaml'
        yaml_path.write_text(_MINIMAL_YAML)
        with patch('limen.cli.commands.commit.load_and_validate', return_value=({}, False)):
            assert run_commit(yaml_path, None, None) is False


def test_run_commit_rejects_development_mode() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_project(root)
        yaml_path = root / 'test.yaml'
        yaml_path.write_text(_MINIMAL_YAML)
        with patch('limen.cli.commands.commit.load_and_validate', return_value=(_DEV_DICT, True)):
            assert run_commit(yaml_path, None, None) is False


def test_run_commit_succeeds_for_production_yaml() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_project(root)
        yaml_path = root / 'test.yaml'
        yaml_path.write_text(_MINIMAL_YAML)
        with patch('limen.cli.commands.commit.load_and_validate', return_value=(_PROD_DICT, True)):
            assert run_commit(yaml_path, None, None) is True


def test_run_commit_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_project(root)
        yaml_path = root / 'test.yaml'
        yaml_path.write_text(_MINIMAL_YAML)
        with patch('limen.cli.commands.commit.load_and_validate', return_value=(_PROD_DICT, True)):
            run_commit(yaml_path, None, None)
            assert run_commit(yaml_path, None, None) is True


def test_run_commit_rejects_invalid_parent_id() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_project(root)
        yaml_path = root / 'test.yaml'
        yaml_path.write_text(_MINIMAL_YAML)
        with patch('limen.cli.commands.commit.load_and_validate', return_value=(_PROD_DICT, True)):
            assert run_commit(yaml_path, 'not-a-valid-id', None) is False
            assert run_commit(yaml_path, 'sha256:short', None) is False
            assert run_commit(yaml_path, 'sha256:' + 'z' * 64, None) is False


def test_run_commit_uses_custom_message() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_project(root)
        yaml_path = root / 'test.yaml'
        yaml_path.write_text(_MINIMAL_YAML)
        with patch('limen.cli.commands.commit.load_and_validate', return_value=(_PROD_DICT, True)):
            assert run_commit(yaml_path, None, 'my custom message') is True
