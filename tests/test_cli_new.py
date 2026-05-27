import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from limen.cli.commands.new import run_new


def test_run_new_fails_when_directory_exists() -> None:
    with tempfile.TemporaryDirectory() as d:
        existing = Path(d) / 'my-project'
        existing.mkdir()
        assert run_new(str(existing), None) is False


def test_run_new_succeeds_with_mocked_clone() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'my-project'
        project_path.mkdir()
        (project_path / 'limen.toml').write_text('backup_remote = ""\n')

        def fake_clone(path: Path) -> bool:
            return True

        with patch('limen.cli.commands.new._clone_template', side_effect=fake_clone), \
             patch('click.prompt', return_value=''):
            result = run_new(str(project_path.parent / 'new-project'), None)

        assert result is True


def test_run_new_sets_backup_remote() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'new-project'

        def fake_clone(path: Path) -> bool:
            path.mkdir(parents=True, exist_ok=True)
            (path / 'limen.toml').write_text('backup_remote = ""\n')
            return True

        with patch('limen.cli.commands.new._clone_template', side_effect=fake_clone):
            result = run_new(str(project_path), 'git@github.com:user/repo.git')

        assert result is True
        assert 'git@github.com:user/repo.git' in (project_path / 'limen.toml').read_text()


def test_run_new_clone_failure_returns_false() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'new-project'

        with patch('limen.cli.commands.new._clone_template', return_value=False):
            result = run_new(str(project_path), None)

        assert result is False


def test_run_new_warns_when_backup_remote_placeholder_missing() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'new-project'

        def fake_clone(path: Path) -> bool:
            path.mkdir(parents=True, exist_ok=True)
            (path / 'limen.toml').write_text('# no placeholder here\n')
            return True

        with patch('limen.cli.commands.new._clone_template', side_effect=fake_clone):
            result = run_new(str(project_path), 'git@github.com:user/repo.git')

        assert result is True
        assert 'git@github.com:user/repo.git' not in (project_path / 'limen.toml').read_text()


def test_run_new_fails_when_git_not_found() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'new-project'
        with patch('limen.cli.commands.new.git_executable', side_effect=FileNotFoundError):
            result = run_new(str(project_path), None)
    assert result is False


def test_run_new_skips_backup_remote_with_invalid_chars() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'new-project'

        def fake_clone(path: Path) -> bool:
            path.mkdir(parents=True, exist_ok=True)
            (path / 'limen.toml').write_text('backup_remote = ""\n')
            return True

        with patch('limen.cli.commands.new._clone_template', side_effect=fake_clone):
            result = run_new(str(project_path), 'git@github.com:user/"bad".git')

        assert result is True
        assert '"bad"' not in (project_path / 'limen.toml').read_text()


def test_clone_template_fails_fast_on_git_init_failure() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'new-project'

        def fake_subprocess(args: list, **kwargs: object) -> MagicMock:
            result = MagicMock()
            if 'clone' in args:
                project_path.mkdir(parents=True, exist_ok=True)
                (project_path / '.git').mkdir()
                result.returncode = 0
                result.stderr = ''
            elif 'init' in args:
                result.returncode = 1
                result.stderr = 'init failed'
            else:
                result.returncode = 0
            return result

        with patch('limen.cli.commands.new.subprocess.run', side_effect=fake_subprocess), \
             patch('limen.cli.commands.new.shutil.rmtree'):
            result = run_new(str(project_path), None)

        assert result is False
