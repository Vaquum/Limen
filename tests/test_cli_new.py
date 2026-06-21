import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from limen.cli.commands.new import _clone_template
from limen.cli.commands.new import run_new


def test_run_new_fails_when_directory_exists() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
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
             patch('click.prompt', return_value=''), \
             patch('click.echo'), patch('click.secho'):
            result = run_new(str(project_path.parent / 'new-project'), None)

        assert result is True


def test_run_new_sets_backup_remote() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'new-project'

        def fake_clone(path: Path) -> bool:
            path.mkdir(parents=True, exist_ok=True)
            (path / 'limen.toml').write_text('backup_remote = ""\n')
            return True

        with patch('limen.cli.commands.new._clone_template', side_effect=fake_clone), \
             patch('click.echo'), patch('click.secho'):
            result = run_new(str(project_path), 'git@github.com:user/repo.git')

        assert result is True
        assert 'git@github.com:user/repo.git' in (project_path / 'limen.toml').read_text()


def test_run_new_clone_failure_returns_false() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'new-project'

        with patch('limen.cli.commands.new._clone_template', return_value=False), \
             patch('click.echo'), patch('click.secho'):
            result = run_new(str(project_path), None)

        assert result is False


def test_run_new_warns_when_backup_remote_placeholder_missing() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'new-project'

        def fake_clone(path: Path) -> bool:
            path.mkdir(parents=True, exist_ok=True)
            (path / 'limen.toml').write_text('# no placeholder here\n')
            return True

        with patch('limen.cli.commands.new._clone_template', side_effect=fake_clone), \
             patch('click.echo'), patch('click.secho'):
            result = run_new(str(project_path), 'git@github.com:user/repo.git')

        assert result is True
        assert 'git@github.com:user/repo.git' not in (project_path / 'limen.toml').read_text()


def test_run_new_fails_when_git_not_found() -> None:
    # _clone_template runs git through git_utils.run_git → git_executable.
    with tempfile.TemporaryDirectory() as d, \
         patch('limen.cli.git_utils.git_executable', side_effect=FileNotFoundError), \
         patch('click.echo'), patch('click.secho'):
        project_path = Path(d) / 'new-project'
        result = run_new(str(project_path), None)
    assert result is False


def test_run_new_skips_backup_remote_with_invalid_chars() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'new-project'

        def fake_clone(path: Path) -> bool:
            path.mkdir(parents=True, exist_ok=True)
            (path / 'limen.toml').write_text('backup_remote = ""\n')
            return True

        with patch('limen.cli.commands.new._clone_template', side_effect=fake_clone), \
             patch('click.echo'), patch('click.secho'):
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

        with patch('limen.cli.git_utils.subprocess.run', side_effect=fake_subprocess), \
             patch('limen.cli.commands.new.shutil.rmtree'), \
             patch('click.echo'), patch('click.secho'):
            result = run_new(str(project_path), None)

        assert result is False


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *args],
        cwd=cwd, capture_output=True, check=True,
    )


def test_clone_template_initializes_on_main_branch() -> None:
    real_run = subprocess.run
    with tempfile.TemporaryDirectory() as d:
        project_path = Path(d) / 'proj'

        def fake_run(args: list, **kwargs: object) -> object:
            if 'clone' in args:
                # Simulate a successful template clone without network access.
                project_path.mkdir(parents=True, exist_ok=True)
                (project_path / 'README.md').write_text('x')
                (project_path / '.git').mkdir()
                result = MagicMock()
                result.returncode = 0
                result.stderr = ''
                return result
            return real_run(args, **kwargs)

        with patch('limen.cli.git_utils.subprocess.run', side_effect=fake_run), \
             patch('click.echo'), patch('click.secho'):
            assert _clone_template(project_path) is True

        branch = real_run(
            ['git', '-C', str(project_path), 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, check=True,
        )
        assert branch.stdout.strip() == 'main'


def test_run_new_from_restores_project_from_backup() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        source = root / 'source'
        source.mkdir()
        (source / 'limen.toml').write_text('[store]\nbackup_remote = "git@example.com:me/p.git"\n')
        (source / 'manifests' / 'committed').mkdir(parents=True)
        (source / 'manifests' / 'committed' / 'a.yaml').write_text('id: a\n')
        _git(['init'], source)
        _git(['add', '.'], source)
        _git(['commit', '-m', 'init'], source)

        dest = root / 'restored'
        with patch('click.echo'), patch('click.secho'):
            result = run_new(str(dest), None, from_remote=str(source))

        assert result is True
        assert (dest / 'limen.toml').exists()
        assert (dest / 'manifests' / 'committed' / 'a.yaml').exists()
        assert (dest / '.git').is_dir()  # history preserved


def test_run_new_from_fails_on_unreachable_remote() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        dest = Path(d) / 'restored'
        result = run_new(str(dest), None, from_remote=str(Path(d) / 'nope.git'))
        assert result is False


def test_run_new_from_fails_when_git_not_found() -> None:
    # The --from path clones via git_utils.git_clone, which resolves git through
    # limen.cli.git_utils.git_executable — patch there so no real clone is attempted.
    with tempfile.TemporaryDirectory() as d, \
         patch('limen.cli.git_utils.git_executable', side_effect=FileNotFoundError), \
         patch('click.echo'), patch('click.secho'):
        dest = Path(d) / 'restored'
        result = run_new(str(dest), None, from_remote='git@example.com:me/p.git')
        assert result is False
