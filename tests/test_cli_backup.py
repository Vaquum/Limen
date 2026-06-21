import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from limen.cli.commands.backup import run_backup


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *args],
        cwd=cwd, capture_output=True, check=True,
    )


def _make_project(root: Path, backup_remote: str) -> Path:
    project = root / 'proj'
    project.mkdir()
    (project / 'limen.toml').write_text(f'[store]\nbackup_remote = "{backup_remote}"\n')
    (project / 'manifests' / 'committed').mkdir(parents=True)
    (project / 'manifests' / 'committed' / 'a.yaml').write_text('id: a\n')
    _git(['init'], project)
    _git(['add', '.'], project)
    _git(['commit', '-m', 'init'], project)
    return project


def test_run_backup_fails_when_no_project() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        assert run_backup(Path(d)) is False


def test_run_backup_fails_when_no_remote_configured() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d)
        _make_project(root, '')
        assert run_backup(root / 'proj') is False


def test_run_backup_pushes_to_bare_remote() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        bare = root / 'backup.git'
        _git(['init', '--bare', str(bare)], root)
        project = _make_project(root, str(bare))

        with patch('click.echo'), patch('click.secho'):
            assert run_backup(project) is True

        # The bare remote received the commit — clone it and verify the file.
        check = root / 'check'
        _git(['clone', str(bare), str(check)], root)
        assert (check / 'manifests' / 'committed' / 'a.yaml').exists()


def test_run_backup_fails_on_unreachable_remote() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d)
        project = _make_project(root, str(root / 'does-not-exist.git'))
        assert run_backup(project) is False


def test_run_backup_hints_on_https_credential_error() -> None:
    with tempfile.TemporaryDirectory() as d, \
         patch('click.echo'), patch('click.secho') as secho, \
         patch('limen.cli.commands.backup.git_push',
               return_value=(False, "fatal: could not read Username for 'https://github.com'")):
        root = Path(d)
        project = _make_project(root, 'https://github.com/user/repo.git')
        assert run_backup(project) is False
        printed = ' '.join(str(c.args[0]) for c in secho.call_args_list if c.args)
        assert 'SSH URL' in printed


def test_run_backup_commits_uncommitted_work_before_pushing() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        bare = root / 'backup.git'
        _git(['init', '--bare', str(bare)], root)
        project = _make_project(root, str(bare))
        # A committed-manifest result the user produced but never git-committed.
        (project / 'results' / 'abc123').mkdir(parents=True)
        (project / 'results' / 'abc123' / 'results.csv').write_text('metric\n1\n')

        with patch('click.echo'), patch('click.secho'):
            assert run_backup(project) is True

        check = root / 'check'
        _git(['clone', str(bare), str(check)], root)
        assert (check / 'results' / 'abc123' / 'results.csv').exists()


def test_run_backup_excludes_gitignored_dev_results() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        bare = root / 'backup.git'
        _git(['init', '--bare', str(bare)], root)
        project = _make_project(root, str(bare))
        (project / '.gitignore').write_text('results/dev/\n')
        (project / 'results' / 'dev' / 'run1').mkdir(parents=True)
        (project / 'results' / 'dev' / 'run1' / 'out.csv').write_text('x\n')

        with patch('click.echo'), patch('click.secho'):
            assert run_backup(project) is True

        check = root / 'check'
        _git(['clone', str(bare), str(check)], root)
        assert not (check / 'results' / 'dev').exists()


def test_run_backup_hints_on_diverged_remote() -> None:
    with tempfile.TemporaryDirectory() as d, \
         patch('click.echo'), patch('click.secho') as secho, \
         patch('limen.cli.commands.backup.git_push',
               return_value=(False, '! [rejected] main -> main (fetch first)\nUpdates were rejected')):
        root = Path(d)
        project = _make_project(root, 'git@github.com:user/repo.git')
        assert run_backup(project) is False
        printed = ' '.join(str(c.args[0]) for c in secho.call_args_list if c.args)
        assert 'force' in printed.lower()
        assert 'empty repository' in printed
