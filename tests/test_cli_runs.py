import tempfile
from pathlib import Path
from unittest.mock import patch

from limen.cli.commands.runs import run_runs


def test_run_runs_fails_when_no_project() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        assert run_runs(Path(d)) is False


def test_run_runs_returns_true_when_no_runs() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo') as echo, patch('click.secho'):
        root = Path(d)
        (root / 'limen.toml').write_text('')
        assert run_runs(root) is True
        out = ' '.join(str(c.args[0]) for c in echo.call_args_list if c.args)
        assert 'No runs yet' in out


def test_run_runs_lists_dev_and_committed() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo') as echo, patch('click.secho'):
        root = Path(d)
        (root / 'limen.toml').write_text('')
        dev = root / 'results' / 'dev' / 'exp_1'
        dev.mkdir(parents=True)
        (dev / 'results.csv').write_text('id,auc\nh1,0.9\nh2,0.8\n')
        committed = root / 'results' / 'abc123' / 'ts'
        committed.mkdir(parents=True)
        (committed / 'results.csv').write_text('id,auc\nh1,0.9\n')
        (committed / 'metadata.json').write_text('{"manifest_id": "sha256:' + 'b' * 64 + '"}')

        assert run_runs(root) is True
        out = ' '.join(str(c.args[0]) for c in echo.call_args_list if c.args)
        assert 'results/dev/exp_1' in out
        assert '[dev]' in out
        assert 'results/abc123/ts' in out
        assert '[committed]' in out
        assert '2 perms' in out
        assert 'sha256:bbbbbbbb' in out  # committed run shows its manifest id
        assert '—' in out  # dev run without metadata shows a placeholder
