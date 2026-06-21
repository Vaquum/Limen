import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from limen.cli.commands.results import run_results

_CSV = (
    'id,val_score,auc,accuracy,C,scaler_type\n'
    'h1,,0.90,0.80,0.1,robust\n'
    'h2,,0.70,0.85,1.0,rank_gauss\n'
    'h3,,0.95,0.82,0.1,rank_gauss\n'
)
_MANIFEST_ID = 'sha256:' + 'a' * 64
_META = {
    'manifest_id': _MANIFEST_ID,
    'yaml_reference': {'sfd': {'params': {
        'C': [0.1, 1.0], 'scaler_type': ['robust', 'rank_gauss'], 'fixed': [1],
    }}},
}


def _make_run(root: Path) -> Path:
    rdir = root / 'run'
    rdir.mkdir()
    (rdir / 'results.csv').write_text(_CSV)
    (rdir / 'metadata.json').write_text(json.dumps(_META))
    return rdir


def _echoed(echo: object) -> str:
    return '\n'.join(str(c.args[0]) for c in echo.call_args_list if c.args)


def test_run_results_ranks_by_default_metric() -> None:
    # val_score is all-null, so the default falls through to auc; h3 (0.95) is top.
    with tempfile.TemporaryDirectory() as d, patch('click.echo') as echo, patch('click.secho'):
        rdir = _make_run(Path(d))
        assert run_results(rdir, None, 5, False, False) is True
        out = _echoed(echo)
        assert 'auc' in out
        assert out.index('h3') < out.index('h2')  # h3 ranked above h2
        assert 'manifest sha256:aaaaaaaa' in out  # manifest id in the header


def test_run_results_metric_override() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo') as echo, patch('click.secho'):
        rdir = _make_run(Path(d))
        assert run_results(rdir, 'accuracy', 1, False, False) is True
        assert 'h2' in _echoed(echo)  # accuracy 0.85 highest


def test_run_results_ascending() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo') as echo, patch('click.secho'):
        rdir = _make_run(Path(d))
        assert run_results(rdir, 'auc', 1, True, False) is True
        assert 'h2' in _echoed(echo)  # lowest auc 0.70


def test_run_results_json_emits_ranked_records() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo') as echo, patch('click.secho'):
        rdir = _make_run(Path(d))
        assert run_results(rdir, 'auc', 2, False, True) is True
        data = json.loads(_echoed(echo))
        assert data['manifest_id'] == _MANIFEST_ID
        assert data['metric'] == 'auc'
        assert data['permutations'] == 3
        assert [r['id'] for r in data['results']] == ['h3', 'h1']
        assert data['results'][0]['scaler_type'] == 'rank_gauss'


def test_run_results_missing_csv() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        assert run_results(Path(d), None, 5, False, False) is False


def test_run_results_unknown_metric() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        rdir = _make_run(Path(d))
        assert run_results(rdir, 'nonexistent', 5, False, False) is False
