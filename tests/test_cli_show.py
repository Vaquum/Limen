import tempfile
from pathlib import Path
from unittest.mock import patch

from limen.cli.commands.show import run_show
from limen.yaml.store import _SHA256_PREFIX
from limen.yaml.store import commit_manifest


_SAMPLE_YAML = '''\
schema_version: "1.0"
metadata:
  name: shown_experiment
  mode: production
sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.get_spot_klines
    reference_architecture: limen.sfd.reference_architecture.logreg_binary.logreg_binary
    target:
      name: target
      class: limen.targets.QuantileBinaryTarget
  params:
    lookback: [12, 24]
uel:
  n_permutations: 4
'''


def _make_committed(root: Path) -> str:
    root = root.resolve()
    (root / 'limen.toml').write_text('')
    (root / 'manifests' / 'committed').mkdir(parents=True)
    yaml_path = root / 'm.yaml'
    yaml_path.write_text(_SAMPLE_YAML)
    manifest_id, _ = commit_manifest(yaml_path, root)
    return manifest_id


def test_run_show_fails_when_no_project() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        assert run_show('sha256:' + 'a' * 64, Path(d)) is False


def test_run_show_prints_manifest_yaml() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo') as echo, patch('click.secho'):
        root = Path(d).resolve()
        manifest_id = _make_committed(root)
        assert run_show(manifest_id, root) is True
        out = ''.join(str(c.args[0]) for c in echo.call_args_list if c.args)
        assert 'schema_version' in out
        assert 'shown_experiment' in out


def test_run_show_accepts_short_hash() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d).resolve()
        manifest_id = _make_committed(root)
        short = manifest_id[len(_SHA256_PREFIX):len(_SHA256_PREFIX) + 8]
        assert run_show(short, root) is True


def test_run_show_rejects_unknown() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d).resolve()
        _make_committed(root)
        assert run_show('sha256:' + 'f' * 64, root) is False
