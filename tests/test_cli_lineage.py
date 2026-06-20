import tempfile
from pathlib import Path
from unittest.mock import patch

from limen.cli.commands.fork import run_fork
from limen.cli.commands.lineage import run_lineage
from limen.yaml.store import _SHA256_PREFIX
from limen.yaml.store import commit_manifest


_SAMPLE_YAML = '''\
schema_version: "1.0"
metadata:
  name: root_experiment
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
    yaml_path = root / 'manifests' / 'root.yaml'
    yaml_path.write_text(_SAMPLE_YAML)
    manifest_id, _ = commit_manifest(yaml_path, root)
    return manifest_id


def _fork_and_commit(root: Path, ref: str, name: str, new_lookback: str) -> str:
    run_fork(ref, name, root)
    dest = root / 'manifests' / f'{name}.yaml'
    text = dest.read_text(encoding='utf-8').replace('lookback: [12, 24]', new_lookback)
    text = text.replace('mode: development', 'mode: production')
    dest.write_text(text, encoding='utf-8')
    child_id, _ = commit_manifest(dest, root)
    return child_id


def test_run_lineage_fails_when_no_project() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        assert run_lineage('sha256:' + 'a' * 64, Path(d)) is False


def test_run_lineage_rejects_unknown_manifest() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d).resolve()
        _make_committed(root)
        assert run_lineage('sha256:' + 'f' * 64, root) is False


def test_run_lineage_shows_root_only_for_root_manifest() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo') as echo, patch('click.secho'):
        root = Path(d).resolve()
        manifest_id = _make_committed(root)
        assert run_lineage(manifest_id, root) is True
        printed = ' '.join(str(c.args[0]) for c in echo.call_args_list if c.args)
        assert 'root_experiment' in printed
        assert '← target' in printed


def test_run_lineage_shows_full_chain() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo') as echo, patch('click.secho'):
        root = Path(d).resolve()
        parent_id = _make_committed(root)
        child_id = _fork_and_commit(root, parent_id, 'child', 'lookback: [36, 48]')
        grandchild_id = _fork_and_commit(root, child_id, 'grandchild', 'lookback: [60, 72]')

        echo.reset_mock()
        assert run_lineage(grandchild_id, root) is True
        printed = ' '.join(str(c.args[0]) for c in echo.call_args_list if c.args)
        # all three generations appear, root-first, target marked
        assert 'root_experiment' in printed
        assert 'child' in printed
        assert 'grandchild' in printed
        assert '← target' in printed
        # the root's chain row precedes the target's chain row (rindex skips the header mention)
        assert printed.rindex('root_experiment') < printed.rindex('grandchild')


def test_run_lineage_accepts_short_hash() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d).resolve()
        manifest_id = _make_committed(root)
        short = manifest_id[len(_SHA256_PREFIX):len(_SHA256_PREFIX) + 8]
        assert run_lineage(short, root) is True
