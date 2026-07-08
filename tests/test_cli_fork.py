import tempfile
from pathlib import Path
from unittest.mock import patch

from limen.cli.commands.fork import run_fork
from limen.yaml.parser import parse
from limen.yaml.store import SHA256_PREFIX
from limen.yaml.store import commit_manifest


_SAMPLE_YAML = '''\
schema_version: "1.0"
metadata:
  name: parent_experiment
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
    yaml_path = root / 'manifests' / 'parent.yaml'
    yaml_path.write_text(_SAMPLE_YAML)
    manifest_id, _ = commit_manifest(yaml_path, root)
    return manifest_id


def test_run_fork_fails_when_no_project() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        assert run_fork('sha256:' + 'a' * 64, 'child', Path(d)) is False


def test_run_fork_creates_working_copy_with_lineage() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d).resolve()
        manifest_id = _make_committed(root)
        assert run_fork(manifest_id, 'child', root) is True
        dest = root / 'manifests' / 'child.yaml'
        assert dest.exists()
        data, errors = parse(dest.read_text(encoding='utf-8'))
        assert errors == []
        assert data['metadata']['name'] == 'child'
        assert data['metadata']['mode'] == 'development'
        assert data['lineage']['parent_id'] == manifest_id


def test_run_fork_accepts_short_hash() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d).resolve()
        manifest_id = _make_committed(root)
        short = manifest_id[len(SHA256_PREFIX):len(SHA256_PREFIX) + 8]
        assert run_fork(short, 'child', root) is True
        assert (root / 'manifests' / 'child.yaml').exists()


def test_run_fork_rejects_unknown_manifest() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d).resolve()
        _make_committed(root)
        assert run_fork('sha256:' + 'f' * 64, 'child', root) is False


def test_run_fork_rejects_invalid_name() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d).resolve()
        manifest_id = _make_committed(root)
        assert run_fork(manifest_id, 'bad name', root) is False


def test_run_fork_refuses_to_overwrite() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d).resolve()
        manifest_id = _make_committed(root)
        (root / 'manifests' / 'child.yaml').write_text('existing')
        assert run_fork(manifest_id, 'child', root) is False
        assert (root / 'manifests' / 'child.yaml').read_text() == 'existing'


def test_run_fork_prompts_for_name_with_default() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'), \
            patch('click.prompt', return_value='parent_experiment_v2') as prompt:
        root = Path(d).resolve()
        manifest_id = _make_committed(root)
        assert run_fork(manifest_id, None, root) is True
        assert prompt.call_args.kwargs['default'] == 'parent_experiment_v2'
        assert (root / 'manifests' / 'parent_experiment_v2.yaml').exists()


def test_run_fork_default_name_increments_past_existing() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'), \
            patch('click.prompt', return_value='parent_experiment_v3') as prompt:
        root = Path(d).resolve()
        manifest_id = _make_committed(root)
        (root / 'manifests' / 'parent_experiment_v2.yaml').write_text('taken')
        assert run_fork(manifest_id, None, root) is True
        assert prompt.call_args.kwargs['default'] == 'parent_experiment_v3'
