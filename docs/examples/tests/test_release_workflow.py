import copy
import json
from pathlib import Path
import subprocess

import pytest

from deploy_publication import activate, require_latest_release, stage
from release_assets import attach
from release_identity import validate_identity

ROOT = Path(__file__).resolve().parents[3]


def test_release_proof_has_its_markdown_runtime_before_any_python_collection():
    from ruamel.yaml import YAML
    workflow = YAML(typ='safe').load((ROOT / '.github/workflows/docs_publish.yml').read_text())
    steps = workflow['jobs']['publish']['steps']
    node = next(index for index, step in enumerate(steps) if step.get('uses', '').startswith('actions/setup-node@'))
    dependencies = next(index for index, step in enumerate(steps) if 'npm ci --prefix docs-site\n' in step.get('run', ''))
    proof = next(index for index, step in enumerate(steps) if 'python -m pytest ' in step.get('run', ''))
    assert node < dependencies < proof


@pytest.fixture
def release_checkout(tmp_path):
    def run(*args):
        return subprocess.check_output(['git', '-c', 'user.name=Research book test', '-c', 'user.email=research-book@example.invalid', *args], cwd=tmp_path, text=True).strip()
    run('init', '-q', '-b', 'main')
    (tmp_path / 'README.md').write_bytes((ROOT / 'README.md').read_bytes())
    run('add', 'README.md')
    run('commit', '-qm', 'docs: retain the research introduction')
    commit = run('rev-parse', 'HEAD')
    run('tag', 'v5.45.3')
    run('update-ref', 'refs/remotes/origin/main', commit)
    metadata = json.loads((ROOT / 'docs/examples/tests/fixtures/pypi-v5.45.3.json').read_text())
    wheel = next(item for item in metadata['urls'] if item['packagetype'] == 'bdist_wheel')
    record = {'schema_version': 1, 'run_id': 39356406676, 'release': 'v5.45.3', 'commit': commit,
              'wheel': {'filename': wheel['filename'], 'sha256': wheel['digests']['sha256']}}
    return tmp_path, record


def test_publishing_identity_requires_the_exact_run_tag_and_merged_commit(release_checkout):
    checkout, record = release_checkout
    validate_identity(checkout, record, record['run_id'])
    changed = copy.deepcopy(record)
    changed['commit'] = '0' * 40
    with pytest.raises(ValueError, match='another commit'):
        validate_identity(checkout, changed, record['run_id'])
    with pytest.raises(ValueError, match='another publishing run'):
        validate_identity(checkout, record, record['run_id'] + 1)
    changed = copy.deepcopy(record)
    changed['wheel']['filename'] = 'wheel-' + changed['wheel']['sha256'] + '.whl'
    with pytest.raises(ValueError, match='wheel identity'):
        validate_identity(checkout, changed, record['run_id'])
    subprocess.run(['git', 'update-ref', '-d', 'refs/remotes/origin/main'], cwd=checkout, check=True)
    with pytest.raises(subprocess.CalledProcessError):
        validate_identity(checkout, record, record['run_id'])


def test_release_assets_never_replace_existing_different_bytes(monkeypatch, tmp_path):
    source = ROOT / 'docs/examples/retained/python-selection.json'
    asset = tmp_path / source.name
    asset.write_bytes(source.read_bytes())
    release = {'tag_name': 'v5.45.3', 'draft': False, 'prerelease': False, 'assets': [{'name': source.name}]}
    monkeypatch.setattr('release_assets.subprocess.check_output', lambda *args, **kwargs: json.dumps(release))
    writes = []
    monkeypatch.setattr('release_assets.subprocess.run', lambda *args, **kwargs: writes.append(args))
    monkeypatch.setattr('release_assets.fetch', lambda url: source.read_bytes())
    attach('v5.45.3', [asset])
    assert writes == []
    monkeypatch.setattr('release_assets.fetch', lambda url: (ROOT / 'docs/examples/retained/compiled-selection.json').read_bytes())
    with pytest.raises(ValueError, match='different bytes'):
        attach('v5.45.3', [asset])
    assert writes == []


def test_an_older_queued_release_cannot_activate(monkeypatch):
    metadata = (ROOT / 'docs/examples/tests/fixtures/pypi-v5.45.3.json').read_bytes()
    monkeypatch.setattr('deploy_publication.fetch', lambda url: metadata)
    require_latest_release('v5.45.3')
    with pytest.raises(ValueError, match='superseded'):
        require_latest_release('v5.44.5')


def test_failed_production_verification_restores_and_checks_the_previous_version(monkeypatch, tmp_path):
    from verify_publication import sha256
    output = tmp_path / '.docs-evidence/deployment'
    output.mkdir(parents=True)
    current = (ROOT / 'docs/examples/retained/python-selection.json').read_bytes()
    previous = (ROOT / 'docs/examples/retained/compiled-selection.json').read_bytes()
    staged = {'release': 'v5.45.3', 'version_id': 'd3143e09-4594-47f7-8344-4dd3d113b414',
              'preview_url': 'https://preview.example.invalid/limen/', 'manifest_sha256': sha256(current),
              'previous': {'version_id': '9d51a409-ca8e-44ca-ab4d-7a0a3958e84d', 'home_sha256': sha256(previous), 'mode': 'legacy'}}
    (output / 'staged.json').write_text(json.dumps(staged))
    operations = []
    monkeypatch.setattr('deploy_publication.require_disconnected_builds', lambda: operations.append('disconnected'))
    monkeypatch.setattr('deploy_publication.require_latest_release', lambda release: operations.append('latest'))
    monkeypatch.setattr('deploy_publication.current_version', lambda *args: staged['previous']['version_id'])
    monkeypatch.setattr('deploy_publication.fetch', lambda url: current if 'preview.' in url else previous)

    def verify(url, release):
        operations.append('verify ' + url)
        if url == 'https://docs.vaquum.fi/limen/':
            raise ValueError('injected production verification failure')
        return {'release': release}

    def deploy(wrangler, config, version, receipt):
        operations.append('deploy ' + version)

    monkeypatch.setattr('deploy_publication.verify', verify)
    monkeypatch.setattr('deploy_publication.deploy_version', deploy)
    with pytest.raises(ValueError, match='injected production'):
        activate(tmp_path, Path('wrangler'))
    assert operations == ['disconnected', 'latest', 'verify ' + staged['preview_url'],
                          'deploy ' + staged['version_id'], 'verify https://docs.vaquum.fi/limen/',
                          'deploy ' + staged['previous']['version_id']]
    assert json.loads((output / 'rollback.json').read_text()) == staged['previous']
    assert not (output / 'accepted.json').exists()


def test_staging_rejects_a_version_change_during_rollback_capture(monkeypatch, tmp_path):
    from verify_publication import sha256
    site = tmp_path / 'docs-site/reader-build'
    site.mkdir(parents=True)
    (site / 'publication.json').write_text(json.dumps({'release': 'v5.45.3'}))
    operations = []
    versions = iter(['d3143e09-4594-47f7-8344-4dd3d113b414', '9d51a409-ca8e-44ca-ab4d-7a0a3958e84d'])
    monkeypatch.setattr('deploy_publication.require_disconnected_builds', lambda: operations.append('disconnected'))
    monkeypatch.setattr('deploy_publication.require_latest_release', lambda release: operations.append('latest'))

    def current(*args):
        operations.append('version')
        return next(versions)

    def previous():
        operations.append('capture')
        return {'mode': 'legacy', 'home_sha256': sha256((ROOT / 'README.md').read_bytes())}

    def forbidden(*args, **kwargs):
        raise AssertionError('Staging uploaded a version after its rollback identity changed')

    monkeypatch.setattr('deploy_publication.current_version', current)
    monkeypatch.setattr('deploy_publication.previous_publication', previous)
    monkeypatch.setattr('deploy_publication.subprocess.run', forbidden)
    with pytest.raises(ValueError, match='Production changed while capturing rollback evidence'):
        stage(tmp_path, Path('wrangler'), 'v5.45.3')
    assert operations == ['disconnected', 'latest', 'version', 'capture', 'version']
    assert not (tmp_path / '.docs-evidence/deployment/staged.json').exists()
