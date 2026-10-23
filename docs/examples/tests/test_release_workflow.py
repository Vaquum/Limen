import copy
import json
from pathlib import Path
import subprocess

import pytest

from deploy_publication import activate, deploy_version, require_latest_release, stage
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


@pytest.mark.parametrize('competing_deployment', [False, True])
def test_failed_production_verification_restores_and_checks_the_previous_version(monkeypatch, tmp_path, competing_deployment):
    from verify_publication import sha256
    output = tmp_path / '.docs-evidence/deployment'
    output.mkdir(parents=True)
    current = (ROOT / 'docs/examples/retained/python-selection.json').read_bytes()
    previous = (ROOT / 'docs/examples/retained/compiled-selection.json').read_bytes()
    staged = {'release': 'v5.45.3', 'version_id': 'd3143e09-4594-47f7-8344-4dd3d113b414',
              'preview_url': 'https://preview.example.invalid/limen/', 'manifest_sha256': sha256(current),
              'previous': {'version_id': '9d51a409-ca8e-44ca-ab4d-7a0a3958e84d', 'home_sha256': sha256(previous), 'mode': 'legacy'}}
    (output / 'staged.json').write_text(json.dumps(staged))
    state = {'version': staged['previous']['version_id']}
    operations = []
    monkeypatch.setattr('deploy_publication.require_disconnected_builds', lambda: operations.append('disconnected'))
    monkeypatch.setattr('deploy_publication.require_latest_release', lambda release: operations.append('latest'))
    monkeypatch.setattr('deploy_publication.current_version', lambda *args: state['version'])
    monkeypatch.setattr('deploy_publication.fetch', lambda url: current if 'preview.' in url else previous)

    def verify(url, release):
        operations.append('verify ' + url)
        if url == 'https://docs.vaquum.fi/limen/':
            if competing_deployment:
                state['version'] = 'separately-deployed-version'
            raise ValueError('injected production verification failure')
        return {'release': release}

    def deploy(wrangler, config, version, receipt):
        operations.append('deploy ' + version)
        state['version'] = version

    monkeypatch.setattr('deploy_publication.verify', verify)
    monkeypatch.setattr('deploy_publication.deploy_version', deploy)
    with pytest.raises(ValueError, match='refuse automatic rollback' if competing_deployment else 'injected production'):
        activate(tmp_path, Path('wrangler'))
    assert operations == ['disconnected', 'latest', 'verify ' + staged['preview_url'], 'latest',
                          'deploy ' + staged['version_id'], 'latest', 'verify https://docs.vaquum.fi/limen/'] + (
                              [] if competing_deployment else ['deploy ' + staged['previous']['version_id']])
    if competing_deployment:
        assert state['version'] == 'separately-deployed-version'
        assert not (output / 'rollback.json').exists()
    else:
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


@pytest.mark.parametrize('phase', ['preview', 'deployment', 'production'])
def test_a_release_superseded_during_verification_or_activation_is_not_accepted(monkeypatch, tmp_path, phase):
    from verify_publication import sha256
    output = tmp_path / '.docs-evidence/deployment'
    output.mkdir(parents=True)
    current = (ROOT / 'docs/examples/retained/python-selection.json').read_bytes()
    previous = (ROOT / 'docs/examples/retained/compiled-selection.json').read_bytes()
    staged = {'release': 'v5.45.3', 'version_id': 'd3143e09-4594-47f7-8344-4dd3d113b414',
              'preview_url': 'https://preview.example.invalid/limen/', 'manifest_sha256': sha256(current),
              'previous': {'version_id': '9d51a409-ca8e-44ca-ab4d-7a0a3958e84d', 'home_sha256': sha256(previous), 'mode': 'legacy'}}
    (output / 'staged.json').write_text(json.dumps(staged))
    state = {'superseded': False, 'version': staged['previous']['version_id']}
    deployments = []

    def latest(release):
        assert release == staged['release']
        if state['superseded']:
            raise ValueError('A newer package release superseded this documentation build')

    def verify(url, release):
        if (phase == 'preview' and url == staged['preview_url']) or (phase == 'production' and url == 'https://docs.vaquum.fi/limen/'):
            state['superseded'] = True
        return {'release': release}

    def deploy(wrangler, config, version, receipt):
        deployments.append(version)
        state['version'] = version
        if phase == 'deployment' and version == staged['version_id']:
            state['superseded'] = True

    def fetch(url):
        return current if url.startswith(staged['preview_url']) or state['version'] == staged['version_id'] else previous

    monkeypatch.setattr('deploy_publication.require_disconnected_builds', lambda: None)
    monkeypatch.setattr('deploy_publication.require_latest_release', latest)
    monkeypatch.setattr('deploy_publication.current_version', lambda *args: state['version'])
    monkeypatch.setattr('deploy_publication.verify', verify)
    monkeypatch.setattr('deploy_publication.deploy_version', deploy)
    monkeypatch.setattr('deploy_publication.fetch', fetch)
    with pytest.raises(ValueError, match='superseded'):
        activate(tmp_path, Path('wrangler'))
    assert state['version'] == staged['previous']['version_id']
    assert not (output / 'accepted.json').exists()
    if phase == 'preview':
        assert deployments == []
        assert not (output / 'rollback.json').exists()
    else:
        assert deployments == [staged['version_id'], staged['previous']['version_id']]
        assert json.loads((output / 'rollback.json').read_text()) == staged['previous']


@pytest.mark.parametrize('production_failure', [False, True])
@pytest.mark.parametrize('propagation_seconds', [2, 40])
def test_public_verification_waits_for_activation_and_rollback_propagation(monkeypatch, tmp_path, production_failure, propagation_seconds):
    from email.message import Message
    from urllib.error import HTTPError
    from verify_publication import PublicationHTTPError, sha256
    output = tmp_path / '.docs-evidence/deployment'
    output.mkdir(parents=True)
    current = (ROOT / 'docs/examples/retained/python-selection.json').read_bytes()
    previous = (ROOT / 'docs/examples/retained/compiled-selection.json').read_bytes()
    staged = {'release': 'v5.45.8', 'version_id': '9c62f8e9-12c5-4341-8948-4d181a5a028e',
              'preview_url': 'https://9c62f8e9-limen.mailme-758.workers.dev/limen/', 'manifest_sha256': sha256(current),
              'previous': {'version_id': '7272c591-c9c1-4d1b-9c08-437acca7437d', 'home_sha256': sha256(previous), 'mode': 'legacy'}}
    (output / 'staged.json').write_text(json.dumps(staged))
    state = {'clock': 0, 'version': staged['previous']['version_id'], 'public': staged['previous']['version_id'], 'pending': None}
    deployments = []

    def propagate():
        if state['pending'] is not None and state['clock'] >= state['pending'][1]:
            state['public'] = state['pending'][0]
            state['pending'] = None

    def run(command, *, env, check):
        assert check is True
        state['clock'] += 3
        propagate()
        version = command[3].removesuffix('@100%')
        deployments.append(version)
        state['version'] = version
        state['pending'] = (version, state['clock'] + propagation_seconds)
        Path(env['WRANGLER_OUTPUT_FILE_PATH']).write_text(json.dumps({'type': 'version-deploy', 'version': 1, 'worker_name': 'limen'}) + '\n')

    def sleep(seconds):
        state['clock'] += seconds
        propagate()

    def fetch(url):
        if url.startswith(staged['preview_url']):
            return current
        propagate()
        if url.endswith('publication.json') and state['public'] != staged['version_id']:
            raise PublicationHTTPError(url, HTTPError(url, 404, 'Not Found', Message(), None))
        return current if state['public'] == staged['version_id'] else previous

    def verify(url, release):
        if url == 'https://docs.vaquum.fi/limen/':
            fetch(url + 'publication.json')
            if production_failure:
                raise ValueError('injected altered public content')
        return {'release': release}

    monkeypatch.setattr('deploy_publication.require_disconnected_builds', lambda: None)
    monkeypatch.setattr('deploy_publication.require_latest_release', lambda release: None)
    monkeypatch.setattr('deploy_publication.current_version', lambda *args: state['version'])
    monkeypatch.setattr('deploy_publication.subprocess.run', run)
    monkeypatch.setattr('deploy_publication.time.sleep', sleep)
    monkeypatch.setattr('deploy_publication.time.monotonic', lambda: state['clock'])
    monkeypatch.setattr('deploy_publication.fetch', fetch)
    monkeypatch.setattr('deploy_publication.verify', verify)
    if production_failure:
        with pytest.raises(ValueError, match='injected altered public content'):
            activate(tmp_path, Path('wrangler'))
        assert deployments == [staged['version_id'], staged['previous']['version_id']]
        assert state['public'] == staged['previous']['version_id']
        assert json.loads((output / 'rollback.json').read_text()) == staged['previous']
        assert not (output / 'accepted.json').exists()
    else:
        activate(tmp_path, Path('wrangler'))
        assert deployments == [staged['version_id']]
        assert state['public'] == staged['version_id']
        assert json.loads((output / 'accepted.json').read_text())['production'] == {'release': staged['release']}
        assert not (output / 'rollback.json').exists()


@pytest.mark.parametrize('phase', ['activation', 'settling'])
def test_deployment_identity_must_hold_through_the_propagation_barrier(monkeypatch, tmp_path, phase):
    expected = '9c62f8e9-12c5-4341-8948-4d181a5a028e'
    previous = '7272c591-c9c1-4d1b-9c08-437acca7437d'
    versions = iter([previous] if phase == 'activation' else [expected, previous])
    monkeypatch.setattr('deploy_publication.subprocess.run', lambda *args, **kwargs: None)
    monkeypatch.setattr('deploy_publication.receipt', lambda *args: {})
    monkeypatch.setattr('deploy_publication.current_version', lambda *args: next(versions))
    monkeypatch.setattr('deploy_publication.time.sleep', lambda seconds: None)
    with pytest.raises(ValueError, match='did not become fully active' if phase == 'activation' else 'changed during deployment propagation'):
        deploy_version(Path('wrangler'), tmp_path / 'wrangler.json', expected, tmp_path / 'operation.ndjson')


@pytest.mark.parametrize('kind,rollback_fails,legacy,write_failure', [
    ('artifact', False, False, None),
    ('artifact', True, False, None),
    ('wrapped', True, False, None),
    ('headers', True, False, None),
    ('non-http', True, False, None),
    ('artifact', True, True, None),
    ('artifact', False, False, 'activation'),
    ('artifact', True, False, 'rollback'),
    ('wrapped', True, False, 'activation'),
    ('preflight', False, False, None),
])
def test_activation_and_rollback_http_failure_receipts(tmp_path, kind, rollback_fails, legacy, write_failure):
    import sys
    from test_publication_http import publication_failure_server
    from verify_publication import sha256
    output = tmp_path / '.docs-evidence/deployment'
    output.mkdir(parents=True)
    with publication_failure_server(rollback_fails) as (base, body, _, _):
        previous = {'version_id': '1d7aee75-aef0-4f4f-ae57-4a84afb9fa9c', 'home_sha256': sha256(body),
                    'mode': 'legacy' if legacy else 'released'}
        if not legacy:
            previous.update(release='v5.45.19', manifest_sha256=sha256(body))
        staged = {'release': 'v5.45.20', 'version_id': 'efc46603-a29e-475c-a9c1-3bf8c86b8f2a',
                  'preview_url': base + '/preview/', 'manifest_sha256': sha256(body), 'previous': previous}
        (output / 'staged.json').write_text(json.dumps(staged))
        script = '''import json, sys
from pathlib import Path
sys.path.insert(0, "docs/examples")
import deploy_publication as deployment
from verify_publication import fetch, verify_site
root, base, kind, rollback_fails, write_failure = json.loads(sys.argv[1])
root = Path(root)
output = root / ".docs-evidence/deployment"
staged = json.loads((output / "staged.json").read_text())
state = {"version": staged["previous"]["version_id"], "deployments": []}
body = (Path("docs/examples/retained/python-selection.json").read_bytes() + b"\\nbody-private-marker")
policy = Path("docs-site/src/security-headers.json").read_bytes()
write = Path.write_text
def deploy(wrangler, config, version, receipt):
    state["version"] = version
    state["deployments"].append(version)
    write(output / "operations.json", json.dumps(state["deployments"]))
def publication_fetch(url):
    if url == deployment.ORIGIN and staged["previous"]["mode"] == "legacy" and rollback_fails:
        return fetch(base + "/phase/rollback")
    return body
def verification(url, release):
    if url == staged["preview_url"]:
        if kind == "preflight":
            fetch(base + "/phase/activation?token=query-private-marker#fragment-private-marker")
        return {"release": release}
    phase = "activation" if state["version"] == staged["version_id"] else "rollback"
    if phase == "activation" and kind == "non-http":
        raise ValueError("injected non-HTTP activation failure")
    if phase == "rollback" and not rollback_fails:
        return {"release": release}
    if kind in ("wrapped", "headers"):
        route = "/headers/" if kind == "headers" else "/site/"
        verify_site(base + route + phase + "/", {"index.html": body, "404.html": body, "security-headers.json": policy})
    else:
        fetch(base + "/phase/" + phase + "?token=query-private-marker#fragment-private-marker")
    raise AssertionError("The real HTTP failure did not propagate")
def retain(path, data, *args, **kwargs):
    if path.name == str(write_failure) + "-http-failure.json":
        raise OSError("storage-private-marker")
    return write(path, data, *args, **kwargs)
deployment.require_disconnected_builds = lambda: None
deployment.require_latest_release = lambda release: None
deployment.current_version = lambda *args: state["version"]
deployment.deploy_version = deploy
deployment.fetch = publication_fetch
deployment.verify = verification
deployment.DEPLOYMENT_VERIFY_SECONDS = 0
Path.write_text = retain
deployment.activate(root, Path("isolated-control-plane"))
'''
        arguments = [str(tmp_path), base, kind, rollback_fails, write_failure]
        result = subprocess.run([sys.executable, '-c', script, json.dumps(arguments)], cwd=ROOT, capture_output=True, text=True)
        rendered = result.stdout + result.stderr
        assert result.returncode != 0
        assert all(secret not in rendered for secret in ('query-private-marker', 'fragment-private-marker', 'reason-private-marker',
                   'body-private-marker', 'cookie-private-marker', 'header-private-marker', 'storage-private-marker'))
        operations = output / 'operations.json'
        if kind == 'preflight':
            assert not operations.exists()
        else:
            assert json.loads(operations.read_text()) == [staged['version_id'], previous['version_id']]
        for phase in ('activation', 'rollback'):
            expected = (kind != 'non-http' if phase == 'activation' else rollback_fails)
            path = output / f'{phase}-http-failure.json'
            assert path.exists() == (expected and phase != write_failure)
            if not expected:
                continue
            assert 'Publication HTTP failure [' + phase + ']' in rendered
            assert sha256(body) in rendered
            if phase == write_failure:
                assert 'Publication HTTP receipt failed [' + phase + ']: OSError' in rendered
            else:
                text = path.read_text()
                assert all(secret not in text for secret in ('private-marker', 'Set-Cookie', 'Location'))
                record = json.loads(text)
                assert record['phase'] == phase
                assert record['release'] == (staged['release'] if phase == 'activation' else previous.get('release'))
                assert record['worker_version'] == (staged['version_id'] if phase == 'activation' else previous['version_id'])
                request = record['request']
                assert request['status'] == (503 if kind == 'wrapped' else 404)
                assert request['body_state'] == 'complete' and request['body_bytes'] == len(body)
                assert request['body_sha256'] == sha256(body)
                assert request['body_read_error_type'] is None
                if kind in ('wrapped', 'headers'):
                    endpoint = ('/headers/' if kind == 'headers' else '/site/') + phase + '/__missing-docs-acceptance__/'
                else:
                    endpoint = '/phase/' + phase
                assert request['requested_url'] == request['response_url'] == base + endpoint
        assert not (output / 'accepted.json').exists()
        assert (output / 'rollback.json').exists() == (kind != 'preflight' and not rollback_fails)
        if (output / 'rollback.json').exists():
            assert json.loads((output / 'rollback.json').read_text()) == previous
