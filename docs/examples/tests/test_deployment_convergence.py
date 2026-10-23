import io
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

import deploy_publication as deployment
from verify_publication import PublicationHTTPError, sha256


@pytest.fixture
def active_publication(monkeypatch, tmp_path):
    # Technical transport identities; no research observations are fabricated.
    clock = [0]
    calls = []
    manifest = b'{"transport":"sealed-publication"}'
    monkeypatch.setattr(deployment.time, 'monotonic', lambda: clock[0])

    def sleep(seconds):
        clock[0] += seconds
        calls.append(('wait', seconds))

    monkeypatch.setattr(deployment.time, 'sleep', sleep)
    monkeypatch.setattr(deployment, 'current_version', lambda *args: 'selected-version')
    monkeypatch.setattr(deployment, 'fetch', lambda url: manifest)
    monkeypatch.setattr(deployment, 'DEPLOYMENT_VERIFY_SECONDS', 20)
    return tmp_path, calls, sha256(manifest)


def http_failure(status):
    return PublicationHTTPError('https://docs.vaquum.fi/limen/evidence/site.tar.gz',
                                HTTPError('https://docs.vaquum.fi/limen/evidence/site.tar.gz', status,
                                          'Transport failure', Message(), io.BytesIO(b'old-version-404')))


@pytest.mark.parametrize('phase', ['activation', 'rollback'])
def test_full_proof_must_converge_before_acceptance(monkeypatch, active_publication, phase):
    output, calls, digest = active_publication
    failures = [http_failure(404), ValueError('Publication belongs to another release')]
    proof = {'release': 'v5.46.2', 'artifacts': 8}

    def verify(origin, release):
        calls.append(('full-proof', origin, release))
        if failures:
            raise failures.pop(0)
        return proof

    monkeypatch.setattr(deployment, 'verify', verify)
    assert deployment.verify_active(Path('wrangler'), Path('config'), 'selected-version', 'v5.46.2', digest, output, phase) == proof
    assert [item[0] for item in calls] == ['full-proof', 'wait', 'full-proof', 'wait', 'full-proof']
    assert (output / f'{phase}-propagation-1-http-failure.json').exists()


@pytest.mark.parametrize('failure', [ValueError('The active site member differs from its sealed payload: index.html'), http_failure(404)])
def test_permanent_failure_exhausts_the_propagation_window(monkeypatch, active_publication, failure):
    output, calls, digest = active_publication

    def verify(*args):
        calls.append(('full-proof',))
        raise failure

    monkeypatch.setattr(deployment, 'verify', verify)
    with pytest.raises(type(failure)) as caught:
        deployment.verify_active(Path('wrangler'), Path('config'), 'selected-version', 'v5.46.2', digest, output, 'activation')
    assert caught.value is failure
    assert sum(item[1] for item in calls if item[0] == 'wait') == 20
    assert len([item for item in calls if item[0] == 'full-proof']) == 3


@pytest.mark.parametrize('status', [401, 403, 500])
def test_non_404_http_failure_is_immediate(monkeypatch, active_publication, status):
    output, calls, digest = active_publication
    failure = http_failure(status)

    def verify(*args):
        raise failure

    monkeypatch.setattr(deployment, 'verify', verify)
    with pytest.raises(PublicationHTTPError) as caught:
        deployment.verify_active(Path('wrangler'), Path('config'), 'selected-version', 'v5.46.2', digest, output, 'activation')
    assert caught.value is failure
    assert calls == []


@pytest.mark.parametrize('versions', [
    ['another-version'],
    ['selected-version', 'another-version'],
])
def test_concurrent_deployment_cannot_be_accepted(monkeypatch, active_publication, versions):
    output, calls, digest = active_publication
    states = iter(versions)
    monkeypatch.setattr(deployment, 'current_version', lambda *args: next(states))
    monkeypatch.setattr(deployment, 'verify', lambda *args: {'artifacts': 8})
    with pytest.raises(ValueError, match='Another deployment changed'):
        deployment.verify_active(Path('wrangler'), Path('config'), 'selected-version', 'v5.46.2', digest, output, 'activation')
    assert calls == []


def test_manifest_digest_is_part_of_the_complete_acceptance(monkeypatch, active_publication):
    output, calls, digest = active_publication
    manifests = iter([b'old-publication', b'{"transport":"sealed-publication"}'])
    proofs = []

    def verify(*args):
        proofs.append(True)
        return {'artifacts': 8}

    monkeypatch.setattr(deployment, 'fetch', lambda *args: next(manifests))
    monkeypatch.setattr(deployment, 'verify', verify)
    assert deployment.verify_active(Path('wrangler'), Path('config'), 'selected-version', 'v5.46.2', digest, output, 'activation') == {'artifacts': 8}
    assert len(proofs) == 2
    assert calls == [('wait', 10)]


def test_missing_route_wrapped_503_is_not_a_propagation_retry(monkeypatch, active_publication):
    from test_publication_http import publication_failure_server
    from verify_publication import verify_site
    output, calls, digest = active_publication
    with publication_failure_server() as (base, body, policy, requests):
        def verify(*args):
            verify_site(base + '/site/activation/', {'index.html': body, '404.html': body, 'security-headers.json': policy})

        monkeypatch.setattr(deployment, 'verify', verify)
        with pytest.raises(ValueError, match='deployed 404 differs') as caught:
            deployment.verify_active(Path('wrangler'), Path('config'), 'selected-version', 'v5.46.2', digest, output, 'activation')
        assert isinstance(caught.value.__cause__, PublicationHTTPError)
        assert caught.value.__cause__.code == 503
        assert calls == []
        assert len(requests) == 3


@pytest.mark.parametrize('released', [True, False])
def test_rollback_home_digest_converges_with_the_other_proof(monkeypatch, active_publication, released):
    output, calls, digest = active_publication
    homes = iter([b'pre-propagation-home', b'retained-home'])
    proofs = []

    def fetch(url):
        return next(homes) if url == deployment.ORIGIN else b'{"transport":"sealed-publication"}'

    def verify(*args):
        proofs.append(True)
        return {'artifacts': 8}

    monkeypatch.setattr(deployment, 'fetch', fetch)
    monkeypatch.setattr(deployment, 'verify', verify)
    assert deployment.verify_active(Path('wrangler'), Path('config'), 'selected-version', 'v5.46.1' if released else None,
                                    digest if released else None, output, 'rollback', home_sha256=sha256(b'retained-home')) == (
                                        {'artifacts': 8} if released else {'mode': 'legacy'})
    assert len(proofs) == (2 if released else 0)
    assert calls == [('wait', 10)]
