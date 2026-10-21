import io
import json
import functools
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tarfile
from threading import Thread

import pytest

from build_bundle import write_archive
from publication import restore, validated_site_files
from verify_publication import ROLES, USER_AGENT, archive_files, fetch, manifest_name, relative_path, retrieve, sha256, verify_release_manifest, verify_site

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def anonymous_server():
    payload = (ROOT / 'docs/examples/retained/python-selection.json').read_bytes()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(dict(self.headers))
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/release/', payload, requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_anonymous_release_retrieval(anonymous_server):
    base, payload, requests = anonymous_server
    assert fetch(base + 'retained-python.json') == payload
    assert len(requests) == 1
    assert requests[0]['User-Agent'] == USER_AGENT
    assert not any(name.lower() in ('authorization', 'cookie') for name in requests[0])


def test_publication_manifest_rebuild_preserves_the_previous_release_identity(tmp_path):
    previous = (ROOT / 'docs/examples/retained/python-selection.json').read_bytes()
    current = (ROOT / 'docs/examples/retained/compiled-selection.json').read_bytes()
    assert previous != current
    for payload in (previous, current):
        (tmp_path / manifest_name(payload)).write_bytes(payload)
    (tmp_path / 'publication.json').write_bytes(current)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        base = f'http://127.0.0.1:{server.server_port}/'
        assert fetch(base + 'publication.json') == current
        verify_release_manifest(base, previous)
        verify_release_manifest(base, current)
        (tmp_path / manifest_name(previous)).write_bytes(current)
        with pytest.raises(ValueError, match='manifest differs'):
            verify_release_manifest(base, previous)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_every_published_member_and_the_404_must_match_the_sealed_bytes(tmp_path):
    source = (ROOT / 'docs/examples/python-research/research.py').read_bytes()
    observation = (ROOT / 'docs/examples/retained/python-selection.json').read_bytes()
    missing = (ROOT / 'docs/examples/retained/compiled-selection.json').read_bytes()
    policy = (ROOT / 'docs-site/src/security-headers.json').read_bytes()
    headers = json.loads(policy)
    missing_headers = dict(headers)
    files = {'index.html': observation, 'chapter/index.html': observation, 'source/research.py': source, '404.html': missing,
             'security-headers.json': policy}
    agents = []
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(tmp_path), **kwargs)

        def do_GET(self):
            agents.append(self.headers.get('User-Agent'))
            super().do_GET()

        def end_headers(self):
            response_headers = missing_headers if self.path == '/__missing-docs-acceptance__/' else headers
            for name, value in response_headers.items():
                self.send_header(name, value)
            super().end_headers()

        def send_error(self, code, message=None, explain=None):
            self.send_response(code)
            self.send_header('Content-Length', str(len(missing)))
            self.end_headers()
            self.wfile.write(missing)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        base = f'http://127.0.0.1:{server.server_port}/'
        verify_site(base, files)
        (tmp_path / 'source/research.py').write_bytes(observation)
        with pytest.raises(ValueError, match='member differs.*source/research.py'):
            verify_site(base, files)
        (tmp_path / 'source/research.py').write_bytes(source)
        with pytest.raises(ValueError, match='404 differs'):
            verify_site(base, files | {'404.html': observation})
        for name, expected in json.loads(policy).items():
            headers[name] = "default-src 'none'" if name == 'Content-Security-Policy' else 'changed'
            with pytest.raises(ValueError, match='sealed security header: ' + name):
                verify_site(base, files)
            headers[name] = expected
        missing_headers['Content-Security-Policy'] = "default-src 'none'"
        with pytest.raises(ValueError, match='sealed security header: Content-Security-Policy'):
            verify_site(base, files)
        assert agents and set(agents) == {USER_AGENT}
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_anonymous_transport_cannot_accept_changed_artifact(anonymous_server):
    base, _, _ = anonymous_server
    manifest = {'schema_version': 1, 'artifacts': {role: {'path': 'changed.json', 'sha256': '0' * 64} for role in ROLES}}
    with pytest.raises(ValueError, match='Anonymous artifact digest differs'):
        retrieve(base, manifest)
    del manifest['artifacts']['rollback']
    with pytest.raises(ValueError, match='Incomplete publication'):
        retrieve(base, manifest)


@pytest.mark.parametrize('path', ['../outside', '/outside', 'a/../outside', 'a//b', './a', 'https://example.invalid/file', 'a%2fb'])
def test_publication_paths_cannot_escape_the_release(path):
    with pytest.raises(ValueError, match='Unsafe publication member'):
        relative_path(path)


def test_archives_retain_real_source_and_reject_links(tmp_path):
    source = (ROOT / 'docs/examples/python-research/research.py').read_bytes()
    archive = tmp_path / 'source.tar.gz'
    write_archive({'python-research/research.py': source}, archive)
    assert archive_files(archive.read_bytes()) == {'python-research/research.py': source}
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as tar:
        link = tarfile.TarInfo('source.py')
        link.type, link.linkname = tarfile.SYMTYPE, '../outside'
        tar.addfile(link)
    with pytest.raises(ValueError, match='non-file members'):
        archive_files(output.getvalue())


def test_rollback_restores_the_site_and_detached_evidence(tmp_path):
    source = (ROOT / 'docs/examples/python-research/research.py').read_bytes()
    archive = tmp_path / 'rollback.tar.gz'
    digest = write_archive({'source/python-research/research.py': source}, archive)
    relative = f'evidence/rollback-{digest}.tar.gz'
    manifest = tmp_path / 'publication.json'
    manifest.write_text(json.dumps({'artifacts': {'rollback': {'path': relative, 'sha256': digest}}}))
    target = tmp_path / 'restored'
    restore(manifest, archive, target)
    assert (target / 'source/python-research/research.py').read_bytes() == source
    assert (target / 'publication.json').read_bytes() == manifest.read_bytes()
    assert sha256((target / relative).read_bytes()) == digest
    with pytest.raises(ValueError, match='fresh directory'):
        restore(manifest, archive, target)
    archive.write_bytes(archive.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='differs from its detached'):
        restore(manifest, archive, tmp_path / 'changed')


def test_unvalidated_page_or_asset_cannot_enter_the_sealed_build(tmp_path):
    source = (ROOT / 'docs/examples/python-research/research.py').read_bytes()
    member = tmp_path / 'source/research.py'
    member.parent.mkdir()
    member.write_bytes(source)
    built = {'files': {'source/research.py': sha256(source)}}
    (tmp_path / 'build.json').write_text(json.dumps(built))
    assert validated_site_files(tmp_path, built)['source/research.py'] == source
    for name in ('book/extra/index.html', 'assets/extra.mjs'):
        added = tmp_path / name
        added.parent.mkdir(parents=True)
        added.write_bytes(source)
        with pytest.raises(ValueError, match='inventory changed'):
            validated_site_files(tmp_path, built)
        added.unlink()
    member.unlink()
    with pytest.raises(ValueError, match='inventory changed'):
        validated_site_files(tmp_path, built)


@contextmanager
def publication_failure_server(rollback_fails=True):
    from urllib.parse import urlsplit
    body = (ROOT / 'docs/examples/retained/python-selection.json').read_bytes() + b'\nbody-private-marker'
    policy_bytes = (ROOT / 'docs-site/src/security-headers.json').read_bytes()
    policy = json.loads(policy_bytes)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            requests.append(format % args)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/redirect':
                self.send_response(302)
                self.send_header('Location', '/missing?signature=query-private-marker#fragment-private-marker')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            missing = path.endswith('__missing-docs-acceptance__/')
            status, payload = 404, body
            if path.endswith('security-headers.json'):
                status, payload = 200, policy_bytes
            elif path.startswith(('/site/', '/good/', '/partial404/')) and not missing:
                status = 200
            elif missing and path.startswith('/site/'):
                status = 503
            elif path == '/phase/rollback' and not rollback_fails:
                status = 200
            partial = path == '/partial' or (missing and path.startswith('/partial404/'))
            if path == '/partial':
                status = 503
            self.send_response(status, 'reason-private-marker')
            for name, value in policy.items():
                self.send_header(name, value)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload) + (10 if partial else 0)))
            self.send_header('Cache-Control', 'public, max-age=0, must-revalidate')
            self.send_header('CF-Cache-Status', 'MISS')
            self.send_header('CF-Ray', 'a4dcf905fa80687c-ARN')
            self.send_header('ETag', '"d51935c8f779b2122d9dfa84c51f4053"')
            self.send_header('Set-Cookie', 'cookie-private-marker')
            self.send_header('Location', '/private?signature=header-private-marker')
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            self.close_connection = True

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', body, policy_bytes, requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_publication_http_failure_retains_safe_request_evidence():
    import subprocess
    import sys
    from urllib.error import HTTPError
    from verify_publication import ERROR_HEADERS, PublicationHTTPError
    command = 'import sys; sys.path.insert(0, "docs/examples"); from verify_publication import fetch; fetch(sys.argv[1])'
    forbidden = ('query-private-marker', 'fragment-private-marker', 'reason-private-marker',
                 'body-private-marker', 'cookie-private-marker', 'header-private-marker', 'read-private-marker')
    assert ERROR_HEADERS == ('Content-Type', 'Content-Length', 'Cache-Control', 'CF-Cache-Status', 'CF-Ray', 'ETag')
    with publication_failure_server() as (base, body, policy, requests):
        for route, status, state in [('/redirect', 404, 'complete'), ('/partial', 503, 'partial')]:
            url = base + route + '?signature=query-private-marker#fragment-private-marker'
            result = subprocess.run([sys.executable, '-c', command, url], cwd=ROOT, capture_output=True, text=True)
            rendered = result.stdout + result.stderr
            assert result.returncode != 0
            assert all(secret not in rendered for secret in forbidden)
            assert 'PublicationHTTPError' in rendered and '"status": ' + str(status) in rendered
            assert '"body_state": "' + state + '"' in rendered
            assert sha256(body) in rendered
            with pytest.raises(HTTPError) as caught:
                fetch(url)
            error = caught.value
            assert isinstance(error, PublicationHTTPError)
            assert error.evidence['requested_url'] == base + route
            assert error.evidence['response_url'] == base + ('/missing' if route == '/redirect' else route)
            assert error.evidence['body_bytes'] == len(body)
            assert error.read() == body
            assert set(error.evidence['headers']) == set(ERROR_HEADERS)
            assert error.evidence['body_read_error_type'] == ('IncompleteRead' if state == 'partial' else None)
        files = {'index.html': body, 'security-headers.json': policy, '404.html': body}
        verify_site(base + '/good/', files)
        with pytest.raises(ValueError, match='could not be read completely'):
            verify_site(base + '/partial404/', files)
        count = len(requests)
        refused = subprocess.run([sys.executable, '-c', command, base.replace('://', '://userinfo-private-marker@') + '/missing'],
                                 cwd=ROOT, capture_output=True, text=True)
        assert refused.returncode != 0 and len(requests) == count
        assert 'must not require credentials' in refused.stderr
        assert 'userinfo-private-marker' not in refused.stdout + refused.stderr
        unreadable = '''import sys, io
from email.message import Message
from urllib.error import HTTPError
sys.path.insert(0, "docs/examples")
import verify_publication as verifier
class Body(io.BytesIO):
    def read(self, *args, **kwargs):
        raise OSError("read-private-marker")
def open_error(request, timeout):
    raise HTTPError(request.full_url, 503, "reason-private-marker", Message(), Body())
verifier.urlopen = open_error
verifier.fetch(sys.argv[1])
'''
        result = subprocess.run([sys.executable, '-c', unreadable, base + '/unreadable?signature=query-private-marker'],
                                cwd=ROOT, capture_output=True, text=True)
        rendered = result.stdout + result.stderr
        assert result.returncode != 0 and all(secret not in rendered for secret in forbidden)
        assert 'PublicationHTTPError' in rendered and '"status": 503' in rendered
        assert '"body_state": "unavailable"' in rendered and '"body_sha256": null' in rendered
        assert '"body_read_error_type": "OSError"' in rendered
