import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tarfile
from threading import Thread

import pytest

from build_bundle import write_archive
from publication import restore
from verify_publication import ROLES, archive_files, fetch, relative_path, retrieve, sha256

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
    assert not any(name.lower() in ('authorization', 'cookie') for name in requests[0])


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
