import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from check_evidence import validate_shape
from released_wheel import wheel_entry

ROLES = {'examples', 'native_outputs', 'validation', 'evidence', 'site', 'rollback', 'wheel', 'pypi_metadata'}
USER_AGENT = 'LimenDocsVerifier/1.0 (+https://github.com/Vaquum/Limen)'
ERROR_HEADERS = ('Content-Type', 'Content-Length', 'Cache-Control', 'CF-Cache-Status', 'CF-Ray', 'ETag')


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(netloc=parsed.netloc.rsplit('@', 1)[-1], query='', fragment='').geturl()


class PublicationHTTPError(HTTPError):
    def __init__(self, requested_url: str, error: HTTPError):
        state, read_error = 'complete', None
        try:
            with error:
                body = error.read()
        except IncompleteRead as failure:
            body, state, read_error = failure.partial, 'partial', type(failure).__name__
        except Exception as failure:
            body, state, read_error = None, 'unavailable', type(failure).__name__
        response_url = _safe_url(error.geturl())
        self.evidence = {
            'requested_url': _safe_url(requested_url), 'response_url': response_url,
            'status': error.code,
            'headers': {name: error.headers[name] for name in ERROR_HEADERS if error.headers.get(name) is not None},
            'body_state': state, 'body_bytes': len(body) if body is not None else None,
            'body_sha256': sha256(body) if body is not None else None, 'body_read_error_type': read_error,
        }
        super().__init__(response_url, error.code, json.dumps(self.evidence, sort_keys=True),
                         error.headers, io.BytesIO(body if body is not None else b''))


def _open(url: str):
    anonymous_url(url)
    try:
        return urlopen(Request(url, headers={'User-Agent': USER_AGENT}), timeout=30)
    except HTTPError as error:
        raise PublicationHTTPError(url, error) from None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_name(payload: bytes) -> str:
    return 'publication-' + sha256(payload) + '.json'


def verify_release_manifest(release_url: str, payload: bytes) -> None:
    if fetch(urljoin(release_url.rstrip('/') + '/', manifest_name(payload))) != payload:
        raise ValueError('The active manifest differs from the public GitHub Release')


def verify_site(base_url: str, files: dict[str, bytes]) -> None:
    policy = json.loads(files['security-headers.json'])

    def require_headers(headers):
        for name in ('Content-Security-Policy', 'Permissions-Policy', 'Referrer-Policy', 'Strict-Transport-Security', 'X-Content-Type-Options', 'X-Frame-Options'):
            expected = policy[name]
            if not expected or headers.get(name) != expected:
                raise ValueError('A published response changed its sealed security header: ' + name)

    def member(item):
        name, expected = item
        relative_path(name)
        route = name.removesuffix('index.html') if name.endswith('index.html') else name
        url = urljoin(base_url, route)
        with _open(url) as response:
            anonymous_url(response.geturl())
            if response.read() != expected:
                raise ValueError('The active site member differs from its sealed payload: ' + name)
            require_headers(response.headers)
    with ThreadPoolExecutor(max_workers=8) as readers:
        list(readers.map(member, ((name, data) for name, data in files.items() if name != '404.html')))
    try:
        with _open(urljoin(base_url, '__missing-docs-acceptance__/')):
            raise ValueError('An unknown documentation route did not return 404')
    except HTTPError as error:
        if isinstance(error, PublicationHTTPError) and error.evidence['body_state'] != 'complete':
            raise ValueError('The deployed 404 body could not be read completely') from error
        if error.code != 404 or error.read() != files['404.html']:
            raise ValueError('The deployed 404 differs from the sealed reader') from error
        require_headers(error.headers)


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or str(path) != value or not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
        raise ValueError('Unsafe publication member path: ' + value)
    return value


def anonymous_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost', '::1')):
        raise ValueError('Anonymous publication retrieval requires HTTPS or a local acceptance server')
    if parsed.username is not None or parsed.password is not None:
        raise ValueError('Publication retrieval must not require credentials')


def fetch(url: str) -> bytes:
    with _open(url) as response:
        anonymous_url(response.geturl())
        return response.read()


def archive_files(payload: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        members = archive.getmembers()
        if len({member.name for member in members}) != len(members) or any(not member.isfile() for member in members):
            raise ValueError('Publication archive has duplicate or non-file members')
        files = {}
        for member in members:
            name = relative_path(member.name)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('Publication archive member is unavailable')
            with stream:
                files[name] = stream.read()
        return files


def retrieve(base_url: str, manifest) -> dict[str, bytes]:
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1 or set(manifest['artifacts']) != ROLES:
        raise ValueError('Incomplete publication artifact inventory')
    files = {}
    for role, artifact in manifest['artifacts'].items():
        path = relative_path(artifact['path'])
        if not re.fullmatch('[0-9a-f]{64}', artifact['sha256']):
            raise ValueError('Publication artifact has no SHA-256: ' + role)
        payload = fetch(urljoin(base_url.rstrip('/') + '/', path))
        if sha256(payload) != artifact['sha256']:
            raise ValueError('Anonymous artifact digest differs: ' + role)
        files[role] = payload
    return files


def verify(base_url: str, release: str):
    base_url = base_url.rstrip('/') + '/'
    manifest_bytes = fetch(urljoin(base_url, 'publication.json'))
    manifest = json.loads(manifest_bytes)
    if manifest['release'] != release or not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', release):
        raise ValueError('Publication belongs to another release')
    files = retrieve(base_url, manifest)
    release_url = f'https://github.com/Vaquum/Limen/releases/download/{release}/'
    verify_release_manifest(release_url, manifest_bytes)
    for role, artifact in manifest['artifacts'].items():
        name = PurePosixPath(artifact['path']).name
        if sha256(fetch(release_url + name)) != sha256(files[role]):
            raise ValueError('The static artifact differs from the public GitHub Release: ' + role)
    record = json.loads(files['evidence'])
    schema = json.loads((Path(__file__).parent / 'evidence.schema.json').read_text())
    validate_shape(record, schema)
    commit = manifest['merged_commit']
    if not re.fullmatch('[0-9a-f]{40}', commit) or record['mode'] != 'production' or record['validation']['status'] != 'passed':
        raise ValueError('Publication requires successful merged-release evidence')
    if record['revision']['dirty'] or record['revision']['head'] != commit or record['revision']['tested_commit'] != commit:
        raise ValueError('Publication describes a PR, dirty checkout or another merged revision')
    if release != 'v' + record['package']['version']:
        raise ValueError('Publication package version differs')
    expected = {'release': release, 'merged_commit': commit, 'released_wheel_sha256': sha256(files['wheel']),
                'bundle_sha256': sha256(files['examples']), 'site_sha256': sha256(files['site']),
                'validation_sha256': sha256(files['validation'])}
    if record['publication'] != expected or record['package']['wheel_sha256'] != expected['released_wheel_sha256']:
        raise ValueError('Publication identities disagree with the anonymously retrieved artifacts')
    if record['bundle']['sha256'] != expected['bundle_sha256'] or record['native_outputs']['sha256'] != sha256(files['native_outputs']):
        raise ValueError('Publication project or native result identity differs')
    execution = json.loads(files['validation'])
    if execution != record | {'mode': 'candidate', 'publication': None}:
        raise ValueError('Publication changed the complete execution record')
    version = record['package']['version']
    retained = wheel_entry(json.loads(files['pypi_metadata']), version)
    current = wheel_entry(json.loads(fetch(f'https://pypi.org/pypi/vaquum-limen/{version}/json')), version)
    if retained['digests']['sha256'] != current['digests']['sha256'] or current['digests']['sha256'] != expected['released_wheel_sha256']:
        raise ValueError('Publication did not execute the actual public PyPI wheel')
    site = archive_files(files['site'])
    rollback = archive_files(files['rollback'])
    if any(rollback.get(name) != data for name, data in site.items()):
        raise ValueError('Rollback does not contain the validated site payload')
    if rollback.get('evidence/current.json') != files['evidence']:
        raise ValueError('Rollback contains another execution envelope')
    for source in record['sources']:
        if source['status'] != 'merged' or not re.fullmatch(r'https://github.com/Vaquum/Limen/pull/[1-9][0-9]*', source['merged_pr'] or '') or not re.fullmatch('[0-9a-f]{40}', source['commit']):
            raise ValueError('Publication advertises unmerged source')
        if source['region'] == '__file__' and sha256(site['source/' + relative_path(source['path'])]) != source['sha256']:
            raise ValueError('Publication source file differs from its evidence')
    verify_site(base_url, rollback)
    return {'release': release, 'merged_commit': commit, 'artifacts': len(files), 'source_records': len(record['sources'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify a published research book and its source, results, wheel and rollback without credentials.')
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--release', required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.base_url, arguments.release), indent=2))
