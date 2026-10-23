import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from verify_publication import PublicationHTTPError, fetch, sha256, verify

ACCOUNT = '758576157d0685fa4eb2aeaf16781577'
ORIGIN = 'https://docs.vaquum.fi/limen/'
DEPLOYMENT_SETTLE_SECONDS = 5
DEPLOYMENT_VERIFY_SECONDS = 180
DEPLOYMENT_VERIFY_INTERVAL = 10


def record_http_failure(output: Path, phase: str, release: str | None, version: str, error: Exception) -> None:
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, PublicationHTTPError):
            print(f'Publication HTTP failure [{phase}]: {cause}', flush=True)
            record = {'phase': phase, 'release': release, 'worker_version': version, 'request': cause.evidence}
            try:
                (output / f'{phase}-http-failure.json').write_text(json.dumps(record, indent=2) + '\n')
            except Exception as failure:
                print(f'Publication HTTP receipt failed [{phase}]: {type(failure).__name__}', file=sys.stderr, flush=True)
                raise RuntimeError('Could not retain the publication HTTP failure receipt') from None
            return
        cause = cause.__cause__


def cloudflare(path: str):
    request = Request(f'https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}' + path,
                      headers={'Authorization': 'Bearer ' + os.environ['CLOUDFLARE_API_TOKEN']})
    with urlopen(request, timeout=30) as response:
        record = json.load(response)
    if record['success'] is not True:
        raise ValueError('Cloudflare did not confirm the requested state')
    return record['result']


def require_disconnected_builds() -> None:
    workers = cloudflare('/workers/scripts')
    if not isinstance(workers, list):
        raise ValueError('Cloudflare returned no complete Worker inventory')
    selected = [worker for worker in workers if worker['id'] == 'limen']
    if len(selected) != 1 or not selected[0]['tag']:
        raise ValueError('Cloudflare returned no unique Limen Worker identity')
    triggers = cloudflare('/builds/workers/' + selected[0]['tag'] + '/triggers')
    if not isinstance(triggers, list):
        raise ValueError('Cloudflare returned no complete build-trigger inventory')
    if triggers:
        raise ValueError('Disconnect independent Cloudflare Git builds before activating release-bound documentation')


def require_latest_release(release: str) -> None:
    latest = json.loads(fetch('https://pypi.org/pypi/vaquum-limen/json'))['info']['version']
    if release != 'v' + latest:
        raise ValueError('A newer package release superseded this documentation build')


def current_version(wrangler: Path, config: Path) -> str:
    payload = subprocess.check_output([str(wrangler), 'deployments', 'status', '--config', str(config), '--json'], text=True)
    record = json.loads(payload)
    versions = record['versions']
    if len(versions) != 1 or versions[0]['percentage'] != 100:
        raise ValueError('Documentation deployment requires one fully active rollback version')
    return versions[0]['version_id']


def receipt(path: Path, kind: str):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [item for item in records if item['type'] == kind]
    if len(selected) != 1 or selected[0]['version'] != 1 or selected[0]['worker_name'] != 'limen':
        raise ValueError('Wrangler did not record one Limen deployment operation')
    return selected[0]


def previous_publication():
    try:
        record = json.loads(fetch(ORIGIN + 'publication.json'))
    except HTTPError as error:
        if error.code != 404:
            raise
        print('Bootstrap: the existing renderer has no release evidence; retain its exact Worker version and home-page digest for rollback.')
        return {'mode': 'legacy', 'home_sha256': sha256(fetch(ORIGIN))}
    verify(ORIGIN, record['release'])
    return {'mode': 'released', 'release': record['release'], 'manifest_sha256': sha256(fetch(ORIGIN + 'publication.json')),
            'home_sha256': sha256(fetch(ORIGIN))}


def stage(root: Path, wrangler: Path, release: str):
    config = root / 'docs-site/publisher/wrangler.json'
    output = root / '.docs-evidence/deployment'
    output.mkdir(parents=True, exist_ok=True)
    manifest = (root / 'docs-site/reader-build/publication.json').read_bytes()
    if json.loads(manifest)['release'] != release:
        raise ValueError('The staged site belongs to another release')
    require_disconnected_builds()
    require_latest_release(release)
    version = current_version(wrangler, config)
    previous = previous_publication()
    if current_version(wrangler, config) != version:
        raise ValueError('Production changed while capturing rollback evidence; stage again')
    previous['version_id'] = version
    upload = output / 'upload.ndjson'
    if upload.exists():
        raise ValueError('Use a fresh deployment evidence directory')
    environment = dict(os.environ, WRANGLER_OUTPUT_FILE_PATH=str(upload), WRANGLER_SEND_METRICS='false')
    subprocess.run([str(wrangler), 'versions', 'upload', '--config', str(config), '--tag', release,
                    '--message', 'Validated research book ' + release], cwd=root, env=environment, check=True)
    uploaded = receipt(upload, 'version-upload')
    if not re.fullmatch('[0-9a-f-]{36}', uploaded['version_id'] or '') or not uploaded['preview_url']:
        raise ValueError('The uploaded version has no immutable preview URL')
    preview = uploaded['preview_url'].rstrip('/') + '/limen/'
    verified = verify(preview, release)
    if fetch(preview + 'publication.json') != manifest:
        raise ValueError('The preview serves another publication manifest')
    record = {'release': release, 'version_id': uploaded['version_id'], 'preview_url': preview,
              'manifest_sha256': sha256(manifest), 'previous': previous, 'verification': verified}
    (output / 'staged.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def deploy_version(wrangler: Path, config: Path, version: str, output: Path) -> None:
    environment = dict(os.environ, WRANGLER_OUTPUT_FILE_PATH=str(output), WRANGLER_SEND_METRICS='false')
    subprocess.run([str(wrangler), 'versions', 'deploy', version + '@100%', '--config', str(config), '--yes'],
                   env=environment, check=True)
    receipt(output, 'version-deploy')
    if current_version(wrangler, config) != version:
        raise ValueError('The requested Worker version did not become fully active')
    print(f'Wait {DEPLOYMENT_SETTLE_SECONDS} seconds for global Worker deployment propagation before HTTP verification.')
    time.sleep(DEPLOYMENT_SETTLE_SECONDS)
    if current_version(wrangler, config) != version:
        raise ValueError('The active Worker version changed during deployment propagation')


def verify_active(wrangler: Path, config: Path, version: str, release: str | None,
                  manifest_sha256: str | None, output: Path, phase: str, *, home_sha256: str | None = None):
    if (release is None) != (manifest_sha256 is None) or (release is None and home_sha256 is None):
        raise ValueError('Incomplete publication or legacy rollback identity')
    deadline = time.monotonic() + DEPLOYMENT_VERIFY_SECONDS
    attempt = 0
    while True:
        if current_version(wrangler, config) != version:
            raise ValueError('Another deployment changed the active Worker during verification')
        attempt += 1
        try:
            verified = {'mode': 'legacy'} if release is None else verify(ORIGIN, release)
            if manifest_sha256 is not None and sha256(fetch(ORIGIN + 'publication.json')) != manifest_sha256:
                raise ValueError('Production serves another publication manifest')
            if home_sha256 is not None and sha256(fetch(ORIGIN)) != home_sha256:
                raise ValueError('Rollback did not restore the retained home page')
        except (PublicationHTTPError, ValueError) as error:
            record_http_failure(output, f'{phase}-propagation-{attempt}', release, version, error)
            print(f'Publication proof has not converged [{phase}, attempt {attempt}]: {error}', flush=True)
            cause: BaseException | None = error
            while cause is not None:
                if isinstance(cause, HTTPError) and cause.code != 404:
                    raise
                cause = cause.__cause__
            if isinstance(error, ValueError) and not str(error).startswith((
                'Publication belongs to another release', 'Production serves another publication manifest',
                'The active site member differs from its sealed payload:', 'The deployed 404 differs from the sealed reader',
                'Rollback did not restore the retained home page',
            )):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(DEPLOYMENT_VERIFY_INTERVAL, remaining))
        else:
            if current_version(wrangler, config) != version:
                raise ValueError('Another deployment changed the active Worker during verification')
            return verified


def activate(root: Path, wrangler: Path):
    output = root / '.docs-evidence/deployment'
    staged = json.loads((output / 'staged.json').read_text())
    config = root / 'docs-site/publisher/wrangler.json'
    previous = staged['previous']
    try:
        require_disconnected_builds()
        require_latest_release(staged['release'])
        verify(staged['preview_url'], staged['release'])
        if sha256(fetch(staged['preview_url'] + 'publication.json')) != staged['manifest_sha256']:
            raise ValueError('The staged preview changed before activation')
        if current_version(wrangler, config) != previous['version_id']:
            raise ValueError('Another deployment changed the rollback baseline; stage again')
        require_latest_release(staged['release'])
    except Exception as error:
        record_http_failure(output, 'activation', staged['release'], staged['version_id'], error)
        raise
    try:
        deploy_version(wrangler, config, staged['version_id'], output / 'activation.ndjson')
        require_latest_release(staged['release'])
        verified = verify_active(wrangler, config, staged['version_id'], staged['release'],
                                 staged['manifest_sha256'], output, 'activation')
        require_latest_release(staged['release'])
    except Exception as error:
        print('Activation or production verification failed; restore and verify the previous Worker version.')
        try:
            record_http_failure(output, 'activation', staged['release'], staged['version_id'], error)
        finally:
            try:
                if current_version(wrangler, config) not in (staged['version_id'], previous['version_id']):
                    raise ValueError('Another deployment changed production; refuse automatic rollback')
                deploy_version(wrangler, config, previous['version_id'], output / 'rollback.ndjson')
                if previous['mode'] == 'released':
                    verify_active(wrangler, config, previous['version_id'], previous['release'],
                                  previous['manifest_sha256'], output, 'rollback', home_sha256=previous['home_sha256'])
                elif previous['mode'] == 'legacy':
                    verify_active(wrangler, config, previous['version_id'], None, None,
                                  output, 'rollback', home_sha256=previous['home_sha256'])
                else:
                    raise ValueError('Unknown rollback publication mode')
                (output / 'rollback.json').write_text(json.dumps(previous, indent=2) + '\n')
            except Exception as rollback_error:
                record_http_failure(output, 'rollback', previous.get('release'), previous['version_id'], rollback_error)
                raise
        raise
    (output / 'accepted.json').write_text(json.dumps(staged | {'production': verified}, indent=2) + '\n')
    return verified


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify a released Worker preview before activation; restore and verify its predecessor on failure.')
    parser.add_argument('command', choices=['stage', 'activate'])
    parser.add_argument('--wrangler', type=Path, required=True)
    parser.add_argument('--release')
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    wrangler = arguments.wrangler.resolve()
    if arguments.command == 'stage':
        if not arguments.release:
            parser.error('stage requires --release')
        print(json.dumps(stage(root, wrangler, arguments.release), indent=2))
    else:
        print(json.dumps(activate(root, wrangler), indent=2))
