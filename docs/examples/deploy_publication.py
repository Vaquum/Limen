import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from verify_publication import fetch, sha256, verify

ACCOUNT = '758576157d0685fa4eb2aeaf16781577'
ORIGIN = 'https://docs.vaquum.fi/limen/'


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
    require_latest_release(release)
    previous = previous_publication()
    previous['version_id'] = current_version(wrangler, config)
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


def activate(root: Path, wrangler: Path):
    output = root / '.docs-evidence/deployment'
    staged = json.loads((output / 'staged.json').read_text())
    config = root / 'docs-site/publisher/wrangler.json'
    require_disconnected_builds()
    require_latest_release(staged['release'])
    verify(staged['preview_url'], staged['release'])
    if sha256(fetch(staged['preview_url'] + 'publication.json')) != staged['manifest_sha256']:
        raise ValueError('The staged preview changed before activation')
    previous = staged['previous']
    if current_version(wrangler, config) != previous['version_id']:
        raise ValueError('Another deployment changed the rollback baseline; stage again')
    try:
        deploy_version(wrangler, config, staged['version_id'], output / 'activation.ndjson')
        verified = verify(ORIGIN, staged['release'])
        if sha256(fetch(ORIGIN + 'publication.json')) != staged['manifest_sha256']:
            raise ValueError('Production serves another publication manifest')
    except Exception:
        print('Activation or production verification failed; restore and verify the previous Worker version.')
        deploy_version(wrangler, config, previous['version_id'], output / 'rollback.ndjson')
        if sha256(fetch(ORIGIN)) != previous['home_sha256']:
            raise ValueError('Rollback did not restore the retained home page')
        if previous['mode'] == 'released':
            verify(ORIGIN, previous['release'])
            if sha256(fetch(ORIGIN + 'publication.json')) != previous['manifest_sha256']:
                raise ValueError('Rollback restored another publication envelope')
        (output / 'rollback.json').write_text(json.dumps(previous, indent=2) + '\n')
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
