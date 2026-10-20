import argparse
import json
from pathlib import Path
import shutil
import subprocess

from build_bundle import write_archive
from check_evidence import maintained_documents, validate_evidence
from released_wheel import verify_download
from verify_publication import archive_files, relative_path, sha256


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(['git', *arguments], cwd=root, text=True).strip()


def release_origin(root: Path, record, release: str) -> None:
    version = record['package']['version']
    head = git(root, 'rev-parse', 'HEAD')
    if release != 'v' + version or record['revision']['head'] != head or record['revision']['tested_commit'] != head or record['revision']['tree'] != git(root, 'rev-parse', 'HEAD^{tree}'):
        raise ValueError('Publication requires the exact tested release checkout, not a PR merge')
    if record['mode'] != 'candidate' or record['publication'] is not None or record['revision']['dirty']:
        raise ValueError('Publication requires an unchanged complete execution record')
    if any(source['status'] != 'merged' or not source['merged_pr'] for source in record['sources']):
        raise ValueError('Publication cannot promote unmerged source')
    if git(root, 'rev-parse', f'refs/tags/{release}^{{commit}}') != head:
        raise ValueError('Release tag does not identify the executed commit')
    subprocess.run(['git', 'merge-base', '--is-ancestor', head, 'origin/main'], cwd=root, check=True)
    if git(root, 'status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Publication source tree changed after execution')
    catalogue = json.loads((root / 'docs/examples/catalogue.json').read_text())
    if set(catalogue['candidate_pages']) != {str(path.relative_to(root)) for path in maintained_documents(root)}:
        raise ValueError('Production requires the complete maintained corpus, not selected candidate pages')


def validate_production(root: Path, record) -> None:
    publication = record['publication']
    execution = (root / '.docs-evidence/execution.json').read_bytes()
    wheel = verify_download(root / '.docs-evidence/released-wheel.json', record['package']['version'])
    expected = {'release': 'v' + record['package']['version'], 'merged_commit': record['revision']['head'],
                'released_wheel_sha256': sha256(wheel.read_bytes()), 'bundle_sha256': record['bundle']['sha256'],
                'site_sha256': publication['site_sha256'], 'validation_sha256': sha256(execution)}
    if publication != expected or record['package']['wheel_sha256'] != expected['released_wheel_sha256']:
        raise ValueError('Released package and publication identities disagree')
    candidate = record | {'mode': 'candidate', 'publication': None}
    if json.loads(execution) != candidate:
        raise ValueError('Production changed the complete execution record')
    release_origin(root, candidate, publication['release'])
    payload = root / '.docs-evidence/publication' / ('site-' + publication['site_sha256'] + '.tar.gz')
    if sha256(payload.read_bytes()) != publication['site_sha256']:
        raise ValueError('Published site payload changed')


def site_files(site: Path) -> dict[str, bytes]:
    files = {}
    for path in sorted(site.rglob('*')):
        if path.is_symlink():
            raise ValueError('The publication site cannot contain a symlink')
        if path.is_file():
            files[str(path.relative_to(site))] = path.read_bytes()
    return files


def validated_site_files(site: Path, built) -> dict[str, bytes]:
    original = site_files(site)
    if set(original) != set(built['files']) | {'build.json'} or 'publication.json' in original:
        raise ValueError('The built reader inventory changed or was already published')
    if any(sha256(original[name]) != value for name, value in built['files'].items()):
        raise ValueError('The built reader member changed')
    return original


def prepare(root: Path, release: str):
    execution = (root / '.docs-evidence/current.json').read_bytes()
    record = json.loads(execution)
    release_origin(root, record, release)
    validate_evidence(record, root, head=git(root, 'rev-parse', 'HEAD'), tree=git(root, 'rev-parse', 'HEAD^{tree}'), version=record['package']['version'])
    receipt = root / '.docs-evidence/released-wheel.json'
    wheel = verify_download(receipt, record['package']['version'])
    if sha256(wheel.read_bytes()) != record['package']['wheel_sha256']:
        raise ValueError('The executed wheel is not the downloaded public release')
    site = root / 'docs-site/reader-build'
    built = json.loads((site / 'build.json').read_text())
    if built['evidence_sha256'] != sha256(execution) or built['version'] != record['package']['version'] or (built['head'], built['tested_commit'], built['tree']) != (record['revision']['head'], record['revision']['tested_commit'], record['revision']['tree']):
        raise ValueError('The built reader belongs to another execution')
    original = validated_site_files(site, built)
    output = root / '.docs-evidence/publication'
    output.mkdir(exist_ok=True)
    artifacts = {}

    def artifact(role: str, payload: bytes, suffix: str):
        digest = sha256(payload)
        name = f'{role}-{digest}.{suffix}'
        (output / name).write_bytes(payload)
        target = site / 'evidence' / name
        target.write_bytes(payload)
        artifacts[role] = {'path': 'evidence/' + name, 'sha256': digest}

    payload = {name: data for name, data in original.items() if not name.startswith('evidence/') and name != 'build.json'}
    staging = output / 'site.tar.gz'
    payload_sha = write_archive(payload, staging)
    artifact('site', staging.read_bytes(), 'tar.gz')
    staging.unlink()
    (root / '.docs-evidence/execution.json').write_bytes(execution)
    artifact('validation', execution, 'json')
    artifact('examples', (root / record['bundle']['path']).read_bytes(), 'tar.gz')
    artifact('native_outputs', (root / record['native_outputs']['path']).read_bytes(), 'tar.gz')
    artifact('wheel', wheel.read_bytes(), 'whl')
    artifact('pypi_metadata', (receipt.parent / f"pypi-{record['package']['version']}.json").read_bytes(), 'json')
    record['mode'] = 'production'
    record['publication'] = {'release': release, 'merged_commit': record['revision']['head'],
                             'released_wheel_sha256': sha256(wheel.read_bytes()), 'bundle_sha256': record['bundle']['sha256'],
                             'site_sha256': payload_sha, 'validation_sha256': sha256(execution)}
    production = (json.dumps(record, indent=2, allow_nan=False) + '\n').encode()
    validate_production(root, record)
    (root / '.docs-evidence/production.json').write_bytes(production)
    (site / 'evidence/current.json').write_bytes(production)
    artifact('evidence', production, 'json')
    built['evidence_sha256'] = sha256(production)
    built['files'] = {name: sha256(data) for name, data in site_files(site).items() if name != 'build.json'}
    (site / 'build.json').write_text(json.dumps(built) + '\n')
    staging = output / 'rollback.tar.gz'
    write_archive(site_files(site), staging)
    artifact('rollback', staging.read_bytes(), 'tar.gz')
    staging.unlink()
    manifest = {'schema_version': 1, 'release': release, 'merged_commit': record['revision']['head'], 'artifacts': artifacts}
    encoded = (json.dumps(manifest, indent=2) + '\n').encode()
    (output / 'publication.json').write_bytes(encoded)
    (site / 'publication.json').write_bytes(encoded)
    return manifest


def restore(manifest_path: Path, archive_path: Path, target: Path):
    manifest = json.loads(manifest_path.read_text())
    rollback_path = relative_path(manifest['artifacts']['rollback']['path'])
    archive = archive_path.read_bytes()
    if sha256(archive) != manifest['artifacts']['rollback']['sha256']:
        raise ValueError('Rollback archive differs from its detached publication manifest')
    files = archive_files(archive)
    if target.exists():
        raise ValueError('Restore into a fresh directory before activation')
    target.mkdir(parents=True)
    for name, data in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    rollback = target / rollback_path
    rollback.parent.mkdir(parents=True, exist_ok=True)
    rollback.write_bytes(archive)
    shutil.copyfile(manifest_path, target / 'publication.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seal a completely validated released research book for publication.')
    parser.add_argument('--release', required=True)
    arguments = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[2], arguments.release), indent=2))
