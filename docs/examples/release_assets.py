import argparse
import json
from pathlib import Path
import re
import subprocess

from build_bundle import build
from verify_publication import fetch, manifest_name, relative_path, sha256

REPOSITORY = 'Vaquum/Limen'


def attach(release: str, assets: list[Path]) -> None:
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', release):
        raise ValueError('Release assets require an exact version tag')
    result = subprocess.check_output(['gh', 'api', f'repos/{REPOSITORY}/releases/tags/{release}'], text=True)
    record = json.loads(result)
    if record['tag_name'] != release or record['draft'] or record['prerelease']:
        raise ValueError('Documentation assets require a published stable GitHub Release')
    existing = {asset['name'] for asset in record['assets']}
    for asset in assets:
        relative_path(asset.name)
        if asset.name not in existing:
            subprocess.run(['gh', 'release', 'upload', release, str(asset), '--repo', REPOSITORY], check=True)
        public = fetch(f'https://github.com/{REPOSITORY}/releases/download/{release}/{asset.name}')
        if sha256(public) != sha256(asset.read_bytes()):
            raise ValueError('An immutable Release asset already has different bytes: ' + asset.name)


def acquire_bundle(root: Path, release: str) -> Path:
    output = root / '.docs-evidence'
    output.mkdir(exist_ok=True)
    source = output / 'examples.tar.gz'
    digest = build(root, source)
    asset = output / f'examples-{digest}.tar.gz'
    asset.write_bytes(source.read_bytes())
    attach(release, [asset])
    downloaded = fetch(f'https://github.com/{REPOSITORY}/releases/download/{release}/{asset.name}')
    if sha256(downloaded) != digest:
        raise ValueError('The anonymously downloaded project bundle changed')
    target = output / 'released-examples.tar.gz'
    target.write_bytes(downloaded)
    return target


def attach_publication(root: Path, release: str) -> None:
    output = root / '.docs-evidence/publication'
    encoded = (output / 'publication.json').read_bytes()
    manifest = json.loads(encoded)
    if manifest['release'] != release:
        raise ValueError('Publication assets belong to another release')
    paths = [output / Path(relative_path(item['path'])).name for item in manifest['artifacts'].values()]
    for path, artifact in zip(paths, manifest['artifacts'].values()):
        if sha256(path.read_bytes()) != artifact['sha256']:
            raise ValueError('The staged publication artifact changed: ' + path.name)
    paths.append(output / manifest_name(encoded))
    if paths[-1].read_bytes() != encoded:
        raise ValueError('The detached manifest changed')
    attach(release, paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retain immutable public release evidence and retrieve the actual pinned projects.')
    parser.add_argument('command', choices=['bundle', 'publication'])
    parser.add_argument('--release', required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if arguments.command == 'bundle':
        print(acquire_bundle(root, arguments.release))
    else:
        attach_publication(root, arguments.release)
