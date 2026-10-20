import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import urlopen


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wheel_entry(metadata, version: str):
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version):
        raise ValueError('A released package requires an exact version')
    if (metadata['info']['name'].replace('_', '-').lower(), metadata['info']['version']) != ('vaquum-limen', version):
        raise ValueError('PyPI metadata belongs to another package or version')
    filename = f'vaquum_limen-{version}-py3-none-any.whl'
    wheels = [entry for entry in metadata['urls'] if entry['packagetype'] == 'bdist_wheel']
    if len(wheels) != 1 or wheels[0]['filename'] != filename or wheels[0]['yanked']:
        raise ValueError('Expected one available platform-independent Limen wheel')
    entry = wheels[0]
    url = urlparse(entry['url'])
    if url.scheme != 'https' or url.netloc != 'files.pythonhosted.org' or not url.path.endswith('/' + filename):
        raise ValueError('The released wheel must come from the public PyPI file host')
    if not re.fullmatch('[0-9a-f]{64}', entry['digests']['sha256']):
        raise ValueError('PyPI did not provide a wheel SHA-256')
    return entry


def download(version: str, output: Path) -> Path:
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version):
        raise ValueError('A released package requires an exact version')
    with urlopen(f'https://pypi.org/pypi/vaquum-limen/{version}/json', timeout=30) as response:
        metadata_bytes = response.read()
    entry = wheel_entry(json.loads(metadata_bytes), version)
    with urlopen(entry['url'], timeout=30) as response:
        wheel_bytes = response.read()
    if sha256(wheel_bytes) != entry['digests']['sha256'] or len(wheel_bytes) != entry['size']:
        raise ValueError('Downloaded wheel differs from the public PyPI release')
    output.mkdir(parents=True, exist_ok=True)
    (output / entry['filename']).write_bytes(wheel_bytes)
    (output / f'pypi-{version}.json').write_bytes(metadata_bytes)
    receipt = output / 'released-wheel.json'
    receipt.write_text(json.dumps({
        'release': 'v' + version, 'version': version, 'filename': entry['filename'],
        'metadata_sha256': sha256(metadata_bytes), 'wheel_sha256': sha256(wheel_bytes),
        'url': entry['url'],
    }, indent=2) + '\n')
    verify_download(receipt, version)
    return receipt


def verify_download(receipt: Path, version: str) -> Path:
    record = json.loads(receipt.read_text())
    metadata = receipt.parent / f'pypi-{version}.json'
    if (record['release'], record['version']) != ('v' + version, version) or sha256(metadata.read_bytes()) != record['metadata_sha256']:
        raise ValueError('Released-wheel receipt belongs to another release or metadata')
    entry = wheel_entry(json.loads(metadata.read_text()), version)
    if (record['filename'], record['wheel_sha256'], record['url']) != (entry['filename'], entry['digests']['sha256'], entry['url']):
        raise ValueError('Released-wheel receipt differs from the retained PyPI metadata')
    wheel = receipt.parent / entry['filename']
    if sha256(wheel.read_bytes()) != record['wheel_sha256'] or wheel.stat().st_size != entry['size']:
        raise ValueError('Released wheel changed after download')
    return wheel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download an actual published Limen wheel and retain its public PyPI identity.')
    parser.add_argument('--version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    print(download(arguments.version, arguments.output).resolve())
