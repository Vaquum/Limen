import argparse
from email.parser import BytesParser
from email.policy import default
import json
from pathlib import Path
import re
import subprocess
import zipfile

from verify_publication import sha256
from released_wheel import verify_download


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(['git', *arguments], cwd=root, text=True).strip()


def validate_identity(root: Path, record, run_id: int) -> None:
    if set(record) != {'schema_version', 'run_id', 'release', 'commit', 'wheel'} or type(record['schema_version']) is not int or record['schema_version'] != 1:
        raise ValueError('Unknown published-release identity format')
    if type(record['run_id']) is not int or record['run_id'] != run_id or run_id <= 0:
        raise ValueError('Release identity belongs to another publishing run')
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', record['release']) or not re.fullmatch('[0-9a-f]{40}', record['commit']):
        raise ValueError('Release tag or merged commit identity is invalid')
    wheel = record['wheel']
    if set(wheel) != {'filename', 'sha256'} or wheel['filename'] != f"vaquum_limen-{record['release'][1:]}-py3-none-any.whl" or not re.fullmatch('[0-9a-f]{64}', wheel['sha256']):
        raise ValueError('Published wheel identity is invalid')
    if git(root, 'rev-parse', f"refs/tags/{record['release']}^{{commit}}") != record['commit']:
        raise ValueError('Published release tag identifies another commit')
    subprocess.run(['git', 'merge-base', '--is-ancestor', record['commit'], 'origin/main'], cwd=root, check=True)


def write_identity(root: Path, release: str, run_id: int, dist: Path, output: Path):
    wheels = list(dist.glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('The published distribution requires exactly one wheel')
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(metadata_files) != 1:
            raise ValueError('The published wheel has ambiguous package metadata')
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_files[0]))
    if metadata['Name'] != 'vaquum-limen' or release != 'v' + str(metadata['Version']):
        raise ValueError('Published wheel metadata differs from its release')
    record = {'schema_version': 1, 'run_id': run_id, 'release': release, 'commit': git(root, 'rev-parse', 'HEAD'),
              'wheel': {'filename': wheel.name, 'sha256': sha256(wheel.read_bytes())}}
    validate_identity(root, record, run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + '\n')
    return record


def validate_download(record, receipt: Path) -> None:
    wheel = verify_download(receipt, record['release'][1:])
    if record['wheel'] != {'filename': wheel.name, 'sha256': sha256(wheel.read_bytes())}:
        raise ValueError('The downloaded PyPI wheel differs from the successful publishing run')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bind successful package publication to its exact run, merged tag and wheel.')
    commands = parser.add_subparsers(dest='command', required=True)
    write = commands.add_parser('write')
    write.add_argument('--release', required=True)
    write.add_argument('--run-id', type=int, required=True)
    write.add_argument('--dist', type=Path, required=True)
    write.add_argument('--output', type=Path, required=True)
    read = commands.add_parser('read')
    read.add_argument('--receipt', type=Path, required=True)
    read.add_argument('--run-id', type=int, required=True)
    read.add_argument('--github-output', type=Path, required=True)
    wheel = commands.add_parser('wheel')
    wheel.add_argument('--identity', type=Path, required=True)
    wheel.add_argument('--receipt', type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if arguments.command == 'write':
        print(json.dumps(write_identity(root, arguments.release, arguments.run_id, arguments.dist, arguments.output)))
    elif arguments.command == 'read':
        record = json.loads(arguments.receipt.read_text())
        validate_identity(root, record, arguments.run_id)
        with arguments.github_output.open('a') as stream:
            stream.write(f"release={record['release']}\ncommit={record['commit']}\nversion={record['release'][1:]}\n")
        print(json.dumps(record))
    else:
        validate_download(json.loads(arguments.identity.read_text()), arguments.receipt)
        print('The downloaded public wheel matches the successful package-publishing run.')
