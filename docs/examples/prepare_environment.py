import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def research_lock(root: Path) -> Path:
    paths = {'linux': 'requirements/ci/research-env.txt',
             'darwin': 'docs/examples/requirements-research-macos.txt'}
    if sys.platform not in paths:
        raise ValueError('No recorded research environment for this platform: ' + sys.platform)
    return root / paths[sys.platform]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_wheelhouse(root: Path) -> dict:
    path = root / '.docs-evidence/environment-wheels.json'
    record = json.loads(path.read_text())
    lock, build = research_lock(root), root / 'docs/examples/requirements-build.txt'
    if (record['runtime_lock'], record['runtime_sha256'], record['build_sha256']) != (str(lock.relative_to(root)), digest(lock), digest(build)):
        raise ValueError('Offline wheels belong to another dependency lock')
    if (record['python_abi'], record['platform'], record['machine']) != (sys.implementation.cache_tag, sys.platform, platform.machine()):
        raise ValueError('Offline wheels belong to another interpreter or platform')
    wheels = root / '.docs-evidence/wheelhouse'
    if record['files'] != {path.name: digest(path) for path in sorted(wheels.glob('*.whl'))} or not record['files']:
        raise ValueError('Offline wheel inventory is absent or changed')
    return record


def prepare(root: Path) -> None:
    output = root / '.docs-evidence/wheelhouse'
    output.mkdir(parents=True, exist_ok=True)
    lock, build = research_lock(root), root / 'docs/examples/requirements-build.txt'
    subprocess.run([sys.executable, '-m', 'pip', 'download', '--require-hashes', '--only-binary=:all:',
                    '--dest', str(output), '-r', str(lock), '-r', str(build)], check=True, timeout=600)
    record = {'runtime_lock': str(lock.relative_to(root)), 'runtime_sha256': digest(lock), 'build_sha256': digest(build),
              'python_abi': sys.implementation.cache_tag, 'platform': sys.platform, 'machine': platform.machine(),
              'files': {path.name: digest(path) for path in sorted(output.glob('*.whl'))}}
    (root / '.docs-evidence/environment-wheels.json').write_text(json.dumps(record, indent=2) + '\n')
    verify_wheelhouse(root)
    print(f"Prepared {len(record['files'])} hash-checked wheels for offline research proof.")


if __name__ == '__main__':
    argparse.ArgumentParser(description='Download pinned wheels before the offline example tests.').parse_args()
    prepare(Path(__file__).resolve().parents[2])
