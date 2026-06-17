from __future__ import annotations

from email.parser import Parser
from pathlib import Path
import argparse
import re
import sys
import tarfile
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
BOUNDED_RE = re.compile(r'>=[^,;]+,<[^,;]+')
SDIST_PARTS_WITH_PREFIX = 2
BASE_FORBIDDEN = {
    'lightgbm',
    'pyarrow',
    'python-dotenv',
    'scipy',
    'statsmodels',
    'ta-lib',
    'tabpfn',
    'tslearn',
    'xgboost',
}
REQUIRED_SDIST_PATHS = {
    'CHANGELOG.md',
    'CONTRIBUTING.md',
    'SECURITY.md',
    'CITATION.cff',
    'docs/README.md',
    'docs/Developer/Packaging.md',
    'limen/py.typed',
    'limen/__main__.py',
    'limen_build_backend.py',
    'requirements/constraints.txt',
    'scripts/create_release.py',
    'scripts/package_audit.py',
    'tests/run.py',
    'tests/utils/runtime_tracking.py',
    'tests/stubs/stubs.py',
    'tests/fixtures/historical_data_spot_2h.csv',
}
REQUIRED_WHEEL_PATHS = {
    'limen/py.typed',
    'limen/__main__.py',
    'share/limen/CHANGELOG.md',
    'share/limen/CONTRIBUTING.md',
    'share/limen/SECURITY.md',
    'share/limen/CITATION.cff',
    'share/limen/docs/README.md',
    'share/limen/docs/Developer/Packaging.md',
    'share/limen/scripts/create_release.py',
    'share/limen/scripts/package_audit.py',
}


def main() -> int:
    parser = argparse.ArgumentParser(description='Audit Limen package metadata and artifacts.')
    parser.add_argument('--dist-dir', default='dist', help='Directory containing built wheel/sdist artifacts.')
    args = parser.parse_args()

    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    _audit_project(project)
    _audit_readme_links()
    _audit_manifest()
    _audit_dist(Path(args.dist_dir))
    sys.stdout.write('package audit passed\n')
    return 0


def _audit_project(project: dict) -> None:
    assert re.fullmatch(r'\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*)?', project['version'])
    assert project['license'] == 'MIT'
    assert project['requires-python'] == '>=3.10,<3.14'
    assert 'Programming Language :: Python :: 3.13' in project['classifiers']
    assert not any(item.startswith('License ::') for item in project['classifiers'])

    dependencies = project['dependencies']
    for requirement in dependencies:
        assert BOUNDED_RE.search(requirement), f'unbounded dependency: {requirement}'
        name = _requirement_name(requirement)
        assert name not in BASE_FORBIDDEN, f'heavy dependency in base install: {requirement}'

    optional = project['optional-dependencies']
    for extra in ('all', 'boosting', 'data', 'dev', 'indicators', 'release', 'stats', 'tabpfn', 'test'):
        assert extra in optional
    for extra, requirements in optional.items():
        for requirement in requirements:
            assert BOUNDED_RE.search(requirement), f'unbounded {extra} dependency: {requirement}'
    assert any(req.startswith('anthropic') for req in optional['release'])


def _audit_readme_links() -> None:
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    relative_links = re.findall(r'\]\((?!https?://|#|mailto:)([^)]+)\)', readme)
    assert not relative_links, f'README has artifact-unsafe relative links: {relative_links}'


def _audit_manifest() -> None:
    manifest = (ROOT / 'MANIFEST.in').read_text(encoding='utf-8')
    for expected in (
        'recursive-include tests *.py *.json',
        'recursive-include tests/fixtures *.csv *.zip',
        'include limen/py.typed',
        'include limen_build_backend.py',
        'recursive-include requirements *.txt',
        'recursive-include scripts *.py',
    ):
        assert expected in manifest


def _audit_dist(dist_dir: Path) -> None:
    dist_path = ROOT / dist_dir
    if not dist_path.exists():
        return

    wheels = sorted(dist_path.glob('*.whl'))
    sdists = sorted(dist_path.glob('*.tar.gz'))
    if not wheels and not sdists:
        return
    assert len(wheels) == 1, f'expected one wheel, found {wheels}'
    assert len(sdists) == 1, f'expected one sdist, found {sdists}'

    with tarfile.open(sdists[0], 'r:gz') as archive:
        sdist_names = _strip_sdist_prefix(archive.getnames())
    missing_sdist = sorted(path for path in REQUIRED_SDIST_PATHS if path not in sdist_names)
    assert not missing_sdist, f'sdist missing: {missing_sdist}'

    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_names = set(wheel.namelist())
        metadata_name = next(name for name in wheel_names if name.endswith('.dist-info/METADATA'))
        metadata = Parser().parsestr(wheel.read(metadata_name).decode('utf-8'))

    missing_wheel = sorted(path for path in REQUIRED_WHEEL_PATHS if not any(name.endswith(path) for name in wheel_names))
    assert not missing_wheel, f'wheel missing: {missing_wheel}'
    assert not any('/tests/' in name or name.startswith('tests/') for name in wheel_names)
    assert metadata['License-Expression'] == 'MIT'
    assert set(metadata['Requires-Python'].split(',')) == {'>=3.10', '<3.14'}
    assert metadata.get_all('Classifier')
    assert metadata.get_all('Keywords')


def _strip_sdist_prefix(names: list[str]) -> set[str]:
    stripped = set()
    for name in names:
        parts = name.split('/', 1)
        if len(parts) == SDIST_PARTS_WITH_PREFIX:
            stripped.add(parts[1])
    return stripped


def _requirement_name(requirement: str) -> str:
    name = re.split(r'[<>=!~;\[]', requirement, maxsplit=1)[0]
    return name.strip().lower().replace('_', '-')


if __name__ == '__main__':
    raise SystemExit(main())
