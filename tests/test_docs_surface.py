import ast
import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding='utf-8')


def _project_metadata() -> dict:
    return tomllib.loads(_read('pyproject.toml'))['project']


def test_examples_notebook_surface_removed() -> None:
    assert not (ROOT / 'examples' / 'Train-Validate-Workflow.ipynb').exists()


def test_docs_audit_public_contract_surfaces() -> None:
    required_files = [
        '.github/FUNDING.yml',
        '.github/ISSUE_TEMPLATE/bug_report.yml',
        '.github/ISSUE_TEMPLATE/feature_request.yml',
        '.github/ISSUE_TEMPLATE/security_report.yml',
        '.github/ISSUE_TEMPLATE/support_request.yml',
        '.markdownlint.json',
        'AUTHORS',
        'CITATION.cff',
        'CODE_OF_CONDUCT.md',
        'CONTRIBUTING.md',
        'GOVERNANCE.md',
        'MAINTAINERS.md',
        'MANIFEST.in',
        'NOTICE',
        'SECURITY.md',
        'SUPPORT.md',
        'THIRD_PARTY.md',
        'docs/Audit-Closeout.md',
    ]
    missing = [path for path in required_files if not (ROOT / path).exists()]
    assert not missing

    readme = _read('README.md')
    assert 'pip install vaquum-limen' in readme
    assert 'pip install vaquum_limen' not in readme
    assert 'not investment advice' in readme
    assert 'CITATION.cff' in readme
    assert 'SUPPORT.md' in readme

    metadata = _project_metadata()
    assert metadata['name'] == 'vaquum-limen'
    assert metadata['version'] == '3.31.1'
    assert metadata['urls']['Homepage'] == 'https://docs.vaquum.fi/limen/'
    assert 'Operating System :: POSIX :: Linux' in metadata['classifiers']
    assert 'Programming Language :: Python :: 3.12' in metadata['classifiers']

    for template in (ROOT / 'limen' / 'yaml' / 'templates').glob('*.yaml'):
        assert 'limen_version: "3.31.1"' in template.read_text(encoding='utf-8')

    manifest = _read('MANIFEST.in')
    for expected in ('recursive-include docs *.md', 'recursive-include .github', 'prune examples'):
        assert expected in manifest

    docs_site_package = json.loads(_read('docs-site/package.json'))
    assert docs_site_package['license'] == 'MIT'
    assert docs_site_package['overrides']['serialize-javascript'] == '7.0.5'

    worker = _read('docs-site/src/worker.js')
    for expected in (
        'Content-Security-Policy',
        'Strict-Transport-Security',
        'X-Content-Type-Options',
        '/robots.txt',
        '/sitemap.xml',
    ):
        assert expected in worker

    closeout = _read('docs/Audit-Closeout.md')
    missing_ids = [f'D{i:03d}' for i in range(1, 95) if f'D{i:03d}' not in closeout]
    assert not missing_ids
    assert '#619' in closeout and '#620' in closeout and '#621' in closeout

    docs_text = '\n'.join(
        path.read_text(encoding='utf-8')
        for path in [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md'))]
    )
    assert 'vaquum_limen' not in docs_text
    assert 'manifest-driven SFMs' not in docs_text
    assert 'SFM has manifest' not in docs_text


def test_python_code_fences_are_parseable() -> None:
    failures: list[str] = []
    checked = 0
    paths = [
        ROOT / 'README.md',
        *sorted((ROOT / 'docs').rglob('*.md')),
        *sorted((ROOT / 'limen').rglob('README.md')),
    ]
    for path in paths:
        text = path.read_text(encoding='utf-8')
        for index, match in enumerate(re.finditer(r'```python\n(.*?)```', text, re.S), start=1):
            checked += 1
            try:
                ast.parse(match.group(1))
            except SyntaxError as exc:
                failures.append(f'{path.relative_to(ROOT)} fence {index}: {exc.msg}')

    assert checked > 0
    assert not failures
