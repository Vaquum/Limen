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
        '.github/workflows/pr_checks_docs_site.yml',
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
        'docs-site/scripts/audit-security.mjs',
        'docs/Audit-Closeout.md',
        'docs/Developer/Packaging.md',
        'limen/py.typed',
        'limen/__main__.py',
        'limen_build_backend.py',
        'requirements/constraints.txt',
        'scripts/package_audit.py',
    ]
    missing = [path for path in required_files if not (ROOT / path).exists()]
    assert not missing

    readme = _read('README.md')
    assert 'pip install vaquum-limen' in readme
    assert 'pip install vaquum_limen' not in readme
    assert 'not investment advice' in readme
    assert 'regulatory approval' in readme
    assert 'Past performance is not predictive' in readme
    assert 'total loss of capital' in readme
    assert 'Supported runtime: Limen requires Python `>=3.10,<3.14`' in readme
    assert 'Python 3.10-3.13 on macOS and Linux' in readme
    assert 'The default install is intentionally light' in readme
    assert 'vaquum-limen[all]' in readme
    assert 'latest released Limen version' in readme
    assert 'CITATION.cff' in readme
    assert 'SUPPORT.md' in readme

    for risk_path in ('README.md', 'SUPPORT.md', 'docs/Benchmark.md', 'docs/Backtest.md'):
        risk_text = _read(risk_path)
        assert 'investment advice' in risk_text
        assert 'regulatory approval' in risk_text
        assert 'Past performance is not predictive' in risk_text
        assert 'total loss of capital' in risk_text

    metadata = _project_metadata()
    assert metadata['name'] == 'vaquum-limen'
    version = metadata['version']
    assert re.fullmatch(r'\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*)?', version)
    assert metadata['urls']['Homepage'] == 'https://docs.vaquum.fi/limen/'
    assert 'Operating System :: POSIX :: Linux' in metadata['classifiers']
    assert 'Programming Language :: Python :: 3.12' in metadata['classifiers']
    assert 'Programming Language :: Python :: 3.13' in metadata['classifiers']
    assert metadata['license'] == 'MIT'
    assert metadata['requires-python'] == '>=3.10,<3.14'
    optional_dependencies = metadata['optional-dependencies']
    for extra in ('all', 'boosting', 'data', 'dev', 'indicators', 'release', 'stats', 'tabpfn', 'test'):
        assert extra in optional_dependencies
    assert any(req.startswith('anthropic') for req in optional_dependencies['release'])

    for template in (ROOT / 'limen' / 'yaml' / 'templates').glob('*.yaml'):
        assert f'limen_version: "{version}"' in template.read_text(encoding='utf-8')

    manifest = _read('MANIFEST.in')
    for expected in (
        'recursive-include docs *.md',
        'recursive-include .github',
        'recursive-include tests *.py *.json',
        'recursive-include tests/fixtures *.csv *.zip',
        'include limen/py.typed',
        'recursive-include requirements *.txt',
        'recursive-include scripts *.py',
        'prune examples',
    ):
        assert expected in manifest

    docs_site_package = json.loads(_read('docs-site/package.json'))
    assert docs_site_package['license'] == 'MIT'
    assert docs_site_package['scripts']['security:audit'] == 'node ./scripts/audit-security.mjs'
    assert docs_site_package['overrides']['serialize-javascript'] == '7.0.5'
    assert docs_site_package['overrides']['uuid'] == '11.1.1'

    docs_site_workflow = _read('.github/workflows/pr_checks_docs_site.yml')
    assert 'npm ci' in docs_site_workflow
    assert 'npm run security:audit' in docs_site_workflow
    assert 'npm run check' in docs_site_workflow

    audit_script = _read('docs-site/scripts/audit-security.mjs')
    assert 'TRACKED_ADVISORIES = new Set([])' in audit_script
    assert 'Untracked docs-site npm vulnerabilities' in audit_script

    worker = _read('docs-site/src/worker.js')
    for expected in (
        'Content-Security-Policy',
        'Strict-Transport-Security',
        'X-Content-Type-Options',
        '/robots.txt',
        '/sitemap.xml',
    ):
        assert expected in worker

    docusaurus_config = _read('docs-site/docusaurus.config.js')
    assert "onBrokenLinks: 'throw'" in docusaurus_config
    assert "onBrokenMarkdownLinks: 'throw'" in docusaurus_config
    assert "siteConfig.onBrokenMarkdownLinks" not in docusaurus_config
    for expected in (
        "property: 'og:type'",
        "property: 'og:site_name'",
        "property: 'og:url'",
        "name: 'twitter:card'",
    ):
        assert expected in docusaurus_config

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
    assert 'unambiguous short prefix' in docs_text
    assert 'metadata.json` records the full canonical `manifest_id`' in docs_text
    assert 'Metric validation is intersection-based' in docs_text
    assert 'not an independent public benchmark suite' in docs_text
    assert 'does not publish a leaderboard' in docs_text
    assert 'walk-forward validation' in docs_text
    assert 'statistical acceptance gates' in docs_text
    assert 'formal research falsification proof' in docs_text
    assert 'not a published JSON Schema or editor schema' in docs_text
    assert 'python -m pip install -e ".[dev]"' in docs_text
    assert 'python -m build' in docs_text
    assert 'python scripts/package_audit.py' in docs_text
    assert 'ruff check .' in docs_text
    assert 'does not define a tox, nox, or Makefile contract' in docs_text
    assert 'Limen release and versioning contracts live in this repository' in docs_text
    assert 'release and versioning policy lives in this repository' in docs_text
    assert 'Limen-local [Semantic Versioning](../Semantic-Versioning.md)' in docs_text
    assert 'github.com/Vaquum/dev-docs/blob/main/src/Semantic-Versioning.md' not in docs_text
    assert 'github.com/Vaquum/dev-docs/blob/main/src/Making-Release.md' not in docs_text
    assert 'Trainer rejects malformed JSONL instead of skipping corrupt lines' in docs_text
    assert 'Sensor callers pass raw klines, not `x_test`' in docs_text
    assert 'non-finite selector metrics are excluded' in docs_text
    assert 'not a code-signing attestation' in docs_text
    assert 'The default package must import without loading LightGBM' in docs_text
    assert 'YAML validation rejects bool, zero, negative, and over-budget values' in docs_text
    assert 'does not read or mutate Python\'s module-global random state' in docs_text
    assert 'Missing control keys are treated as false' in docs_text
    assert 'without mutating the passed split list or column list' in docs_text
    assert 'resolve to fitted parameters only when that key is available' in docs_text
    assert 'Unknown `bar_type` values raise `ValueError`' in docs_text
    assert 'price-derived confusion-return and backtest metrics are skipped' in docs_text
    assert 'If columns cannot be aligned safely, the helper returns `NaN`' in docs_text
    assert 'Threshold grids are caller-owned inputs' in docs_text
    assert 'excluded from feature correlation by default' in docs_text
    assert 'rejects malformed custom strategy outputs' in docs_text
    assert 'Rows with zero or near-zero financial returns' in docs_text
    assert 'LightGBM binary wrapper rejects non-binary objectives before training' in docs_text
    assert 'target_confidence` must be finite and in `[0.0, 1.0]`' in docs_text
    assert 'Bundled live-safe templates set `include_research_only: false`' in docs_text


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
