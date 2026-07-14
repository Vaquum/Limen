import ast
import importlib
import json
import re
import subprocess
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from urllib.parse import unquote

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER = ROOT / 'docs-site' / 'scripts' / 'assemble-docs.mjs'


def _read(path: str | Path) -> str:
    return (ROOT / path).read_text(encoding='utf-8')


def _public_markdown_paths() -> list[Path]:
    return [
        ROOT / 'README.md',
        *sorted((ROOT / 'docs').rglob('*.md')),
        *sorted((ROOT / 'limen').rglob('README.md')),
    ]


def _doc_blocks() -> list[dict[str, str | int]]:
    blocks: list[dict[str, str | int]] = []
    for match in re.finditer(r"  \{\n(?P<body>.*?)\n  \},", ASSEMBLER.read_text(encoding='utf-8'), re.S):
        body = match.group('body')
        source = re.search(r"source: '([^']+)'", body)
        dest = re.search(r"dest: '([^']+)'", body)
        if source is None or dest is None:
            continue
        position = re.search(r'sidebarPosition: (\d+)', body)
        blocks.append({
            'source': source.group(1),
            'dest': dest.group(1),
            'position': int(position.group(1)) if position else -1,
        })
    return blocks


def _fences(path: Path, language: str) -> Iterator[tuple[int, str]]:
    pattern = re.compile(rf'^```{re.escape(language)}\n(.*?)^```\s*$', re.M | re.S)
    for index, match in enumerate(pattern.finditer(path.read_text(encoding='utf-8')), start=1):
        yield index, match.group(1)


def _assert_fences_parse(language: str, parser: Callable[[str], object]) -> None:
    failures: list[str] = []
    checked = 0
    for path in _public_markdown_paths():
        for index, content in _fences(path, language):
            checked += 1
            try:
                parser(content)
            except Exception as exc:
                failures.append(f'{path.relative_to(ROOT)} fence {index}: {exc}')
    assert checked > 0
    assert not failures


def _heading_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    seen: dict[str, int] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        match = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if match is None:
            continue
        heading = re.sub(r'\[([^]]+)]\([^)]+\)', r'\1', match.group(1))
        heading = heading.replace('`', '')
        slug = re.sub(r'[^\w\- ]', '', heading.lower()).strip().replace(' ', '-')
        duplicate = seen.get(slug, 0)
        seen[slug] = duplicate + 1
        ids.add(slug if duplicate == 0 else f'{slug}-{duplicate}')
    return ids


def test_documentation_corpus_is_fully_routed_into_the_site() -> None:
    blocks = _doc_blocks()
    mapped = [str(block['source']) for block in blocks]
    public = [str(path.relative_to(ROOT)) for path in _public_markdown_paths()]

    assert len(mapped) == len(set(mapped))
    assert set(mapped) == set(public)
    assert 'docs/Audit-Closeout.md' not in mapped
    assert not (ROOT / 'docs' / 'Audit-Closeout.md').exists()

    positions: dict[str, list[int]] = {}
    for block in blocks:
        position = int(block['position'])
        if position < 0:
            continue
        section = str(Path(str(block['dest'])).parent)
        positions.setdefault(section, []).append(position)
    assert all(len(values) == len(set(values)) for values in positions.values())


def test_site_assembly_preserves_authoring_and_navigation_contracts() -> None:
    script = ASSEMBLER.read_text(encoding='utf-8')
    assert "repoEditBaseUrl = 'https://github.com/Vaquum/Limen/edit/main'" in script
    assert 'custom_edit_url: ${repoEditBaseUrl}/${doc.source}' in script
    assert 'collapsed: true' in script
    assert "source: 'docs/Command-Line-Interface.md'" in script
    assert "dest: 'guides/command-line-interface.md'" in script
    assert "source: 'docs/Developer/Packaging.md'" in script
    assert "dir: 'packages/internals'" in script

    guide_sources = [
        str(block['source'])
        for block in _doc_blocks()
        if str(block['dest']).startswith('guides/')
    ]
    assert guide_sources
    assert all('## Prerequisites' in _read(source) for source in guide_sources)
    developer_sources = [
        str(block['source'])
        for block in _doc_blocks()
        if str(block['dest']).startswith('developer/')
    ]
    assert developer_sources
    assert all('## Prerequisites' in _read(source) for source in developer_sources)

    mobile_css = _read('docs-site/src/css/custom.css').split('@media (max-width: 996px)', maxsplit=1)[1]
    assert '.theme-doc-markdown table' in mobile_css
    assert 'overflow-x: auto' in mobile_css


def test_local_markdown_links_and_anchors_resolve() -> None:
    failures: list[str] = []
    link_pattern = re.compile(r'!?\[[^]]*]\(([^)]+)\)')

    for source in _public_markdown_paths():
        text = re.sub(r'^```.*?^```\s*$', '', source.read_text(encoding='utf-8'), flags=re.M | re.S)
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip().strip('<>')
            if target.startswith(('http://', 'https://', 'mailto:', '/')):
                continue
            path_text, _, fragment = target.partition('#')
            if not path_text:
                target_path = source
            else:
                path_text = unquote(path_text.split('?', 1)[0])
                target_path = (source.parent / path_text).resolve()
                if target_path.is_dir():
                    target_path /= 'README.md'
            if not target_path.exists():
                failures.append(f'{source.relative_to(ROOT)} -> {target}')
                continue
            if fragment and target_path.suffix.lower() == '.md':
                fragment = unquote(fragment).lower()
                if fragment not in _heading_ids(target_path):
                    failures.append(f'{source.relative_to(ROOT)} -> {target} (missing anchor)')

    assert not failures


def test_python_code_fences_parse() -> None:
    _assert_fences_parse('python', ast.parse)


def test_yaml_code_fences_parse() -> None:
    _assert_fences_parse('yaml', YAML(typ='safe').load)


def test_json_code_fences_parse() -> None:
    _assert_fences_parse('json', json.loads)


def test_bash_code_fences_parse() -> None:
    def parse_bash(content: str) -> object:
        subprocess.run(['bash', '-n'], input=content, text=True, check=True, capture_output=True)
        return None

    _assert_fences_parse('bash', parse_bash)


def test_examples_use_concrete_manifest_types() -> None:
    failures: list[str] = []
    for path in _public_markdown_paths():
        for index, content in _fences(path, 'python'):
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Manifest':
                    failures.append(f'{path.relative_to(ROOT)} fence {index}: bare Manifest()')
    assert not failures


def test_release_metadata_and_embedded_versions_are_consistent() -> None:
    project_version = tomllib.loads(_read('pyproject.toml'))['project']['version']
    citation = YAML(typ='safe').load(_read('CITATION.cff'))
    assert str(citation['version']) == project_version

    embedded_versions: list[str] = []
    for path in _public_markdown_paths():
        embedded_versions.extend(re.findall(r'limen_version:\s*["\']([^"\']+)["\']', path.read_text(encoding='utf-8')))
    for template in (ROOT / 'limen' / 'yaml' / 'templates').glob('*.yaml'):
        embedded_versions.extend(re.findall(r'limen_version:\s*["\']([^"\']+)["\']', template.read_text(encoding='utf-8')))
    assert embedded_versions
    assert set(embedded_versions) == {project_version}

    releases: list[tuple[tuple[int, int, int], date]] = []
    for line in _read('CHANGELOG.md').splitlines():
        if not line.startswith('## '):
            continue
        match = re.fullmatch(r'## \[(\d+)\.(\d+)\.(\d+)] - (\d{4}-\d{2}-\d{2})', line)
        assert match is not None, line
        releases.append((tuple(map(int, match.groups()[:3])), date.fromisoformat(match.group(4))))
    assert releases
    versions = [version for version, _ in releases]
    assert versions == sorted(versions)
    # Legacy releases were cut from parallel branches, so their dates do not
    # always increase with their versions.
    assert '.'.join(map(str, releases[-1][0])) == project_version
    assert releases[-1][1].isoformat() == str(citation['date-released'])


def test_package_docs_match_public_exports() -> None:
    from limen import experiment
    from limen import inference
    from limen import metrics
    from limen.experiment.reducer import REDUCER_REGISTRY
    from limen.sfd import foundational_sfd
    from limen.sfd import reference_architecture

    assert not hasattr(experiment, 'Trainer')
    assert not hasattr(experiment, 'Sensor')
    assert {'Trainer', 'Sensor', 'BarPrediction', 'ReconstructionError'} <= set(inference.__all__)
    assert 'rule_based_metrics' in metrics.__all__
    assert set(REDUCER_REGISTRY) == {'budget', 'correlation', 'focus', 'sanity', 'saturation'}

    experiment_docs = _read('limen/experiment/README.md')
    inference_docs = _read('limen/inference/README.md')
    metrics_docs = _read('limen/metrics/README.md')
    reducer_docs = _read('docs/Developer/Pruning-Strategies.md')
    reference_docs = _read('docs/Reference-Architecture.md')
    foundational_docs = _read('docs/Built-In-SFDs.md')

    assert all(name in experiment_docs for name in experiment.__all__)
    assert all(name in inference_docs for name in inference.__all__)
    assert all(name in metrics_docs for name in metrics.__all__)
    assert all(name in reducer_docs for name in REDUCER_REGISTRY)
    assert all(name in reference_docs for name in reference_architecture.__all__)
    assert all(name in foundational_docs for name in foundational_sfd.__all__)


def test_every_package_root_export_is_documented() -> None:
    package_names = (
        'calibration',
        'cli',
        'cohort',
        'data',
        'experiment',
        'inference',
        'log',
        'metrics',
        'scalers',
        'sfd',
        'targets',
        'transforms',
        'utils',
        'yaml',
    )
    for package_name in package_names:
        module = importlib.import_module(f'limen.{package_name}')
        package_docs = _read(f'limen/{package_name}/README.md')
        assert all(name in package_docs for name in module.__all__), package_name

    for package_name in ('features', 'indicators'):
        module = importlib.import_module(f'limen.{package_name}')
        reference_docs = _read(f'docs/{package_name.title()}.md')
        assert all(f'`{name}`' in reference_docs for name in module.__all__), package_name


def test_known_semantic_regressions_are_absent() -> None:
    docs_text = '\n'.join(path.read_text(encoding='utf-8') for path in _public_markdown_paths())
    assert 'train_sensors' not in docs_text
    assert 'https://docs.vaquum.fi/limen/developer/' not in docs_text
    assert '/reference/command-line-interface' not in docs_text
    assert 'limen_version: "4.' not in docs_text
    assert 'Single File Decoder' not in docs_text
    assert 'Command Line Interface' not in docs_text
    assert 'experiment logs, Trainer, and Sensor reconstruction' not in docs_text

    assert 'Trainer does not clone the manifest with `split_config=(1, 0, 0)`' in docs_text
    assert 'caller-owned values are unchanged' in docs_text
    assert '`get_arrow_file()`' in docs_text
    assert '`random_state`' in _read('docs/Targets.md')
    assert '`datetime` when the reconstructed price frame contains it' in _read('docs/Log.md')

    required_contracts = (
        'unambiguous short prefix',
        'metadata.json` records the full canonical `manifest_id`',
        'Metric validation is intersection-based',
        'not an independent public benchmark suite',
        'does not publish a leaderboard',
        'walk-forward validation',
        'statistical acceptance gates',
        'formal research falsification proof',
        'not a published JSON Schema or editor schema',
        'does not define a tox, nox, or Makefile contract',
        'Trainer rejects malformed JSONL instead of skipping corrupt lines',
        'Sensor callers pass raw klines, not `x_test`',
        'non-finite selector metrics are excluded',
        'not a code-signing attestation',
        'The default package must import without loading LightGBM',
        'YAML validation rejects bool, zero, negative, and over-budget values',
        "does not read or mutate Python's module-global random state",
        'Missing control keys are treated as false',
        'resolve to fitted parameters only when that key is available',
        'Unknown `bar_type` values raise `ValueError`',
        'price-derived confusion-return and backtest metrics are skipped',
        'If columns cannot be aligned safely, the helper returns `NaN`',
        'Threshold grids are caller-owned inputs',
        'excluded from feature correlation by default',
        'rejects malformed custom strategy outputs',
        'Rows with zero or near-zero financial returns',
        'rejects other objectives before training',
        'target_confidence` must be finite and in `[0.0, 1.0]`',
        'Bundled live-safe templates set `include_research_only: false`',
        'Previously called SFM',
        'Split-wide normalization',
        'leaks future information within the split',
        'This guide takes a contributor from a fresh checkout to a benchmarked classifier',
    )
    assert all(contract in docs_text for contract in required_contracts)


def test_public_risk_and_install_contracts_remain_visible() -> None:
    readme = _read('README.md')
    assert 'pip install "vaquum-limen[data]"' in readme
    assert 'Python `>=3.10,<3.14`' in readme
    assert 'vaquum-limen[all]' in readme

    for risk_path in ('README.md', 'SUPPORT.md', 'docs/Benchmark.md', 'docs/Backtest.md'):
        risk_text = _read(risk_path)
        assert 'investment advice' in risk_text
        assert 'regulatory approval' in risk_text
        assert 'Past performance is not predictive' in risk_text
        assert 'total loss of capital' in risk_text
