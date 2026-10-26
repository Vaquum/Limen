import ast
import importlib
from functools import lru_cache
import json
import re
import subprocess
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
import unicodedata

from markdown_it import MarkdownIt
from urllib.parse import unquote

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / 'docs-site' / 'reader' / 'content.json'


def _read(path: str | Path) -> str:
    return (ROOT / path).read_text(encoding='utf-8')


def _public_markdown_paths() -> list[Path]:
    return [
        ROOT / 'README.md',
        *sorted((ROOT / 'docs').rglob('*.md')),
        *sorted((ROOT / 'limen').rglob('README.md')),
    ]


def _doc_blocks() -> list[dict]:
    return [dict(source=source, **page) for source, page in json.loads(CONTENT.read_text())['pages'].items()]


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
    return set(_heading_ids_for_text(path.read_text(encoding='utf-8')))


@lru_cache(maxsize=128)
def _heading_ids_for_text(text: str) -> frozenset[str]:
    ids: set[str] = set()
    seen: dict[str, int] = {}
    tokens = MarkdownIt().parse(text)
    for index, token in enumerate(tokens):
        if token.type != 'heading_open':
            continue
        heading = tokens[index + 1].content.replace('`', '')
        if token.tag == 'h1':
            heading = ''.join(child.content if child.type in ('text', 'code_inline', 'image') else ' ' if child.type in ('softbreak', 'hardbreak') else '' for child in tokens[index + 1].children or [])
        stem = ''.join(char for char in heading if unicodedata.category(char)[0] in 'LN' or char in '_- ').strip().lower().replace(' ', '-')
        duplicate = seen.get(stem, 0)
        if token.tag != 'h1':
            seen[stem] = duplicate + 1
        before = tokens[index - 1] if index and tokens[index - 1].type == 'html_block' else tokens[index - 2] if index > 1 and tokens[index - 1].type == 'paragraph_close' and tokens[index - 2].type == 'inline' else None
        explicit = re.fullmatch(r'<a id="([^"<>]+)" data-reader-heading="(?:python|compiled|plain)"></a>\s*', before.content.strip()) if before else None
        ids.add(explicit.group(1) if explicit else stem if duplicate == 0 else f'{stem}-{duplicate}')
    return frozenset(ids)


def test_documentation_corpus_is_fully_routed_into_the_site() -> None:
    blocks = _doc_blocks()
    mapped = [str(block['source']) for block in blocks]
    public = [str(path.relative_to(ROOT)) for path in _public_markdown_paths()]

    assert len(mapped) == len(set(mapped))
    assert set(mapped) == set(public)
    assert 'docs/Audit-Closeout.md' not in mapped
    assert not (ROOT / 'docs' / 'Audit-Closeout.md').exists()

    routes = [str(block['route']) for block in blocks]
    assert len(routes) == len(set(routes))
    assert all(route.startswith('/') and route.endswith('/') and '..' not in route for route in routes)
    positions = [block['bookOrder'] for block in blocks if 'bookOrder' in block]
    assert len(positions) == len(set(positions))


def test_site_assembly_preserves_authoring_and_navigation_contracts() -> None:
    pages = {block['source']: block for block in _doc_blocks()}
    assert pages['docs/Command-Line-Interface.md']['part'] == 'Reference'
    assert pages['docs/Developer/Packaging.md']['part'] == 'Developer'
    assert pages['docs/Roadmap.md']['part'] == 'Book'
    assert pages['docs/Developer/Security-Assurance-Case.md']['part'] == 'Developer'
    assert pages['limen/cli/commands/README.md']['part'] == 'Reference'
    for part in ('Book', 'Developer'):
        sources = [source for source, page in pages.items() if page['part'] == part and (part == 'Developer' or 'bookOrder' in page)]
        assert sources
        assert all('## Prerequisites' in _read(source) for source in sources)
    assert 'Edit this source' in _read('docs-site/reader/render.mjs')
    css = _read('docs-site/reader/book.css').replace(' ', '')
    assert re.search(r'\.table-scroll\{[^}]*\boverflow:(?:auto|scroll)(?:;|})', css)


def test_prelaunch_route_decisions_match_the_current_reader() -> None:
    history = json.loads(_read('docs-site/reader/route-decisions.json'))
    origin = 'https://docs.vaquum.fi'
    current = {origin + '/limen' + block['route'] for block in _doc_blocks()} | {origin + '/limen/reference/'}
    assert len(history['legacy_routes']) == len({item['url'] for item in history['legacy_routes']})
    for item in history['legacy_routes']:
        assert item['status'] == ('retained' if item['url'] in current else 'retired')
        assert item['replacement'] in current
    for item in history['published_links']:
        assert item['status'] in {'retained', 'retired'}
        assert item['replacement'].split('#')[0] in current


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
    version_pattern = r'limen_version:\s*["\']?(\d+\.\d+\.\d+)["\']?'
    for path in _public_markdown_paths():
        embedded_versions.extend(re.findall(version_pattern, path.read_text(encoding='utf-8')))
    for template in [*(ROOT / 'limen' / 'yaml' / 'templates').glob('*.yaml'),
                     *(ROOT / 'docs' / 'examples').glob('*.yaml')]:
        embedded_versions.extend(re.findall(version_pattern, template.read_text(encoding='utf-8')))
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
        'backtest',
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
        'the canonical JSON Schema draft-07 artifact for that package version',
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
    )
    assert all(contract in docs_text for contract in required_contracts)
    workflow = _read('docs/End-to-End-Workflow.md')
    for interface in ('python', 'compiled'):
        assert f'source=docs/examples/commands.json region={interface}-all scenario=selection interface={interface} role=executed' in workflow
        assert f'examples/{interface}-research/README.md' in workflow


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

    security = _read('SECURITY.md')
    assert 'credited in the release notes' in security
    assert 'provenance.intoto.jsonl' in security
    assert 'gh attestation verify' in security


def test_shared_editor_heading_identities() -> None:
    vectors = json.loads(_read('docs-site/editor/tests/heading-vectors.json'))
    for vector in vectors:
        assert _heading_ids_for_text(vector['markdown']) == frozenset(vector['ids'])
