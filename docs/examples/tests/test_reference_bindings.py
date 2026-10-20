from pathlib import Path
import os
import json

import pytest

from check_evidence import reference_bindings, region, validate_fences
from provenance import source_paths

ROOT = Path(__file__).resolve().parents[3]
ROC = "def roc(data: pl.DataFrame, price_col: str='close', period: int=10) -> pl.DataFrame:\n    ...\n"
MARKER = '<!-- reference: checked-roc source=limen/indicators/roc.py region=signature:roc role=signature -->'


def test_public_signature_uses_current_source_and_parameters(tmp_path):
    from limen.indicators import roc
    from inspect import signature
    assert signature(roc).parameters['period'].default == 10
    assert region(ROOT / 'limen/indicators/roc.py', 'signature:roc') == ROC
    document = tmp_path / 'reference.md'
    document.write_text(MARKER + '\n```python\n' + ROC + '```\n')
    assert reference_bindings(ROOT, document)[0]['role'] == 'signature'
    validate_fences(ROOT, document)
    assert 'limen/indicators/roc.py' in source_paths(ROOT)


def test_changed_signature_or_reference_role_fails(tmp_path):
    document = tmp_path / 'reference.md'
    for changed in (ROC.replace('period: int=10', 'period: int=20'), ROC.replace('price_col', 'unknown_column')):
        document.write_text(MARKER + '\n```python\n' + changed + '```\n')
        with pytest.raises(ValueError, match='differs from current package'):
            reference_bindings(ROOT, document)
    document.write_text(MARKER.replace('role=signature', 'role=definition') + '\n```python\n' + ROC + '```\n')
    with pytest.raises(ValueError, match='differs from current package'):
        reference_bindings(ROOT, document)


def test_signature_marker_inside_unbound_code_cannot_certify_it(tmp_path):
    document = tmp_path / 'reference.md'
    document.write_text('~~~~markdown\n' + MARKER + '\n```python\n' + ROC + '```\n~~~~\n')
    with pytest.raises(ValueError, match='Unbound code fence'):
        validate_fences(ROOT, document)


def test_reference_source_and_symbol_boundaries(tmp_path):
    document = tmp_path / 'reference.md'
    document.write_text(MARKER.replace('limen/indicators/roc.py', 'pyproject.toml') + '\n```python\n' + ROC + '```\n')
    with pytest.raises(ValueError, match='escapes the package'):
        reference_bindings(ROOT, document)
    for symbol in ('_private', 'unknown_function', 'roc.unknown_member'):
        with pytest.raises(ValueError, match='Invalid reference symbol|Missing or ambiguous|require a class'):
            region(ROOT / 'limen/indicators/roc.py', 'signature:' + symbol)
    with pytest.raises(ValueError, match='module constant'):
        region(ROOT / 'limen/indicators/roc.py', 'definition:roc')
    assert 'SCALER_REGISTRY' in region(ROOT / 'limen/scalers/registry.py', 'definition:SCALER_REGISTRY')


def test_package_layout_is_the_actual_tracked_directory(tmp_path):
    source = ROOT / 'limen/calibration/README.md'
    layout = region(source, 'layout:package')
    assert layout.splitlines() == ['calibration/', '├── README.md', '├── __init__.py', '├── _objective_threshold.py', '├── pipeline.py', '├── probability.py', '└── threshold.py']
    marker = '<!-- reference: calibration-layout-test source=limen/calibration/README.md region=layout:package role=layout -->'
    document = tmp_path / 'layout.md'
    document.write_text(marker + '\n```text\n' + layout + '```\n')
    validate_fences(ROOT, document)
    assert reference_bindings(ROOT, document)[0]['role'] == 'layout'
    document.write_text(marker + '\n```text\n' + layout.replace('pipeline.py', 'missing.py') + '```\n')
    with pytest.raises(ValueError, match='differs from current package'):
        reference_bindings(ROOT, document)
    document.write_text(marker + '\n```python\n' + layout + '```\n')
    with pytest.raises(ValueError, match='requires a text fence'):
        reference_bindings(ROOT, document)
    with pytest.raises(ValueError, match='must name its package README'):
        region(ROOT / 'limen/calibration/pipeline.py', 'layout:package')


def test_page_connections_must_be_visible_links_to_each_project(tmp_path):
    document = tmp_path / 'chapter.md'
    links = {name: f'[{name}]({os.path.relpath(ROOT / f"docs/examples/{name}-research/README.md", tmp_path)})'
             for name in ('python', 'compiled')}
    document.write_text(links['python'] + '\n\n' + links['compiled'] + '\n')
    validate_fences(ROOT, document, interfaces=('python', 'compiled'))
    document.write_text(links['python'] + '\n\n<!-- ' + links['compiled'] + ' -->\n')
    with pytest.raises(ValueError, match='Missing visible canonical project connection: compiled'):
        validate_fences(ROOT, document, interfaces=('python', 'compiled'))
    public = 'https://github.com/Vaquum/Limen/blob/main/docs/examples/'
    document.write_text('[Python](' + public + 'python-research/README.md)\n\n[Compiled](' + public + 'compiled-research/README.md)\n')
    validate_fences(ROOT, document, interfaces=('python', 'compiled'))
    document.write_text(document.read_text().replace('/Vaquum/Limen/', '/another/repository/'))
    with pytest.raises(ValueError, match='Missing visible canonical project connection: python'):
        validate_fences(ROOT, document, interfaces=('python', 'compiled'))


def test_current_pages_have_exact_source_and_visible_project_connections():
    catalogue = json.loads((ROOT / 'docs/examples/catalogue.json').read_text())
    assert set(catalogue['page_contexts']) == set(catalogue['candidate_pages'])
    for page, contexts in catalogue['page_contexts'].items():
        assert contexts and len(contexts) == len(set(contexts))
        interfaces = tuple(sorted({catalogue['scenario_details'][name]['interface'] for name in contexts}))
        validate_fences(ROOT, ROOT / page, interfaces=interfaces)
