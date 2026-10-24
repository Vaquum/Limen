from pathlib import Path
import ast
import os
import json
import hashlib
import re
import subprocess

import pytest

from check_evidence import reference_bindings, region, validate_fences, validate_page_catalogue
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


def test_a_changed_package_file_cannot_reuse_its_previous_signature(tmp_path):
    original = (ROOT / 'limen/indicators/roc.py').read_text()
    source = tmp_path / 'roc.py'
    source.write_text(original)
    assert region(source, 'signature:roc') == ROC
    changed = original.replace('period: int = 10', 'period: int = 20')
    assert changed != original
    source.write_text(changed)
    assert region(source, 'signature:roc') == ROC.replace('period: int=10', 'period: int=20')
    source.write_text(original)
    assert region(source, 'signature:roc') == ROC


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
    for page, contexts in validate_page_catalogue(ROOT, catalogue).items():
        interfaces = tuple(sorted({catalogue['scenario_details'][name]['interface'] for name in contexts}))
        validate_fences(ROOT, ROOT / page, interfaces=interfaces)

    history = json.loads((ROOT / 'docs-site/reader/route-decisions.json').read_text())
    baseline = history['baseline']
    assert re.fullmatch(r'[a-f0-9]{40}', baseline)
    originals = {}
    for source, digest in history['sources'].items():
        content = subprocess.check_output(['git', 'show', baseline + ':' + source], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == digest
        originals[source] = content.decode()
    assert set(originals) == {'README.md', 'pyproject.toml', 'docs-site/scripts/assemble-docs.mjs'}
    published = set(re.findall(r'https://docs\.vaquum\.fi[^\s"<>)]*', originals['README.md'] + originals['pyproject.toml']))
    assert {item['url'] for item in history['published_links']} == published
    origin = 'https://docs.vaquum.fi'
    legacy = {origin + '/limen' + slug.rstrip('/') + '/' for slug in re.findall(r"\bslug:\s*'([^']+)'", originals['docs-site/scripts/assemble-docs.mjs'])}
    legacy.update(url.split('#')[0].rstrip('/') + '/' for url in published)
    assert {item['url'] for item in history['legacy_routes']} == legacy


def test_new_or_removed_pages_cannot_escape_the_complete_catalogue(monkeypatch):
    from copy import deepcopy
    import check_evidence
    catalogue = json.loads((ROOT / 'docs/examples/catalogue.json').read_text())
    changed = deepcopy(catalogue)
    page = changed['candidate_pages'].pop()
    del changed['page_contexts'][page]
    with pytest.raises(ValueError, match='every maintained document exactly once'):
        validate_page_catalogue(ROOT, changed)
    changed = deepcopy(catalogue)
    changed['candidate_pages'].append(changed['candidate_pages'][0])
    with pytest.raises(ValueError, match='every maintained document exactly once'):
        validate_page_catalogue(ROOT, changed)
    original = check_evidence.maintained_documents(ROOT)
    monkeypatch.setattr(check_evidence, 'maintained_documents', lambda root: [*original, root / 'docs/new-chapter.md'])
    with pytest.raises(ValueError, match='every maintained document exactly once'):
        validate_page_catalogue(ROOT, catalogue)


def test_current_results_reject_wrong_scenarios_fields_and_changed_artifacts(tmp_path):
    from check_evidence import result_bindings
    document = tmp_path / 'result.md'
    marker = '<!-- result: actual-selection interface=compiled scenario=selection fields=rounds -->'
    document.write_text(marker)
    assert result_bindings(ROOT, document)[0]['fields'] == ['rounds']
    for invalid in (marker.replace('scenario=selection', 'scenario=unknown'), marker + '\n' + marker,
                    marker.replace('fields=rounds', 'fields=rounds,,missing')):
        document.write_text(invalid)
        with pytest.raises(ValueError):
            result_bindings(ROOT, document)
    document.write_text(marker)
    artifact = ROOT / '.docs-evidence/result-mutation-test.json'
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text('{"rounds": 2}')
    record = {'interface': 'compiled', 'id': 'selection', 'output_path': str(artifact.relative_to(ROOT)),
              'output_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
    try:
        assert result_bindings(ROOT, document, [record])
        document.write_text(marker.replace('fields=rounds', 'fields=missing'))
        with pytest.raises(ValueError, match='Missing current result field'):
            result_bindings(ROOT, document, [record])
        document.write_text(marker)
        artifact.write_text('{"rounds": 3}')
        with pytest.raises(ValueError, match='identity differs'):
            result_bindings(ROOT, document, [record])
    finally:
        artifact.unlink()


def test_every_declared_export_and_public_member_has_an_exact_reference():
    from catalogue_admission import validate_admission
    contracts = json.loads((ROOT / 'docs/examples/api-contracts.json').read_text())
    validate_admission(ROOT, contracts)
    changed = dict(contracts)
    del changed['limen.Sensor.from_model']
    with pytest.raises(ValueError, match='admission differs'):
        validate_admission(ROOT, changed)
    assert 'limen.Log.experiment_parameter_importance' in contracts
    property_source = region(ROOT / 'limen/experiment/reducer/pruning_strategy.py', 'signature:PruningStrategy.active')
    assert '@property' in property_source and '@active.setter' in property_source


def test_current_result_row_projections_require_exact_artifacts_and_columns():
    from check_evidence import result_bindings
    document = ROOT / '.docs-evidence/result-projection-test.md'
    artifact = ROOT / '.docs-evidence/result-projection-test.json'
    document.parent.mkdir(exist_ok=True)
    marker = '<!-- result: projection-test interface=compiled scenario=selection fields=rows[].score -->'
    document.write_text(marker)
    try:
        for value, expected_error in (({'rows': [{'score': 1}, {'score': 2}]}, None),
                                      ({'rows': [{'score': 1}, {}]}, 'Missing current result field'),
                                      ({'rows': {'score': 1}}, 'not a row collection')):
            artifact.write_text(json.dumps(value))
            record = {'interface': 'compiled', 'id': 'selection', 'output_path': str(artifact.relative_to(ROOT)),
                      'output_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
            if expected_error is None:
                assert result_bindings(ROOT, document, [record])[0]['fields'] == ['rows[].score']
            else:
                with pytest.raises(ValueError, match=expected_error):
                    result_bindings(ROOT, document, [record])
    finally:
        document.unlink()
        artifact.unlink()


def test_every_cli_argument_option_default_and_type_matches_current_source():
    from catalogue_admission import cli_contracts, validate_cli_contracts
    validate_cli_contracts(ROOT)
    text = (ROOT / 'docs/Command-Line-Interface.md').read_text()
    for entries in cli_contracts().values():
        for entry in entries:
            assert all('`' + spelling + '`' in text for spelling in entry['spellings'])
    workers = next(item for item in cli_contracts()['limen run'] if item['name'] == 'workers')
    assert workers['default'] is None and workers['type']['min'] == 1


def test_yaml_inventory_includes_conditional_leaves_and_nested_contracts():
    from schema_contracts import inventory, validate
    validate(ROOT)
    current = inventory(ROOT)
    assert any(rule['path'] == 'sfd.manifest' and 'allocation' in rule.get('known', []) for rule in current['rules'])
    assert {rule['when'] for rule in current['rules']} >= {'always', 'sfd.manifest.type = ml', 'sfd.manifest.type = rule_based'}
    assert set(current['nested']['sfd.manifest.split_walk_forward']['fields']) == {'n_folds', 'test_bars', 'purge_bars', 'embargo_bars', 'anchored'}


def test_export_resolution_rereads_changed_source_at_the_same_path(tmp_path):
    from catalogue_admission import resolve
    directory = tmp_path / 'limen'
    directory.mkdir()
    source = directory / 'roc.py'
    original = (ROOT / 'limen/indicators/roc.py').read_text()
    changed = original.replace('period: int = 10', 'period: int = 20')
    assert changed != original
    for text, expected in ((original, 10), (changed, 20), (original, 10)):
        source.write_text(text)
        _, node = resolve(tmp_path, 'limen.roc', 'roc')
        assert isinstance(node, ast.FunctionDef)
        default = node.args.defaults[-1]
        assert isinstance(default, ast.Constant)
        assert default.value == expected
