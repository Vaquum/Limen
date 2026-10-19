import copy
import json
from pathlib import Path

import pytest

from check_evidence import bindings, validate_shape

ROOT = Path(__file__).resolve().parents[3]


def test_exact_bindings_and_coverage(research_proof):
    found = bindings(ROOT, ROOT / 'docs/Research-Book.md')
    assert {(item['interface'], item['scenario']) for item in found} == {('python', 'selection'), ('compiled', 'selection')}
    assert research_proof['python']['rounds'] == 4
    assert research_proof['compiled']['rounds'] == 2


def test_divergent_authored_source_fails(tmp_path):
    document = (ROOT / 'docs/Research-Book.md').read_text().replace('research = bars[:1440]', 'research = bars[:1500]')
    changed = tmp_path / 'changed.md'
    changed.write_text(document)
    with pytest.raises(ValueError, match='differs from executable'):
        bindings(ROOT, changed)


def test_wrong_interface_binding_fails(tmp_path):
    document = (ROOT / 'docs/Research-Book.md').read_text().replace('interface=python role=executed', 'interface=compiled role=executed')
    changed = tmp_path / 'changed.md'
    changed.write_text(document)
    with pytest.raises(ValueError, match='Wrong research interface'):
        bindings(ROOT, changed)


def test_unknown_evidence_identity_fails():
    schema = json.loads((ROOT / 'docs/examples/evidence.schema.json').read_text())
    with pytest.raises(ValueError, match='missing required'):
        validate_shape({'schema_version': 1}, schema)


def test_missing_catalogue_scenario_fails():
    from check_evidence import validate_scenarios
    catalogue = json.loads((ROOT / 'docs/examples/catalogue.json').read_text())
    assert set(catalogue['required_scenarios']) == {'selection', 'search', 'recovery', 'walk_forward', 'holdout', 'calibration', 'execution', 'bars', 'allocation', 'command_line'}
    assert catalogue['required_scenarios']['allocation'] == ['compiled']
    records = [{'interface': interface, 'id': scenario, 'status': 'passed', 'tests': ['semantic oracle'], 'assertions': ['result']} for scenario, interfaces in catalogue['required_scenarios'].items() for interface in interfaces]
    assert len(records) == 18
    validate_scenarios(records, catalogue)
    for changed in (records[:-1], records + [records[0]]):
        with pytest.raises(ValueError, match='incomplete or duplicated'):
            validate_scenarios(changed, catalogue)
    records[0]['status'] = 'skipped'
    with pytest.raises(ValueError, match='successful semantic proof'):
        validate_scenarios(records, catalogue)


def test_public_model_catalogue_requires_every_export():
    from check_evidence import validate_model_catalogue
    catalogue = json.loads((ROOT / 'docs/examples/catalogue.json').read_text())
    validate_model_catalogue(ROOT, catalogue)
    changed = copy.deepcopy(catalogue)
    del changed['model_families']['ridge_regressor']
    with pytest.raises(ValueError, match='catalogue is incomplete'):
        validate_model_catalogue(ROOT, changed)
    changed = copy.deepcopy(catalogue)
    changed['model_families']['ridge_regressor']['configuration'] = 'docs/examples/compiled-research/selection.yaml'
    with pytest.raises(ValueError, match='definition differs'):
        validate_model_catalogue(ROOT, changed)


def test_nested_yaml_version_carriers_cannot_drift():
    from check_sources import validate_versions
    from limen import __version__
    from ruamel.yaml import YAML
    document = YAML(typ='safe').load(ROOT / 'docs/examples/compiled-research/selection.yaml')
    validate_versions(document, __version__, canonical=True)
    document['metadata']['nested'] = [{'limen_version': '0.0.0'}]
    with pytest.raises(ValueError, match='Stale documentation YAML'):
        validate_versions(document, __version__, canonical=False)
    document['metadata']['nested'][0]['limen_version'] = __version__
    with pytest.raises(ValueError, match='without optional version literals'):
        validate_versions(document, __version__, canonical=True)


def test_public_cli_catalogue_requires_every_command():
    from check_evidence import validate_cli_catalogue
    catalogue = json.loads((ROOT / 'docs/examples/catalogue.json').read_text())
    validate_cli_catalogue(ROOT, catalogue)
    del catalogue['cli_commands']['backup']
    with pytest.raises(ValueError, match='CLI catalogue is incomplete'):
        validate_cli_catalogue(ROOT, catalogue)


def test_observed_outputs_and_tolerances_are_explicit(tmp_path):
    from check_evidence import compare_output, output_bindings
    assert {item['interface'] for item in output_bindings(ROOT, ROOT / 'docs/Research-Book.md')} == {'python', 'compiled'}
    document = (ROOT / 'docs/Research-Book.md').read_text().replace('"retained_models": 4', '"retained_models": 40')
    changed = tmp_path / 'changed.md';changed.write_text(document)
    with pytest.raises(ValueError, match='differs from its retained'):
        output_bindings(ROOT, changed)
    retained = json.loads((ROOT / 'docs/examples/retained/python-selection.json').read_text())
    limits = json.loads((ROOT / 'docs/examples/retained/selection.json').read_text())['displays'][0]['metrics']
    current = copy.deepcopy(retained);current['last_prediction'] += 1e-12
    compare_output(current, retained, limits)
    current['last_prediction'] += .01
    with pytest.raises(ValueError, match='declared tolerance'):
        compare_output(current, retained, limits)
    current = copy.deepcopy(retained);current['retained_models'] += 1
    with pytest.raises(ValueError, match='exact value'):
        compare_output(current, retained, limits)


def test_retained_source_is_bound_to_its_observed_revision():
    from check_evidence import validate_retained_source
    retained = json.loads((ROOT / 'docs/examples/retained/selection.json').read_text())
    for index, (source, display) in enumerate(zip(retained['sources'], retained['displays'])):
        validate_retained_source(ROOT, retained, source, display)
        changed = copy.deepcopy(retained)
        changed['sources'][index]['sha256'] = '0' * 64
        with pytest.raises(ValueError, match='immutable execution revision'):
            validate_retained_source(ROOT, changed, source, display)
        with pytest.raises(ValueError, match='Retained run identity'):
            validate_retained_source(ROOT, retained, source, display | {'retained_commit': '0' * 40})
