import copy
import json
from pathlib import Path

import pytest

from check_evidence import bindings, region, validate_evidence, validate_shape

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
