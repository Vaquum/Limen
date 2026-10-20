from pathlib import Path

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
