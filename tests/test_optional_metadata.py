from pathlib import Path

import pytest

from limen.yaml import parse, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('declaration,path', [
    ('tags: {a: {b: 1}}', 'metadata.tags'),
    ('created_at: 2025-10-25', 'metadata.created_at'),
    ('created_at: 2025-10-25T01:02:03Z', 'metadata.created_at'),
    ('limen_version: 5.4', 'metadata.limen_version'),
])
def test_optional_metadata_shape_is_checked_after_real_yaml_parsing(declaration, path):
    source = (ROOT / 'limen/yaml/templates/logreg_binary.yaml').read_text()
    config, errors = parse(source)
    assert not errors
    config['metadata'].pop('limen_version', None)
    # Parse the declaration separately to retain ruamel's native scalar types.
    declared, errors = parse(declaration)
    assert not errors
    config['metadata'].update(declared)
    result = validate(config)
    assert not result.valid and len(result.errors) == 1
    assert result.errors[0].path == path
