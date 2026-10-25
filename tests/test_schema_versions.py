from pathlib import Path

import pytest

from limen.yaml import parse, validate

ROOT = Path(__file__).resolve().parents[1]


def _config():
    config, errors = parse(ROOT / 'limen/yaml/templates/logreg_binary.yaml')
    assert not errors
    return config


@pytest.mark.parametrize('version', ['999.0', '0.9', '1.1', '1', '1.00', '1.0.0', ' 1.0', ''])
def test_unsupported_schema_is_refused_before_interpreting_its_body(version):
    config = _config()
    config['schema_version'] = version
    config['metadata']['mode'] = {'future': 'shape'}
    config['uel']['output_format'] = []
    result = validate(config)
    assert not result.valid and not result.warnings
    assert len(result.errors) == 1
    assert result.errors[0].path == 'schema_version'
    assert f"Unsupported schema_version '{version}'" in result.errors[0].message
    assert "supports '1.0'" in result.errors[0].message
