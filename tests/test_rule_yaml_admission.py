from pathlib import Path

import pytest

from limen.yaml import CompiledSFD, build_manifest, parse, validate

ROOT = Path(__file__).resolve().parents[1]


def _config():
    config, errors = parse(ROOT / 'docs/examples/compiled-research/rule_contracts.yaml')
    assert not errors
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')
    return config


@pytest.mark.parametrize('field', ['target', 'bar_formation'])
@pytest.mark.parametrize('value', [None, {}, False, 0, '', [], {'func': 'limen.does.not.exist'}])
def test_rule_yaml_rejects_discarded_fields_by_presence(field, value):
    config = _config()
    config['sfd']['manifest'][field] = value
    result = validate(config)
    path = f'sfd.manifest.{field}'
    assert not result.valid
    assert [(error.path, error.message) for error in result.errors] == [(path, f"Field '{field}' is unsupported in rule_based YAML and would be discarded")]
    assert not [warning for warning in result.warnings if warning.path == path]
    for compile_manifest in (lambda: build_manifest(config), lambda: CompiledSFD(config).manifest()):
        with pytest.raises(ValueError, match=rf'{path}:.*unsupported in rule_based YAML.*discarded'):
            compile_manifest()
