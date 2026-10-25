from pathlib import Path

import pytest

from limen.yaml import parse, validate

ROOT = Path(__file__).resolve().parents[1]


def _config():
    config, errors = parse(ROOT / 'docs/examples/compiled-research/target_nextbarup.yaml')
    assert not errors
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')
    config['sfd']['params'].pop('C')
    return config


@pytest.mark.parametrize('name', ['C', 'feature_groups', 'auto_pca', 'feature_drop_count'])
def test_forward_lookup_requires_a_real_declaration(name):
    config = _config()
    config['sfd']['manifest'].update(pca_compression={}, feature_ablation={})
    config['sfd']['manifest']['indicators'][0]['params']['period'] = '{' + name + '}'
    result = validate(config)
    assert not result.valid
    assert any(error.path == 'sfd.manifest.indicators[0].params.period' and
        f"'{name}' is not defined in sfd.params" in error.message for error in result.errors)
