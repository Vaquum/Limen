from pathlib import Path

import pytest

from limen.experiment import MLManifest
from limen.yaml import build_manifest, parse, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('columns,error_type,index', [
    (None, TypeError, None), ([], ValueError, None), ('close', TypeError, None),
    ([''], ValueError, 0), ([' \t'], ValueError, 0), (['close', 'close'], ValueError, 1),
    ([True], TypeError, 0), ([['close']], TypeError, 0),
])
def test_explicit_required_columns_refuse_at_both_declaration_boundaries(columns, error_type, index):
    config, errors = parse(ROOT / 'limen/yaml/templates/common_strategies.yaml')
    assert not errors
    config['sfd']['manifest']['required_columns'] = columns
    result = validate(config)
    assert not result.valid and len(result.errors) == 1
    assert result.errors[0].path == 'sfd.manifest.required_columns' + (f'[{index}]' if index is not None else '')
    with pytest.raises(error_type, match='required_columns'):
        build_manifest(config)
    with pytest.raises(error_type, match='required_columns'):
        MLManifest().set_required_bar_columns(columns)
