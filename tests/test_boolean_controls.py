import pytest

from limen.experiment import MLManifest
from limen.yaml import CompiledSFD, validate
from tests.test_sklearn_input_boundaries import _config


@pytest.mark.parametrize('name', ['strict_mode', 'use_calibration', 'use_threshold'])
@pytest.mark.parametrize('value', ['false', '', 0, 1, 0.0, 1.0, None, [], {}])
def test_declared_controls_reject_non_booleans(name, value):
    config = _config('logreg', 'sigmoid')
    if name == 'strict_mode':
        config['sfd']['manifest'][name] = value
    else:
        config['sfd']['params'][name] = [True, value]
    result = validate(config)
    assert not result.valid
    assert any(name in error.path and 'must be a bool' in error.message for error in result.errors)
    if name == 'strict_mode':
        with pytest.raises(ValueError, match='strict_mode must be a bool'):
            CompiledSFD(config).manifest()
        with pytest.raises(ValueError, match='strict_mode must be a bool'):
            MLManifest(strict_mode=value)
        with pytest.raises(ValueError, match='strict_mode must be a bool'):
            MLManifest().set_strict_mode(value)
