from pathlib import Path

import pytest

from limen.yaml import parse, validate

ROOT = Path(__file__).resolve().parents[1]


def _config():
    config, errors = parse(ROOT / 'docs/examples/compiled-research/recovery.yaml')
    assert not errors
    return config


@pytest.mark.parametrize('field', ['feedback_interval', 'checkpoint_interval'])
@pytest.mark.parametrize('value', [0, -1])
def test_nonpositive_intervals_are_rejected_at_admission(field, value):
    config = _config()
    config['uel'][field] = value
    result = validate(config)
    assert not result.valid
    assert [(error.path, error.message) for error in result.errors] == [
        (f'uel.{field}', f"'uel.{field}' must be a positive integer")]
