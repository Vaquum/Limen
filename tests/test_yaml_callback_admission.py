from pathlib import Path

import pytest

from limen.yaml import parse, validate
from limen.yaml.schema import UEL_OPTIONAL

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('template', ('logreg_binary', 'rule_based'))
@pytest.mark.parametrize('value', (
    'this.path.does.not.resolve', 'limen.indicators.roc',
    {'func': 'this.path.does.not.resolve'}, {'func': 'limen.indicators.roc'},
    None, False, True, '', {}, [], ['callback'], 42, lambda log, msq: None,
))
def test_yaml_callback_presence_has_one_actionable_owner(template, value):
    config, errors = parse(ROOT / 'limen/yaml/templates' / f'{template}.yaml')
    assert not errors
    assert validate(config).valid
    config['uel']['intra_callback'] = value
    result = validate(config)
    assert not result.valid
    errors = [error for error in result.errors if error.path == 'uel.intra_callback']
    assert len(errors) == 1
    assert 'UniversalExperimentLoop(intra_callback=<callable(log, msq)>)' in errors[0].message
    assert '#manual-intervention' in errors[0].message
    assert not any(warning.path == 'uel.intra_callback' for warning in result.warnings)
    assert 'intra_callback' not in UEL_OPTIONAL
