import copy

import pytest

from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.experiment.param_search.random_strategy import RandomStrategy
from limen.yaml import build_search_strategy, validate
from tests.test_factorized_preparation import _config


@pytest.mark.parametrize('seed', (0, 42, -7))
def test_shuffled_yaml_contract(seed):
    config = _config()
    config['uel'].update(factorize=True, search_strategy={
        'type': 'grid', 'seed': seed, 'shuffle': True,
    })
    result = validate(config)
    assert result.valid, [error.message for error in result.errors]
    assert not any('shuffle' in warning.message for warning in result.warnings)
    before = copy.deepcopy(config)
    compiled = build_search_strategy(config)
    native = GridStrategy(ParamDomain(config['sfd']['params']), seed=seed, shuffle=True)
    assert type(compiled) is GridStrategy
    assert compiled.get_state() == native.get_state()
    assert list(compiled) == list(native)
    assert compiled.get_state() == native.get_state()
    assert config == before


@pytest.mark.parametrize('declaration', (
    {'type': 'grid', 'shuffle': True},
    {'type': 'grid', 'shuffle': True, 'seed': None},
    {'type': 'grid', 'shuffle': True, 'seed': True},
    {'type': 'grid', 'shuffle': True, 'seed': 1.5},
    {'type': 'grid', 'shuffle': True, 'seed': '42'},
    {'type': 'grid', 'shuffle': True, 'seed': '{fee}'},
    {'type': 'grid', 'shuffle': True, 'seed': []},
    {'type': 'grid', 'shuffle': False, 'seed': True},
    {'type': 'grid', 'shuffle': False, 'seed': '{fee}'},
    {'type': 'grid', 'shuffle': None, 'seed': 42},
    {'type': 'grid', 'shuffle': 0, 'seed': 42},
    {'type': 'grid', 'shuffle': 1, 'seed': 42},
    {'type': 'grid', 'shuffle': 'true', 'seed': 42},
    {'type': 'grid', 'shuffle': '{fee}', 'seed': 42},
    {'type': 'grid', 'shuffle': [], 'seed': 42},
    {'type': 'random', 'shuffle': False, 'seed': 42},
    {'type': 'random', 'shuffle': True, 'seed': 42},
    {'type': 'random', 'shuffle': None, 'seed': 42},
))
def test_invalid_shuffle_declarations_fail(declaration):
    config = _config()
    config['uel']['search_strategy'] = declaration
    result = validate(config)
    assert not result.valid
    assert any(error.path in ('uel.search_strategy.seed', 'uel.search_strategy.shuffle')
               for error in result.errors)
    with pytest.raises(ValueError, match='seed|shuffle'):
        build_search_strategy(config)


@pytest.mark.parametrize('declaration', (
    {'type': 'grid'},
    {'type': 'grid', 'shuffle': False},
    {'type': 'grid', 'shuffle': False, 'seed': 42},
    {'type': 'grid', 'seed': None},
    {'type': 'grid', 'seed': 42},
))
def test_unshuffled_defaults_match_native(declaration):
    config = _config()
    config['uel']['search_strategy'] = declaration
    assert validate(config).valid
    compiled = build_search_strategy(config)
    native = GridStrategy(ParamDomain(config['sfd']['params']), seed=declaration.get('seed'))
    assert compiled.get_state() == native.get_state()
    assert list(compiled) == list(native)


def test_random_search_retains_existing_compilation():
    config = _config()
    config['uel']['search_strategy'] = {'type': 'random', 'seed': 42}
    assert validate(config).valid
    compiled = build_search_strategy(config)
    native = RandomStrategy(ParamDomain(config['sfd']['params']), seed=42)
    initial_state = native.get_state()
    assert compiled.get_state() == initial_state
    assert next(compiled) == next(native)
    config['uel']['search_strategy']['seed'] = '42'
    assert not validate(config).valid
    assert build_search_strategy(config).get_state() == initial_state
