from limen.experiment.param_domain import ParamDomain


def test_param_domain_init():

    params = {'a': [1, 2, 3], 'b': ['x', 'y']}
    domain = ParamDomain(params)

    assert domain.total_combinations == 6
    assert domain.version == 0
    assert set(domain.keys) == {'a', 'b'}
    assert domain.values_for('a') == [1, 2, 3]
    assert domain.values_for('b') == ['x', 'y']


def test_param_domain_init_validation():

    try:
        ParamDomain({'a': []})
        assert False, 'Should have raised ValueError'
    except ValueError:
        pass

    try:
        ParamDomain({'a': 'not_a_list'})
        assert False, 'Should have raised ValueError'
    except ValueError:
        pass


def test_param_domain_defensive_copy():

    original = {'a': [1, 2, 3]}
    domain = ParamDomain(original)

    original['a'].append(4)
    assert domain.values_for('a') == [1, 2, 3]

    p = domain.params
    p['a'].append(99)
    assert domain.values_for('a') == [1, 2, 3]


def test_remove_value():

    domain = ParamDomain({'a': [1, 2, 3], 'b': ['x', 'y']})

    result = domain.remove_value('a', 2)
    assert result is True
    assert domain.values_for('a') == [1, 3]
    assert domain.total_combinations == 4
    assert domain.version == 1

    result = domain.remove_value('a', 99)
    assert result is False
    assert domain.version == 1


def test_remove_value_prevents_empty():

    domain = ParamDomain({'a': [1]})

    try:
        domain.remove_value('a', 1)
        assert False, 'Should have raised ValueError'
    except ValueError:
        pass

    assert domain.values_for('a') == [1]


def test_remove_values_ge():

    domain = ParamDomain({'lr': [0.001, 0.01, 0.1, 1.0]})

    removed = domain.remove_values_ge('lr', 0.1)
    assert removed == 2
    assert domain.values_for('lr') == [0.001, 0.01]
    assert domain.version == 1


def test_remove_values_ge_prevents_empty():

    domain = ParamDomain({'lr': [0.1, 1.0]})

    try:
        domain.remove_values_ge('lr', 0.1)
        assert False, 'Should have raised ValueError'
    except ValueError:
        pass


def test_remove_values_le():

    domain = ParamDomain({'lr': [0.001, 0.01, 0.1, 1.0]})

    removed = domain.remove_values_le('lr', 0.01)
    assert removed == 2
    assert domain.values_for('lr') == [0.1, 1.0]


def test_keep_values():

    domain = ParamDomain({'opt': ['adam', 'sgd', 'rmsprop']})

    removed = domain.keep_values('opt', ['adam', 'rmsprop'])
    assert removed == 1
    assert domain.values_for('opt') == ['adam', 'rmsprop']


def test_keep_values_no_overlap():

    domain = ParamDomain({'opt': ['adam', 'sgd']})

    try:
        domain.keep_values('opt', ['ranger'])
        assert False, 'Should have raised ValueError'
    except ValueError:
        pass


def test_keep_between():

    domain = ParamDomain({'lr': [0.001, 0.01, 0.05, 0.1, 1.0]})

    removed = domain.keep_between('lr', 0.01, 0.1)
    assert removed == 2
    assert domain.values_for('lr') == [0.01, 0.05, 0.1]


def test_inject_value():

    domain = ParamDomain({'a': [1, 2]})

    result = domain.inject_value('a', 3)
    assert result is True
    assert domain.values_for('a') == [1, 2, 3]
    assert domain.total_combinations == 3
    assert domain.version == 1

    result = domain.inject_value('a', 3)
    assert result is False
    assert domain.version == 1


def test_observer_notification():

    notifications = []

    class Watcher:
        def on_domain_changed(self, domain, changed_params):
            notifications.append(changed_params)

    domain = ParamDomain({'a': [1, 2, 3], 'b': ['x', 'y']})
    watcher = Watcher()
    domain.add_observer(watcher)

    domain.remove_value('a', 2)
    assert notifications == [['a']]

    domain.inject_value('b', 'z')
    assert notifications == [['a'], ['b']]


def test_observer_version_increments():

    domain = ParamDomain({'a': [1, 2, 3]})
    assert domain.version == 0

    domain.remove_value('a', 1)
    assert domain.version == 1

    domain.inject_value('a', 4)
    assert domain.version == 2

    domain.remove_value('a', 99)
    assert domain.version == 2


def test_is_valid_combination():

    domain = ParamDomain({'a': [1, 2], 'b': ['x', 'y']})

    assert domain.is_valid_combination({'a': 1, 'b': 'x'}) is True
    assert domain.is_valid_combination({'a': 3, 'b': 'x'}) is False
    assert domain.is_valid_combination({'a': 1, 'c': 'x'}) is False

    domain.remove_value('a', 2)
    assert domain.is_valid_combination({'a': 2, 'b': 'x'}) is False


def test_comparison_type_error():

    domain = ParamDomain({'opt': ['adam', 'sgd', 'rmsprop']})

    try:
        domain.remove_values_ge('opt', 0.1)
        assert False, 'Should have raised TypeError'
    except TypeError as e:
        assert 'opt' in str(e)

    try:
        domain.remove_values_le('opt', 0.1)
        assert False, 'Should have raised TypeError'
    except TypeError as e:
        assert 'opt' in str(e)

    try:
        domain.keep_between('opt', 0.01, 0.1)
        assert False, 'Should have raised TypeError'
    except TypeError as e:
        assert 'opt' in str(e)


def test_total_combinations_updates():

    domain = ParamDomain({'a': [1, 2, 3], 'b': ['x', 'y']})
    assert domain.total_combinations == 6

    domain.remove_value('a', 3)
    assert domain.total_combinations == 4

    domain.inject_value('b', 'z')
    assert domain.total_combinations == 6

    domain.keep_values('a', [1])
    assert domain.total_combinations == 3
