from limen.experiment.reducer.pruning_strategy import PruningStrategy
from tests.stubs.stubs import StubPruningStrategy


def test_cannot_instantiate_abc():

    try:
        PruningStrategy()
        assert False, 'Should have raised TypeError'
    except TypeError:
        pass


def test_active_flag():

    interventions = [
        {'op': 'remove_is', 'param': 'a', 'value': 1, 'source': 'stub'},
    ]

    # Default active
    reducer = StubPruningStrategy(interventions=interventions)
    assert reducer.active is True
    assert len(reducer.analyze_and_intervene(None, None)) == 1

    # Init override
    reducer2 = StubPruningStrategy(active=False, interventions=interventions)
    assert reducer2.active is False
    assert reducer2.analyze_and_intervene(None, None) == []

    # Runtime toggle
    reducer.active = False
    assert reducer.analyze_and_intervene(None, None) == []
    reducer.active = True
    assert len(reducer.analyze_and_intervene(None, None)) == 1
