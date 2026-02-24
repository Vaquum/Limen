from limen.experiment.pruning_strategy import PruningStrategy


class StubPruningStrategy(PruningStrategy):

    def __init__(self, *, active=True, interventions=None):

        super().__init__(active=active)

        self._interventions = interventions or []

    def analyze_and_intervene(self, _log, _msq):
        if not self._active:
            return []
        return list(self._interventions)

    def get_state(self):
        return {'active': self._active}

    def set_state(self, state):
        self._active = state['active']


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
