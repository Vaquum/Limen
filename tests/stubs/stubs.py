from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.experiment.reducer.pruning_strategy import PruningStrategy
from limen.experiment.msq import MSQ


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


def make_msq(params=None, n_permutations=None):

    if params is None:
        params = {'a': [1, 2, 3], 'b': ['x', 'y']}
    domain = ParamDomain(params)
    strategy = GridStrategy(domain)
    msq = MSQ(strategy, domain, n_permutations=n_permutations)
    return msq, strategy, domain
