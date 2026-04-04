import random

from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.search_strategy import SearchStrategy
from limen.experiment.reducer.pruning_strategy import PruningStrategy
from limen.experiment.msq import MSQ

MAX_STUB_COMBOS = 100


class StubStrategy(SearchStrategy):

    def __init__(self, domain, *, seed=None):
        super().__init__(domain, seed=seed)
        self._combos = self._build_combos()
        self._index = 0
        self._feedback_calls = []

    @property
    def is_finite(self):
        return True

    def _build_combos(self):
        params = self._domain.params
        keys = sorted(params.keys())
        sizes = [len(params[k]) for k in keys]
        total = 1
        for s in sizes:
            total *= s

        if total <= MAX_STUB_COMBOS:
            indices = list(range(total))
        else:
            rng_seed = 0 if self._seed is None else self._seed
            rng = random.Random(rng_seed)
            indices = sorted(rng.sample(range(total), MAX_STUB_COMBOS))

        combos = []
        for idx in indices:
            combo = {}
            remaining = idx
            for i, key in enumerate(keys):
                combo[key] = params[key][remaining % sizes[i]]
                remaining //= sizes[i]
            combos.append(combo)
        return combos

    def __next__(self):
        if self._index >= len(self._combos):
            raise StopIteration
        combo = self._combos[self._index]
        self._index += 1
        self._generated_count += 1
        return combo

    def on_domain_changed(self, domain, changed_params):
        self._combos = self._build_combos()
        self._index = 0

    def update_from_feedback(self, _log, _interventions):
        self._feedback_calls.append(_interventions)

    def get_state(self):
        return {'index': self._index, 'generated_count': self._generated_count}

    def set_state(self, state):
        self._index = state['index']
        self._generated_count = state['generated_count']


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
    strategy = StubStrategy(domain)
    msq = MSQ(strategy, domain, n_permutations=n_permutations)
    return msq, strategy, domain
