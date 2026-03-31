from limen.experiment.reducer import BudgetReducer
from limen.experiment.reducer import CorrelationReducer
from limen.experiment.reducer import FocusReducer
from limen.experiment.reducer import REDUCER_REGISTRY
from limen.experiment.reducer import SanityReducer
from limen.experiment.reducer import SaturationReducer


def test_reducer_registry_has_all_types():

    assert 'budget' in REDUCER_REGISTRY
    assert 'correlation' in REDUCER_REGISTRY
    assert 'focus' in REDUCER_REGISTRY
    assert 'sanity' in REDUCER_REGISTRY
    assert 'saturation' in REDUCER_REGISTRY
    assert len(REDUCER_REGISTRY) == 5


def test_reducer_registry_maps_to_correct_classes():

    assert REDUCER_REGISTRY['budget'] is BudgetReducer
    assert REDUCER_REGISTRY['correlation'] is CorrelationReducer
    assert REDUCER_REGISTRY['focus'] is FocusReducer
    assert REDUCER_REGISTRY['sanity'] is SanityReducer
    assert REDUCER_REGISTRY['saturation'] is SaturationReducer


def test_reducer_registry_unknown_key_raises():

    try:
        REDUCER_REGISTRY['nonexistent']
        assert False, 'Expected KeyError'
    except KeyError:
        pass
