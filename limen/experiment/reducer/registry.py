from limen.experiment.reducer.budget_reducer import BudgetReducer
from limen.experiment.reducer.correlation_reducer import CorrelationReducer
from limen.experiment.reducer.focus_reducer import FocusReducer
from limen.experiment.reducer.pruning_strategy import PruningStrategy
from limen.experiment.reducer.sanity_reducer import SanityReducer
from limen.experiment.reducer.saturation_reducer import SaturationReducer

REDUCER_REGISTRY: dict[str, type[PruningStrategy]] = {
    'budget': BudgetReducer,
    'correlation': CorrelationReducer,
    'focus': FocusReducer,
    'sanity': SanityReducer,
    'saturation': SaturationReducer,
}
