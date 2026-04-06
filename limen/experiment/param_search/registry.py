from __future__ import annotations

from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.experiment.param_search.random_strategy import RandomStrategy
from limen.experiment.param_search.search_strategy import SearchStrategy

STRATEGY_REGISTRY: dict[str, type[SearchStrategy]] = {
    'random': RandomStrategy,
    'grid': GridStrategy,
}
