from limen.experiment.grid_strategy import GridStrategy
from limen.experiment.random_strategy import RandomStrategy
from limen.experiment.search_strategy import SearchStrategy

STRATEGY_REGISTRY: dict[str, type[SearchStrategy]] = {
    'random': RandomStrategy,
    'grid': GridStrategy,
}
