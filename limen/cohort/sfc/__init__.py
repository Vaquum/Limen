from collections.abc import Callable
from typing import Any

from limen.cohort.sfc.all import select as select_all
from limen.cohort.sfc.backtest_pareto import select as select_backtest_pareto
from limen.cohort.sfc.diverse_metrics import select as select_diverse_metrics
from limen.cohort.sfc.top_n import select as select_top_n

SelectorContext = dict[str, Any]
Selector = Callable[..., list[int | str]]

BUILTIN_SELECTORS: dict[str, Selector] = {
    'all': select_all,
    'top_n': select_top_n,
    'backtest_pareto': select_backtest_pareto,
    'diverse_metrics': select_diverse_metrics,
}

__all__ = [
    'BUILTIN_SELECTORS',
    'Selector',
    'SelectorContext',
    'select_all',
    'select_backtest_pareto',
    'select_diverse_metrics',
    'select_top_n',
]
