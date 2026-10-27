from bisect import bisect_left, bisect_right, insort
from collections import deque
from collections.abc import Sequence

from limen.features._regime_contract import RANKS, RANK_WINDOW

__all__: list[str] = []


class Ranks:
    def __init__(self) -> None:
        super().__init__()
        self.history: list[deque[float]] = [deque() for _ in RANKS]
        self.sorted: list[list[float]] = [[] for _ in RANKS]

    def update(self, values: Sequence[float | None], *, valid: bool) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for name, current, history, ordered in zip(RANKS, values, self.history, self.sorted, strict=True):
            rank = None
            if current is not None and len(history) == RANK_WINDOW:
                rank = (bisect_left(ordered, current) + bisect_right(ordered, current)) / (2 * RANK_WINDOW)
            result[name] = rank
            if current is not None and valid:
                if len(history) == RANK_WINDOW:
                    old = history.popleft()
                    _ = ordered.pop(bisect_left(ordered, old))
                history.append(current)
                insort(ordered, current)
        return result


def rank_values(audit: dict[str, float | None]) -> list[float | None]:
    score, coherence = audit['trend_score_20_60'], audit['coherence_20']
    return [abs(score) if score is not None else None, audit['trend_slope_20'],
            audit['vol_rs_20'], audit['vol_expansion_20_60'], audit['breakout_up_atr'],
            audit['breakout_down_atr'], 1 - coherence if coherence is not None else None,
            audit['participation_20_60']]
