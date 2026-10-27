from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import cast

from limen.features._regime_contract import AUDIT, FAST, SLOW

__all__: list[str] = []


@dataclass(frozen=True)
class _Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float


def _window(history: Sequence[_Bar | None], size: int) -> Sequence[_Bar] | None:
    subset = history[-size:]
    if len(subset) != size or any(bar is None for bar in subset):
        return None
    return cast(Sequence[_Bar], subset)


def _mean(values: Sequence[float]) -> float:
    anchor = values[0]
    return anchor + math.fsum((value - anchor) / len(values) for value in values)


def _log_ratio(numerator: float, denominator: float) -> float:
    return math.log(numerator) - math.log(denominator)


def _rs(bar: _Bar) -> float:
    return (_log_ratio(bar.high, bar.close) * _log_ratio(bar.high, bar.open)
            + _log_ratio(bar.low, bar.close) * _log_ratio(bar.low, bar.open))


def _tr(bar: _Bar, previous: _Bar | None) -> float:
    if previous is None:
        return bar.high - bar.low
    return max(bar.high - bar.low, abs(bar.high - previous.close), abs(bar.low - previous.close))


def _audit(history: Sequence[_Bar | None], eps: float) -> tuple[dict[str, float | None], str]:
    values: dict[str, float | None] = dict.fromkeys(AUDIT)
    current = history[-1]
    if current is None:
        return values, 'insufficient_history'
    previous = history[-2] if len(history) > 1 else None
    if previous is not None or len(history) == 1:
        values['true_range'] = _tr(current, previous)
    values['rs_var'] = _rs(current)
    if previous is not None:
        values['ret_1_log'] = _log_ratio(current.close, previous.close)
    fast, slow = _window(history, FAST), _window(history, SLOW)
    if fast is not None:
        prior = _window(history, FAST + 1)
        if len(history) == FAST or prior is not None:
            true_ranges = [_tr(bar, prior[index] if prior is not None else fast[index - 1] if index else None)
                           for index, bar in enumerate(fast)]
            atr = _mean(true_ranges)
            values['atr_20'], values['atr_pct_20'] = atr, atr / current.close
        values['vol_rs_20'] = math.sqrt(max(_mean([_rs(bar) for bar in fast]), 0.))
        values['sma_fast_20'] = _mean([bar.close for bar in fast])
    if slow is not None:
        values['sma_slow_60'] = _mean([bar.close for bar in slow])
        values['vol_rs_60'] = math.sqrt(max(_mean([_rs(bar) for bar in slow]), 0.))
        values['participation_20_60'] = _log_ratio(
            _mean([bar.close * bar.volume for bar in slow[-FAST:]]) + eps,
            _mean([bar.close * bar.volume for bar in slow]) + eps)
    atr, fast_ma, slow_ma = values['atr_20'], values['sma_fast_20'], values['sma_slow_60']
    vol_fast, vol_slow = values['vol_rs_20'], values['vol_rs_60']
    if vol_fast is not None and vol_slow is not None:
        values['vol_expansion_20_60'] = _log_ratio(vol_fast + eps, vol_slow + eps)
    if atr is not None and fast_ma is not None and slow_ma is not None:
        values['trend_score_20_60'] = (fast_ma - slow_ma) / (atr + eps)
    prior_ma = _window(history[:-FAST], FAST)
    if fast_ma is not None and prior_ma is not None:
        values['trend_slope_20'] = _log_ratio(fast_ma + eps, _mean([bar.close for bar in prior_ma]) + eps) / FAST
    prior_range = _window(history[:-1], SLOW)
    if prior_range is not None:
        high, low = max(bar.high for bar in prior_range), min(bar.low for bar in prior_range)
        values['range_high_60_prev'], values['range_low_60_prev'] = high, low
        if high - low > eps:
            values['range_position_60'] = min(max((current.close - low) / (high - low), 0.), 1.)
        if atr is not None:
            values['range_width_atr_60'] = (high - low) / (atr + eps)
            values['breakout_up_atr'] = max(current.close - high, 0.) / (atr + eps)
            values['breakout_down_atr'] = max(low - current.close, 0.) / (atr + eps)
    returns = _window(history, FAST + 1)
    if returns is not None:
        changes = [_log_ratio(bar.close, before.close) for before, bar in zip(returns[:-1], returns[1:], strict=True)]
        values['coherence_20'] = abs(math.fsum(changes)) / (math.fsum(abs(value) for value in changes) + eps)
    complete = _window(history, SLOW + 1) is not None
    reason = ''
    for name, value in values.items():
        if value is None or not math.isfinite(value):
            values[name] = None
            if not reason:
                reason = 'invalid_input:' + name if complete else 'insufficient_history'
    return values, reason
