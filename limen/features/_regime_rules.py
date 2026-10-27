from dataclasses import dataclass

from limen.features._regime_config import RegimeConfig
from limen.features._regime_contract import MAX_BARS, TRANSITIONS, _Regime

__all__: list[str] = []
_TRENDS = {_Regime.TREND_UP, _Regime.TREND_DOWN}
_RANGES = {_Regime.RANGE_LOW_VOL, _Regime.RANGE_CHOP}
_BREAKOUTS = {_Regime.BREAKOUT_UP, _Regime.BREAKOUT_DOWN}


def _base(audit: dict[str, float], ranks: dict[str, float], config: RegimeConfig) -> tuple[_Regime, str]:
    score, slope = audit['trend_score_20_60'], audit['trend_slope_20']
    strong = ranks['rank_abs_trend_score_252'] >= config.trend_rank_entry and audit['coherence_20'] >= config.coherence_entry
    if strong and score > 0 and slope > 0 and ranks['rank_trend_slope_252'] >= config.trend_slope_up:
        return _Regime.TREND_UP, ''
    if strong and score < 0 and slope < 0 and ranks['rank_trend_slope_252'] <= config.trend_slope_down:
        return _Regime.TREND_DOWN, ''
    if ranks['rank_vol_rs_20_252'] <= config.range_vol_entry and ranks['rank_abs_trend_score_252'] <= config.range_trend_entry:
        return _Regime.RANGE_LOW_VOL, ''
    if ranks['rank_chop_252'] >= config.chop_rank_entry and ranks['rank_abs_trend_score_252'] <= config.range_trend_entry:
        return _Regime.RANGE_CHOP, ''
    return _Regime.RANGE_CHOP, 'finite_fallback'


def _retained(regime: _Regime, audit: dict[str, float], ranks: dict[str, float], config: RegimeConfig) -> bool:
    if regime in _TRENDS:
        direction = 1 if regime == _Regime.TREND_UP else -1
        return (audit['trend_score_20_60'] * direction > 0 and audit['trend_slope_20'] * direction > 0
                and ranks['rank_abs_trend_score_252'] >= config.trend_rank_exit
                and audit['coherence_20'] >= config.coherence_exit)
    return (regime == _Regime.RANGE_LOW_VOL and ranks['rank_vol_rs_20_252'] <= config.range_vol_exit
            and ranks['rank_abs_trend_score_252'] <= config.range_trend_exit)


def _transition(before: _Regime, after: _Regime) -> str:
    if before == after or after == _Regime.INITIALIZING:
        return 'none'
    if before == _Regime.INITIALIZING:
        return 'from_initializing'
    def family(value: _Regime) -> str:
        return 'trend' if value in _TRENDS else 'range' if value in _RANGES else 'breakout'
    result = family(before) + '_to_' + family(after)
    return result if result in TRANSITIONS else 'flip'


@dataclass
class _Decision:
    regime: _Regime = _Regime.INITIALIZING
    bars: int = 0
    pending: _Regime | None = None
    pending_count: int = 0
    boundary: float | None = None
    previous_up: float | None = None
    previous_down: float | None = None

    def _commit(self, regime: _Regime, reason: str) -> tuple[_Regime, int, str, str]:
        count = self.bars + 1 if regime == self.regime else 1
        if count > MAX_BARS:
            raise ValueError('bars_in_regime exceeds UInt32 capacity')
        transition = _transition(self.regime, regime)
        self.regime, self.bars = regime, count
        return regime, count, transition, reason

    def step(self, audit: dict[str, float], ranks: dict[str, float], close: float,
             config: RegimeConfig, reason: str) -> tuple[_Regime, int, str, str]:
        if reason:
            self.pending, self.pending_count, self.boundary = None, 0, None
            return self._commit(_Regime.INITIALIZING, reason)
        candidate, fallback = _base(audit, ranks, config)
        if self.regime in _BREAKOUTS:
            if self.boundary is None:
                raise ValueError('Breakout state has no entry boundary')
            inside = close <= self.boundary if self.regime == _Regime.BREAKOUT_UP else close >= self.boundary
            if not inside and self.bars < config.breakout_ttl:
                return self._commit(self.regime, '')
            self.pending, self.pending_count, self.boundary = None, 0, None
            return self._commit(candidate, fallback)
        for direction, previous, regime, boundary in (
                ('up', self.previous_up, _Regime.BREAKOUT_UP, 'range_high_60_prev'),
                ('down', self.previous_down, _Regime.BREAKOUT_DOWN, 'range_low_60_prev')):
            if (previous is not None and previous < config.breakout_atr <= audit[f'breakout_{direction}_atr']
                    and ranks[f'rank_breakout_{direction}_252'] >= config.breakout_rank
                    and ranks['rank_participation_252'] >= config.participation_rank):
                self.pending, self.pending_count = None, 0
                self.boundary = audit[boundary]
                return self._commit(regime, '')
        if _retained(self.regime, audit, ranks, config):
            candidate, fallback = self.regime, ''
        if candidate == self.regime or self.regime == _Regime.INITIALIZING:
            self.pending, self.pending_count = None, 0
            return self._commit(candidate, fallback)
        self.pending_count = self.pending_count + 1 if candidate == self.pending else 1
        self.pending = candidate
        if self.pending_count >= config.confirmation_rows:
            self.pending, self.pending_count = None, 0
            return self._commit(candidate, fallback)
        return self._commit(self.regime, '')
