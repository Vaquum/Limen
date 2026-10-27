from dataclasses import dataclass

from limen.features._regime_config import RegimeConfig
from limen.features._regime_contract import MAX_BARS, TRANSITIONS, Regime

__all__: list[str] = []
_TRENDS = {Regime.TREND_UP, Regime.TREND_DOWN}
_RANGES = {Regime.RANGE_LOW_VOL, Regime.RANGE_CHOP}
BREAKOUTS = {Regime.BREAKOUT_UP, Regime.BREAKOUT_DOWN}


def _base(audit: dict[str, float], ranks: dict[str, float], config: RegimeConfig) -> tuple[Regime, str]:
    score, slope = audit['trend_score_20_60'], audit['trend_slope_20']
    strong = ranks['rank_abs_trend_score_252'] >= config.trend_rank_entry and audit['coherence_20'] >= config.coherence_entry
    if strong and score > 0 and slope > 0 and ranks['rank_trend_slope_252'] >= config.trend_slope_up:
        return Regime.TREND_UP, ''
    if strong and score < 0 and slope < 0 and ranks['rank_trend_slope_252'] <= config.trend_slope_down:
        return Regime.TREND_DOWN, ''
    if ranks['rank_vol_rs_20_252'] <= config.range_vol_entry and ranks['rank_abs_trend_score_252'] <= config.range_trend_entry:
        return Regime.RANGE_LOW_VOL, ''
    if ranks['rank_chop_252'] >= config.chop_rank_entry and ranks['rank_abs_trend_score_252'] <= config.range_trend_entry:
        return Regime.RANGE_CHOP, ''
    return Regime.RANGE_CHOP, 'finite_fallback'


def _retained(regime: Regime, audit: dict[str, float], ranks: dict[str, float], config: RegimeConfig) -> bool:
    if regime in _TRENDS:
        direction = 1 if regime == Regime.TREND_UP else -1
        return (audit['trend_score_20_60'] * direction > 0 and audit['trend_slope_20'] * direction > 0
                and ranks['rank_abs_trend_score_252'] >= config.trend_rank_exit
                and audit['coherence_20'] >= config.coherence_exit)
    return (regime == Regime.RANGE_LOW_VOL and ranks['rank_vol_rs_20_252'] <= config.range_vol_exit
            and ranks['rank_abs_trend_score_252'] <= config.range_trend_exit)


def _transition(before: Regime, after: Regime) -> str:
    if after in (before, Regime.INITIALIZING):
        return 'none'
    if before == Regime.INITIALIZING:
        return 'from_initializing'
    def family(value: Regime) -> str:
        return 'trend' if value in _TRENDS else 'range' if value in _RANGES else 'breakout'
    result = family(before) + '_to_' + family(after)
    return result if result in TRANSITIONS else 'flip'


@dataclass
class Decision:
    regime: Regime = Regime.INITIALIZING
    bars: int = 0
    pending: Regime | None = None
    pending_count: int = 0
    boundary: float | None = None
    previous_up: float | None = None
    previous_down: float | None = None

    def _commit(self, regime: Regime, reason: str) -> tuple[Regime, int, str, str]:
        count = self.bars + 1 if regime == self.regime else 1
        if count > MAX_BARS:
            raise ValueError('bars_in_regime exceeds UInt32 capacity')
        transition = _transition(self.regime, regime)
        self.regime, self.bars = regime, count
        return regime, count, transition, reason

    def step(self, audit: dict[str, float], ranks: dict[str, float], close: float | None,
             config: RegimeConfig, reason: str) -> tuple[Regime, int, str, str]:
        if reason:
            self.pending, self.pending_count, self.boundary = None, 0, None
            return self._commit(Regime.INITIALIZING, reason)
        if close is None:
            raise ValueError('A ready regime row requires an observed close')
        candidate, fallback = _base(audit, ranks, config)
        if self.regime in BREAKOUTS:
            if self.boundary is None:
                raise ValueError('Breakout state has no entry boundary')
            inside = close <= self.boundary if self.regime == Regime.BREAKOUT_UP else close >= self.boundary
            if not inside and self.bars < config.breakout_ttl:
                return self._commit(self.regime, '')
            self.pending, self.pending_count, self.boundary = None, 0, None
            return self._commit(candidate, fallback)
        for direction, previous, regime, boundary in (
                ('up', self.previous_up, Regime.BREAKOUT_UP, 'range_high_60_prev'),
                ('down', self.previous_down, Regime.BREAKOUT_DOWN, 'range_low_60_prev')):
            if (previous is not None and previous < config.breakout_atr <= audit[f'breakout_{direction}_atr']
                    and ranks[f'rank_breakout_{direction}_252'] >= config.breakout_rank
                    and ranks['rank_participation_252'] >= config.participation_rank):
                self.pending, self.pending_count = None, 0
                self.boundary = audit[boundary]
                return self._commit(regime, '')
        if _retained(self.regime, audit, ranks, config):
            candidate, fallback = self.regime, ''
        if self.regime in (candidate, Regime.INITIALIZING):
            self.pending, self.pending_count = None, 0
            return self._commit(candidate, fallback)
        self.pending_count = self.pending_count + 1 if candidate == self.pending else 1
        self.pending = candidate
        if self.pending_count >= config.confirmation_rows:
            self.pending, self.pending_count = None, 0
        else:
            candidate, fallback = self.regime, ''
        return self._commit(candidate, fallback)
