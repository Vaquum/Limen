from enum import IntEnum
from functools import lru_cache
import hashlib
from pathlib import Path

import polars as pl

__all__: list[str] = []


class Regime(IntEnum):
    INITIALIZING = 0
    TREND_UP = 1
    TREND_DOWN = 2
    RANGE_LOW_VOL = 3
    RANGE_CHOP = 4
    BREAKOUT_UP = 5
    BREAKOUT_DOWN = 6


REGIMES = [value.name for value in Regime]
TRANSITIONS = ['none', 'from_initializing', 'trend_to_range', 'range_to_trend',
               'range_to_breakout', 'breakout_to_trend', 'breakout_to_range', 'flip']
PRICES = ('open', 'high', 'low', 'close', 'volume')
AUDIT = ('ret_1_log', 'true_range', 'atr_20', 'atr_pct_20', 'rs_var', 'vol_rs_20',
         'vol_rs_60', 'vol_expansion_20_60', 'sma_fast_20', 'sma_slow_60',
         'trend_score_20_60', 'trend_slope_20', 'range_high_60_prev', 'range_low_60_prev',
         'range_width_atr_60', 'range_position_60', 'breakout_up_atr', 'breakout_down_atr',
         'coherence_20', 'participation_20_60')
RANKS = ('rank_abs_trend_score_252', 'rank_trend_slope_252', 'rank_vol_rs_20_252',
         'rank_vol_expansion_252', 'rank_breakout_up_252', 'rank_breakout_down_252',
         'rank_chop_252', 'rank_participation_252')
PRODUCTION: dict[str, pl.DataType | type[pl.DataType]] = {
    'series_id': pl.String, 'datetime': pl.Datetime('ns', 'UTC'),
    'market_regime': pl.Enum(REGIMES), 'market_regime_code': pl.UInt8,
    'stress_level': pl.Float32, 'is_stress': pl.Boolean,
    'transition_kind': pl.Enum(TRANSITIONS), 'bars_in_regime': pl.UInt32,
    'fallback_reason': pl.String, 'regime_config_version': pl.String,
    'regime_config_hash': pl.String, 'regime_feature_hash': pl.String,
}
FAST = 20
SLOW = 60
RANK_WINDOW = 252
MAX_BARS = 2**32 - 1


@lru_cache(maxsize=1)
def feature_hash() -> str:
    sources = ('_regime_contract.py', '_regime_audit.py', '_regime_ranks.py', '_regime_rules.py',
               'classify_market_regime.py', '_regime_state.py')
    digest = hashlib.sha256()
    for name in sources:
        digest.update(name.encode() + b'\0' + Path(__file__).with_name(name).read_bytes() + b'\0')
    return digest.hexdigest()
