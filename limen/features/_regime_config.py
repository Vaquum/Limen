from dataclasses import asdict, dataclass
import hashlib
import json
import math
import sys
from typing import cast

__all__: list[str] = []


@dataclass(frozen=True)
class RegimeConfig:
    """Versioned baseline hypotheses; thresholds require use-specific validation."""

    version: str
    trend_rank_entry: float = .65
    trend_slope_up: float = .60
    trend_slope_down: float = .40
    coherence_entry: float = .35
    trend_rank_exit: float = .50
    coherence_exit: float = .20
    range_vol_entry: float = .35
    range_trend_entry: float = .50
    range_vol_exit: float = .45
    range_trend_exit: float = .60
    chop_rank_entry: float = .65
    breakout_atr: float = .25
    breakout_rank: float = .95
    participation_rank: float = .60
    stress_threshold: float = .95
    confirmation_rows: int = 2
    breakout_ttl: int = 3
    eps: float = 1e-12

    def __post_init__(self) -> None:
        values = cast(dict[str, object], asdict(self))
        version = values.pop('version')
        if not isinstance(version, str) or not version.strip():
            raise ValueError('version must be a non-empty string')
        for name, value in values.items():
            if name in ('confirmation_rows', 'breakout_ttl'):
                if type(value) is not int or value < 1:
                    raise ValueError(f'{name} must be a positive integer')
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or abs(value) > sys.float_info.max or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite real number')
            elif name in ('eps', 'breakout_atr'):
                if value <= 0:
                    raise ValueError(f'{name} must be positive')
            elif not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if self.trend_rank_exit > self.trend_rank_entry or self.coherence_exit > self.coherence_entry:
            raise ValueError('Trend retention thresholds cannot exceed entry thresholds')
        if self.range_vol_exit < self.range_vol_entry or self.range_trend_exit < self.range_trend_entry:
            raise ValueError('Range retention thresholds cannot be below entry thresholds')
        if self.trend_slope_down >= self.trend_slope_up:
            raise ValueError('trend_slope_down must be below trend_slope_up')

    @property
    def config_hash(self) -> str:
        payload = {'config': asdict(self), 'windows': [20, 60, 252],
                   'regimes': ['INITIALIZING', 'TREND_UP', 'TREND_DOWN', 'RANGE_LOW_VOL',
                               'RANGE_CHOP', 'BREAKOUT_UP', 'BREAKOUT_DOWN']}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()
