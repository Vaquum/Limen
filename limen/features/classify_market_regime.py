from collections.abc import Iterator
from copy import deepcopy
from typing import cast

import numpy as np
import polars as pl

from limen.features._regime_audit import audit as _audit, admit_bar as _bar
from limen.features._regime_config import RegimeConfig
from limen.features._regime_contract import AUDIT, PRICES, PRODUCTION, RANKS, feature_hash
from limen.features._regime_ranks import rank_values
from limen.features._regime_state import Series as _Series, dump_state, restore_state

__all__ = ['MarketRegimeClassifier', 'RegimeConfig', 'classify_market_regime']







def _config(value: object) -> RegimeConfig:
    if not isinstance(value, RegimeConfig):
        raise TypeError('config must be a RegimeConfig')
    return value


def _frame(value: object, include_audit: object) -> pl.DataFrame:
    if not isinstance(include_audit, bool):
        raise TypeError('include_audit must be a bool')
    if not isinstance(value, pl.DataFrame):
        raise TypeError('data must be a Polars DataFrame')
    return value


class MarketRegimeClassifier:
    """Apply the same bounded, causal computation to batch and streaming rows."""

    def __init__(self, config: RegimeConfig) -> None:
        super().__init__()
        self._config = _config(config)
        self._config_hash = config.config_hash
        self._feature_hash = feature_hash()
        self._series: dict[str, _Series] = {}

    def snapshot(self) -> str:
        """Serialize bounded state as JSON with configuration and formula identity."""
        return dump_state(self._series, self._config_hash, self._feature_hash)

    @classmethod
    def from_snapshot(cls, config: RegimeConfig, snapshot: str) -> 'MarketRegimeClassifier':
        """Restore a checkpoint only under its original configuration and formulas."""
        result = cls(config)
        result._series = restore_state(snapshot, config, result._config_hash, result._feature_hash)
        return result

    def update(self, data: pl.DataFrame, *, include_audit: bool = False) -> pl.DataFrame:
        """Classify a chronological chunk; rejected keys/times leave state unchanged."""
        data = _frame(data, include_audit)
        required = ('series_id', 'datetime', *PRICES)
        missing = [name for name in required if name not in data.columns]
        if missing:
            raise ValueError(f'Missing regime input columns: {missing}')
        timestamp = data.schema['datetime']
        if data.schema['series_id'] != pl.String or not isinstance(timestamp, pl.Datetime) or timestamp.time_zone != 'UTC':
            raise ValueError('series_id must be String and datetime must be Datetime[UTC]')
        for name in PRICES:
            if not data.schema[name].is_numeric():
                raise ValueError(f'{name} must be a numeric column')
        rows = data.select('series_id', pl.col('datetime').dt.epoch('ns'),
                           *[pl.col(name).cast(pl.Float64) for name in PRICES])
        values = list(cast(Iterator[tuple[object, ...]], rows.iter_rows()))
        latest: dict[str, int] = {}
        for row in values:
            key, time = row[:2]
            if not isinstance(key, str) or not isinstance(time, int):
                raise ValueError('series_id and datetime must be non-null')
            previous = latest.get(key, self._series[key].last_timestamp if key in self._series else None)
            if previous is not None and time <= previous:
                raise ValueError('datetime must be unique and increasing within series_id')
            latest[key] = time
        working = {key: deepcopy(self._series[key]) if key in self._series else _Series()
                   for key in latest}
        output: list[dict[str, object]] = []
        for row in values:
            key, time = cast(str, row[0]), cast(int, row[1])
            state = working[key]
            observation, error = _bar(row[2:])
            state.bars.append(observation)
            audit, reason = _audit(list(state.bars), self._config.eps)
            ranks = state.ranks.update(rank_values(audit), valid=observation is not None)
            if error:
                reason = 'invalid_input:' + error
            elif not reason and any(value is None for value in ranks.values()):
                reason = 'insufficient_history'
            regime, count, transition, fallback = state.decision.step(
                cast(dict[str, float], audit), cast(dict[str, float], ranks),
                observation.close if observation is not None else None, self._config, reason)
            state.decision.previous_up = audit['breakout_up_atr']
            state.decision.previous_down = audit['breakout_down_atr']
            vol, expansion = ranks['rank_vol_rs_20_252'], ranks['rank_vol_expansion_252']
            level = float(np.float32(max(vol, expansion))) if vol is not None and expansion is not None else None
            record: dict[str, object] = {
                'series_id': key, 'datetime': time, 'market_regime': regime.name,
                'market_regime_code': int(regime), 'stress_level': level,
                'is_stress': None if level is None else level >= self._config.stress_threshold,
                'transition_kind': transition, 'bars_in_regime': count, 'fallback_reason': fallback,
                'regime_config_version': self._config.version, 'regime_config_hash': self._config_hash,
                'regime_feature_hash': self._feature_hash,
            }
            if include_audit:
                record.update(audit)
                record.update(ranks)
            output.append(record)
            state.last_timestamp = time
        schema = dict(PRODUCTION)
        schema['datetime'] = pl.Int64
        if include_audit:
            schema.update(dict.fromkeys((*AUDIT, *RANKS), pl.Float64))
        result = pl.DataFrame(output, schema=schema).with_columns(pl.col('datetime').cast(PRODUCTION['datetime']))
        self._series.update(working)
        return result


def classify_market_regime(data: pl.DataFrame, *, config: RegimeConfig,
                           include_audit: bool = False) -> pl.DataFrame:
    """Return the narrow regime schema, optionally with exact audit/rank evidence."""
    return MarketRegimeClassifier(config).update(data, include_audit=include_audit)
