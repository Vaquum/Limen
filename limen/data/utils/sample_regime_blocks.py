from typing import cast

import numpy as np
import polars as pl

from limen.data.utils.lookback_window import lookback_window

__all__ = ['sample_regime_blocks']


def sample_regime_blocks(data: pl.DataFrame, target_regime: str, min_block_size: int, *,
                         seed: int, regime_version: str, lookback_days: float | None = None) -> pl.DataFrame:
    """Choose one complete contiguous block of a declared causal market regime."""
    from limen.features.classify_market_regime import RegimeConfig, classify_market_regime
    from limen.features._regime_contract import Regime

    if type(min_block_size) is not int or min_block_size < 1:
        raise ValueError('min_block_size must be a positive integer')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    regimes = {value.name for value in Regime} - {'INITIALIZING'}
    if target_regime not in regimes:
        raise ValueError(f'target_regime must be one of {sorted(regimes)}')
    if lookback_days is not None:
        data = lookback_window(data, lookback_days)
    if data.is_empty():
        raise ValueError('No observations for regime sampling')
    if 'series_id' in data.columns and data['series_id'].n_unique() != 1:
        raise ValueError('Regime block sampling requires exactly one series')
    source = data if 'series_id' in data.columns else data.with_columns(pl.lit('input').alias('series_id'))
    states = classify_market_regime(source, config=RegimeConfig(version=regime_version))['market_regime'].to_list()
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < len(states):
        end = start + 1
        while end < len(states) and states[end] == states[start]:
            end += 1
        if states[start] == target_regime and end - start >= min_block_size:
            blocks.append((start, end - start))
        start = end
    if not blocks:
        raise ValueError('No contiguous target-regime block meets min_block_size')
    first, length = blocks[cast(int, np.random.default_rng(seed).integers(len(blocks)))]
    return data.slice(first, length)
