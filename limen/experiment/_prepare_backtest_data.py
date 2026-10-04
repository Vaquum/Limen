from datetime import datetime

import polars as pl

from limen.experiment._backtest_provenance import PRICE_COLUMNS as _PRICE_COLUMNS, source_splits as _source_splits


def prepare_backtest_data(
    splits: list[pl.DataFrame], all_datetimes: list[datetime] | list[int],
) -> tuple[list[pl.DataFrame], list[datetime] | list[int], pl.DataFrame | None, list[pl.DataFrame]]:
    price = splits[2].select(_PRICE_COLUMNS) if all(col in splits[2].columns for col in _PRICE_COLUMNS) else None
    sources = _source_splits(splits)
    return sources.copy(), all_datetimes, price, sources


__all__ = ['prepare_backtest_data']
