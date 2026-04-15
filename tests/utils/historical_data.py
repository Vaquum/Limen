from functools import lru_cache

import polars as pl

from limen.data import HistoricalData


@lru_cache(maxsize=1)
def _cached_spot_klines_2h() -> pl.DataFrame:
    historical = HistoricalData()
    return historical.get_spot_klines(kline_size=7200)


def get_cached_spot_klines_2h(n_rows: int | None = None) -> pl.DataFrame:
    data = _cached_spot_klines_2h()
    if n_rows is None:
        return data.clone()
    return data.head(n_rows).clone()
