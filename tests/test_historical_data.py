from math import isclose, sqrt

import polars as pl
import pytest

from limen.data import HistoricalData


TEST_FILE = str(HistoricalData.DEFAULT_TEST_FILE_PATH)


def test_get_any_file_loads_local_csv() -> None:
    historical = HistoricalData()

    data = historical.get_any_file(TEST_FILE, n_rows=3)

    assert isinstance(data, pl.DataFrame)
    assert data.height == 3
    assert data.columns == historical.data_columns
    assert data["datetime"].is_sorted()


def test_get_spot_klines_reaggregates_latest_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    historical = HistoricalData()
    base = historical.get_any_file(TEST_FILE, n_rows=2)
    monkeypatch.setattr(HistoricalData, "DEFAULT_SPOT_KLINES_DATASET_REPO", TEST_FILE)

    aggregated = historical.get_spot_klines(
        n_rows=1,
        kline_size=14400,
    )

    row = aggregated.row(0, named=True)
    trade_count = int(base["no_of_trades"].sum())
    weighted_mean = (
        (base["mean"] * base["no_of_trades"].cast(pl.Float64)).sum() / trade_count
    )
    sum_of_squares = (
        ((base["std"] ** 2) + (base["mean"] ** 2))
        * base["no_of_trades"].cast(pl.Float64)
    ).sum()
    weighted_variance = max((sum_of_squares / trade_count) - (weighted_mean ** 2), 0.0)

    assert row["datetime"] == base["datetime"][0]
    assert row["open"] == base["open"][0]
    assert row["high"] == base["high"].max()
    assert row["low"] == base["low"].min()
    assert row["close"] == base["close"][-1]
    assert isclose(row["mean"], round(weighted_mean, 5), rel_tol=0.0, abs_tol=1e-9)
    assert isclose(
        row["std"],
        round(sqrt(weighted_variance), 6),
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert row["volume"] == round(base["volume"].sum(), 9)
    assert isclose(
        row["maker_ratio"],
        ((base["maker_ratio"] * base["no_of_trades"].cast(pl.Float64)).sum() / trade_count),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert row["no_of_trades"] == trade_count
    assert row["open_liquidity"] == base["open_liquidity"][0]
    assert row["high_liquidity"] == base["high_liquidity"].max()
    assert row["low_liquidity"] == base["low_liquidity"].min()
    assert row["close_liquidity"] == base["close_liquidity"][-1]
    assert row["liquidity_sum"] == round(base["liquidity_sum"].sum(), 1)
    assert row["maker_volume"] == base["maker_volume"].sum()
    assert row["maker_liquidity"] == round(base["maker_liquidity"].sum(), 1)
    assert "median" not in aggregated.columns
    assert "iqr" not in aggregated.columns


def test_get_spot_klines_rejects_sub_base_intervals() -> None:
    historical = HistoricalData()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(HistoricalData, "DEFAULT_SPOT_KLINES_DATASET_REPO", TEST_FILE)
        with pytest.raises(ValueError, match="Sub-base aggregation is not supported"):
            historical.get_spot_klines(kline_size=60)
