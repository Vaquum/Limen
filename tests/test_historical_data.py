from datetime import datetime, timedelta, timezone
from math import isclose, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import polars as pl
import pytest

import limen.data.historical_data as historical_module
from limen.data import HistoricalData


TEST_FILE = str(Path(__file__).parent / 'fixtures' / 'historical_data_spot_2h.csv')


def _spot_frame(interval_seconds: int) -> pl.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame({
        'datetime': [
            start,
            start + timedelta(seconds=interval_seconds),
        ],
        'open': [1.0, 2.0],
        'high': [2.0, 3.0],
        'low': [0.5, 1.5],
        'close': [1.5, 2.5],
        'mean': [1.25, 2.25],
        'std': [0.1, 0.2],
        'volume': [10.0, 20.0],
        'maker_ratio': [0.4, 0.6],
        'no_of_trades': [2, 4],
        'open_liquidity': [100.0, 200.0],
        'high_liquidity': [110.0, 210.0],
        'low_liquidity': [90.0, 190.0],
        'close_liquidity': [105.0, 205.0],
        'liquidity_sum': [1000.0, 2000.0],
        'maker_volume': [4.0, 12.0],
        'maker_liquidity': [400.0, 1200.0],
    })


def test_get_any_file_loads_local_csv() -> None:
    historical = HistoricalData()

    full_data = historical.get_any_file(TEST_FILE)
    data = historical.get_any_file(TEST_FILE, row_count_limit=2)
    legacy_data = historical.get_any_file(TEST_FILE, n_rows=2)

    assert isinstance(data, pl.DataFrame)
    assert data.height == 2
    assert data["datetime"].to_list() == full_data["datetime"].tail(2).to_list()
    assert legacy_data["datetime"].to_list() == full_data["datetime"].tail(2).to_list()
    assert data.columns == historical.data_columns
    assert data["datetime"].is_sorted()


def test_get_spot_klines_reaggregates_latest_dataset() -> None:
    historical = HistoricalData()
    base = historical.get_any_file(TEST_FILE).head(2)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(HistoricalData, "DEFAULT_SPOT_KLINES_DATASET_REPO", TEST_FILE)
        aggregated = historical.get_spot_klines(
            kline_size=14400,
            end_date_limit=base["datetime"][1].strftime('%Y-%m-%dT%H:%M:%S'),
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


def test_get_spot_klines_row_count_limit_returns_latest_rows() -> None:
    historical = HistoricalData()
    full_data = historical.get_any_file(TEST_FILE)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(HistoricalData, "DEFAULT_SPOT_KLINES_DATASET_REPO", TEST_FILE)
        limited = historical.get_spot_klines(
            row_count_limit=2,
            kline_size=7200,
        )
        legacy_limited = historical.get_spot_klines(
            n_rows=2,
            kline_size=7200,
        )

    assert limited["datetime"].to_list() == full_data["datetime"].tail(2).to_list()
    assert legacy_limited["datetime"].to_list() == full_data["datetime"].tail(2).to_list()


def test_get_spot_klines_uses_native_huggingface_sources() -> None:
    resolved_repos: list[str] = []

    with TemporaryDirectory() as tmpdir:
        paths = {
            HistoricalData.DEFAULT_SPOT_KLINES_DATASET_REPO: Path(tmpdir) / '1m.parquet',
            'vaquum/binance_btcusdt_15m_klines': Path(tmpdir) / '15m.parquet',
            'vaquum/binance_btcusdt_30m_klines': Path(tmpdir) / '30m.parquet',
            'vaquum/binance_btcusdt_1h_klines': Path(tmpdir) / '1h.parquet',
            'vaquum/binance_btcusdt_2h_klines': Path(tmpdir) / '2h.parquet',
            'vaquum/binance_btcusdt_4h_klines': Path(tmpdir) / '4h.parquet',
        }
        _spot_frame(60).write_parquet(paths[HistoricalData.DEFAULT_SPOT_KLINES_DATASET_REPO])
        _spot_frame(900).write_parquet(paths['vaquum/binance_btcusdt_15m_klines'])
        _spot_frame(1800).write_parquet(paths['vaquum/binance_btcusdt_30m_klines'])
        _spot_frame(3600).write_parquet(paths['vaquum/binance_btcusdt_1h_klines'])
        _spot_frame(7200).write_parquet(paths['vaquum/binance_btcusdt_2h_klines'])
        _spot_frame(14400).write_parquet(paths['vaquum/binance_btcusdt_4h_klines'])

        def fake_resolve_latest(repo_id: str) -> str:
            resolved_repos.append(repo_id)
            return str(paths[repo_id])

        historical = HistoricalData()
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                historical_module,
                '_resolve_huggingface_latest_file',
                fake_resolve_latest,
            )

            quarter_hour = historical.get_spot_klines(kline_size=900)
            half_hour = historical.get_spot_klines(kline_size=1800)
            one_hour = historical.get_spot_klines(kline_size=3600)
            two_hour = historical.get_spot_klines(kline_size=7200)
            four_hour = historical.get_spot_klines(kline_size=14400)
            fallback = historical.get_spot_klines(kline_size=300)

    assert resolved_repos == [
        'vaquum/binance_btcusdt_15m_klines',
        'vaquum/binance_btcusdt_30m_klines',
        'vaquum/binance_btcusdt_1h_klines',
        'vaquum/binance_btcusdt_2h_klines',
        'vaquum/binance_btcusdt_4h_klines',
        HistoricalData.DEFAULT_SPOT_KLINES_DATASET_REPO,
    ]
    assert quarter_hour.height == 2
    assert half_hour.height == 2
    assert one_hour.height == 2
    assert two_hour.height == 2
    assert four_hour.height == 2
    assert fallback.height == 1


def test_get_spot_klines_rejects_sub_base_intervals() -> None:
    historical = HistoricalData()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(HistoricalData, "DEFAULT_SPOT_KLINES_DATASET_REPO", TEST_FILE)
        with pytest.raises(ValueError, match="Sub-base aggregation is not supported"):
            historical.get_spot_klines(kline_size=60)


def test_resolve_file_path_or_url_expands_huggingface_references() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            historical_module,
            '_resolve_huggingface_latest_file',
            lambda repo_id: f'https://example.com/{repo_id}/latest.parquet',
        )

        assert historical_module._resolve_file_path_or_url(
            HistoricalData.DEFAULT_SPOT_KLINES_DATASET_REPO
        ) == 'https://example.com/vaquum/binance_btcusdt_1m_klines/latest.parquet'
        assert historical_module._resolve_file_path_or_url(
            'vaquum/binance_btcusdt_15m_klines'
        ) == 'https://example.com/vaquum/binance_btcusdt_15m_klines/latest.parquet'
        assert historical_module._resolve_file_path_or_url(
            'vaquum/binance_btcusdt_30m_klines'
        ) == 'https://example.com/vaquum/binance_btcusdt_30m_klines/latest.parquet'
        assert historical_module._resolve_file_path_or_url(
            'vaquum/binance_btcusdt_1h_klines'
        ) == 'https://example.com/vaquum/binance_btcusdt_1h_klines/latest.parquet'
        assert historical_module._resolve_file_path_or_url(
            'vaquum/binance_btcusdt_2h_klines'
        ) == 'https://example.com/vaquum/binance_btcusdt_2h_klines/latest.parquet'
        assert historical_module._resolve_file_path_or_url(
            'vaquum/binance_btcusdt_4h_klines'
        ) == 'https://example.com/vaquum/binance_btcusdt_4h_klines/latest.parquet'
        assert historical_module._resolve_file_path_or_url(
            'https://huggingface.co/datasets/foo/bar'
        ) == 'https://example.com/foo/bar/latest.parquet'
        assert historical_module._resolve_file_path_or_url(TEST_FILE) == TEST_FILE


def test_get_any_file_loads_zipped_csv_and_derives_datetime_from_timestamp() -> None:
    with TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'sample.csv'
        csv_path.write_text(
            'timestamp,value\n'
            '1700000000000,2\n'
            '1700000100000,3\n'
        )
        zip_path = Path(tmpdir) / 'sample.zip'

        with zipfile.ZipFile(zip_path, 'w') as archive:
            archive.write(csv_path, arcname='nested/sample.csv')

        historical = HistoricalData()
        data = historical.get_any_file(str(zip_path))

    assert data['timestamp'].dtype == pl.UInt64
    assert 'datetime' in data.columns
    assert data['datetime'].is_sorted()
    assert data['value'].to_list() == [2, 3]


def test_get_any_file_validates_requested_row_count_and_column_names() -> None:
    historical = HistoricalData()

    with pytest.raises(ValueError, match='row_count_limit must be at least 1'):
        historical.get_any_file(TEST_FILE, row_count_limit=0)

    with pytest.raises(TypeError, match='row_count_limit must be an int'):
        historical.get_any_file(TEST_FILE, row_count_limit='3')

    with pytest.raises(ValueError, match='Only one of row_count_limit and n_rows'):
        historical.get_any_file(TEST_FILE, row_count_limit=1, n_rows=1)

    with pytest.raises(ValueError, match=r'Expected .* column names'):
        historical.get_any_file(TEST_FILE, columns=['only_one'])


def test_get_any_file_rejects_unsupported_extensions() -> None:
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / 'data.txt'
        file_path.write_text('not,a,supported,format\n')
        historical = HistoricalData()

        with pytest.raises(ValueError, match='Unsupported file type'):
            historical.get_any_file(str(file_path))


def test_get_spot_klines_accepts_iso_date_limits_and_rejects_invalid_literals() -> None:
    baseline = HistoricalData()
    full_data = baseline.get_any_file(TEST_FILE)
    cutoff = full_data['datetime'][1]
    expected_start = full_data.filter(pl.col('datetime') >= cutoff)
    expected_end = full_data.filter(pl.col('datetime') <= cutoff)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(HistoricalData, "DEFAULT_SPOT_KLINES_DATASET_REPO", TEST_FILE)

        historical = HistoricalData()
        filtered = historical.get_spot_klines(
            kline_size=7200,
            start_date_limit=cutoff.strftime('%Y-%m-%dT%H:%M:%S'),
        )

        assert filtered.height == expected_start.height
        assert filtered['datetime'][0] == cutoff

        filtered = historical.get_spot_klines(
            kline_size=7200,
            end_date_limit=cutoff.strftime('%Y-%m-%dT%H:%M:%S'),
        )

        assert filtered.height == expected_end.height
        assert filtered['datetime'][-1] == cutoff

        filtered = historical.get_spot_klines(
            kline_size=7200,
            end_date_limit='2020-01-01',
        )

        assert filtered.height == full_data.height
        assert filtered['datetime'][-1] == full_data['datetime'][-1]

        with pytest.raises(ValueError, match='start_date_limit must match one of'):
            historical.get_spot_klines(
                kline_size=7200,
                start_date_limit='2025/01/01',
            )

        with pytest.raises(ValueError, match='end_date_limit must match one of'):
            historical.get_spot_klines(
                kline_size=7200,
                end_date_limit='2025/01/01',
            )


def test_get_spot_klines_rejects_row_limit_with_closed_date_window() -> None:
    historical = HistoricalData()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(HistoricalData, "DEFAULT_SPOT_KLINES_DATASET_REPO", TEST_FILE)
        with pytest.raises(ValueError, match='row_count_limit must be None'):
            historical.get_spot_klines(
                row_count_limit=1,
                kline_size=7200,
                start_date_limit='2020-01-01',
                end_date_limit='2020-01-02',
            )
