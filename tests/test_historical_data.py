from datetime import datetime, timedelta, timezone
from io import BytesIO
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


def _dollar_frame(
    dollar_bar_size: int,
    *,
    second_encoded_datetimes: bool = False,
) -> pl.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    datetimes = [start + timedelta(hours=idx) for idx in range(4)]
    if second_encoded_datetimes:
        datetimes = [
            datetime.fromtimestamp(dt.timestamp() / 1000, tz=timezone.utc)
            for dt in datetimes
        ]

    return pl.DataFrame({
        'start_datetime': datetimes,
        'end_datetime': [dt + timedelta(minutes=30) for dt in datetimes],
        'dollar_bar_id': [0, 1, 2, 3],
        'open': [1.0, 2.0, 3.0, 4.0],
        'high': [2.0, 3.0, 4.0, 5.0],
        'low': [0.5, 1.5, 2.5, 3.5],
        'close': [1.5, 2.5, 3.5, 4.5],
        'mean': [1.25, 2.25, 3.25, 4.25],
        'std': [0.1, 0.2, 0.3, 0.4],
        'volume': [10.0, 20.0, 30.0, 40.0],
        'maker_ratio': [0.4, 0.5, 0.6, 0.7],
        'no_of_trades': [2, 4, 6, 8],
        'open_liquidity': [100.0, 200.0, 300.0, 400.0],
        'high_liquidity': [110.0, 210.0, 310.0, 410.0],
        'low_liquidity': [90.0, 190.0, 290.0, 390.0],
        'close_liquidity': [105.0, 205.0, 305.0, 405.0],
        'liquidity_sum': [float(dollar_bar_size)] * 4,
        'maker_volume': [4.0, 10.0, 18.0, 28.0],
        'maker_liquidity': [400.0, 1000.0, 1800.0, 2800.0],
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
    native_repos = {
        60: HistoricalData.DEFAULT_SPOT_KLINES_DATASET_REPO,
        900: 'vaquum/binance_btcusdt_15m_klines',
        1800: 'vaquum/binance_btcusdt_30m_klines',
        3600: 'vaquum/binance_btcusdt_1h_klines',
        7200: 'vaquum/binance_btcusdt_2h_klines',
        14400: 'vaquum/binance_btcusdt_4h_klines',
    }

    with TemporaryDirectory() as tmpdir:
        paths = {
            repo_id: Path(tmpdir) / f'{kline_size}.parquet'
            for kline_size, repo_id in native_repos.items()
        }
        for kline_size, repo_id in native_repos.items():
            _spot_frame(kline_size).write_parquet(paths[repo_id])

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

            one_minute = historical.get_spot_klines(kline_size=60)
            quarter_hour = historical.get_spot_klines(kline_size=900)
            half_hour = historical.get_spot_klines(kline_size=1800)
            one_hour = historical.get_spot_klines(kline_size=3600)
            two_hour = historical.get_spot_klines(kline_size=7200)
            four_hour = historical.get_spot_klines(kline_size=14400)
            fallback = historical.get_spot_klines(kline_size=300)

    assert resolved_repos == [
        HistoricalData.DEFAULT_SPOT_KLINES_DATASET_REPO,
        'vaquum/binance_btcusdt_15m_klines',
        'vaquum/binance_btcusdt_30m_klines',
        'vaquum/binance_btcusdt_1h_klines',
        'vaquum/binance_btcusdt_2h_klines',
        'vaquum/binance_btcusdt_4h_klines',
        HistoricalData.DEFAULT_SPOT_KLINES_DATASET_REPO,
    ]
    assert one_minute.height == 2
    assert quarter_hour.height == 2
    assert half_hour.height == 2
    assert one_hour.height == 2
    assert two_hour.height == 2
    assert four_hour.height == 2
    assert fallback.height == 1
    assert len(set(native_repos.values())) == 6


def test_get_spot_dollar_klines_uses_native_huggingface_sources() -> None:
    resolved_repos: list[str] = []
    native_repos = {
        1_000_000: 'vaquum/binance_btcusdt_1M_dollar_klines',
        15_000_000: 'vaquum/binance_btcusdt_15M_dollar_klines',
        30_000_000: 'vaquum/binance_btcusdt_30M_dollar_klines',
        60_000_000: 'vaquum/binance_btcusdt_60M_dollar_klines',
        120_000_000: 'vaquum/binance_btcusdt_120M_dollar_klines',
        240_000_000: 'vaquum/binance_btcusdt_240M_dollar_klines',
    }

    with TemporaryDirectory() as tmpdir:
        paths = {
            repo_id: Path(tmpdir) / f'{dollar_bar_size}.parquet'
            for dollar_bar_size, repo_id in native_repos.items()
        }
        for dollar_bar_size, repo_id in native_repos.items():
            _dollar_frame(dollar_bar_size).write_parquet(paths[repo_id])

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

            results = [
                historical.get_spot_dollar_klines(dollar_bar_size=dollar_bar_size)
                for dollar_bar_size in native_repos
            ]
            fallback = historical.get_spot_dollar_klines(
                dollar_bar_size=3_000_000,
            )

    assert resolved_repos == [
        *native_repos.values(),
        HistoricalData.DEFAULT_SPOT_DOLLAR_KLINES_DATASET_REPO,
    ]
    assert all(result.height == 4 for result in results)
    assert fallback.height == 2
    assert fallback['liquidity_sum'].to_list() == [3_000_000.0, 1_000_000.0]
    assert 'start_datetime' not in fallback.columns
    assert 'dollar_bar_id' not in fallback.columns
    assert len(set(native_repos.values())) == 6


def test_get_spot_dollar_klines_repairs_second_encoded_huggingface_datetimes() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / '1m-dollar.parquet'
        _dollar_frame(
            1_000_000,
            second_encoded_datetimes=True,
        ).write_parquet(path)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                HistoricalData,
                'DEFAULT_SPOT_DOLLAR_KLINES_DATASET_REPO',
                str(path),
            )
            historical = HistoricalData()
            data = historical.get_spot_dollar_klines(
                dollar_bar_size=1_000_000,
                start_date_limit='2020-01-01T02:00:00',
            )

    assert data.height == 2
    assert data['datetime'][0] == datetime(2020, 1, 1, 2, tzinfo=timezone.utc)
    assert data['datetime'].dt.year().to_list() == [2020, 2020]


def test_get_spot_dollar_klines_repairs_only_second_encoded_datetime_rows() -> None:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    starts = [start + timedelta(hours=idx) for idx in range(4)]
    ends = [dt + timedelta(minutes=30) for dt in starts]
    mixed_starts = [
        datetime.fromtimestamp(starts[0].timestamp() / 1000, tz=timezone.utc),
        starts[1],
        datetime.fromtimestamp(starts[2].timestamp() / 1000, tz=timezone.utc),
        starts[3],
    ]
    mixed_ends = [
        datetime.fromtimestamp(ends[0].timestamp() / 1000, tz=timezone.utc),
        ends[1],
        datetime.fromtimestamp(ends[2].timestamp() / 1000, tz=timezone.utc),
        ends[3],
    ]

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / '1m-dollar.parquet'
        _dollar_frame(1_000_000).with_columns([
            pl.Series('start_datetime', mixed_starts),
            pl.Series('end_datetime', mixed_ends),
        ]).write_parquet(path)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                HistoricalData,
                'DEFAULT_SPOT_DOLLAR_KLINES_DATASET_REPO',
                str(path),
            )
            historical = HistoricalData()
            data = historical.get_spot_dollar_klines(dollar_bar_size=1_000_000)

    assert data['datetime'].to_list() == starts


def test_get_spot_dollar_klines_rejects_sub_base_intervals() -> None:
    data = historical_module._normalize_spot_dollar_klines(_dollar_frame(1_000_000))

    with pytest.raises(ValueError, match='Sub-base aggregation is not supported'):
        historical_module._aggregate_spot_dollar_klines(
            data,
            500_000,
            1_000_000,
        )


def test_get_spot_dollar_klines_rejects_non_multiple_sizes() -> None:
    data = historical_module._normalize_spot_dollar_klines(_dollar_frame(1_000_000))

    with pytest.raises(ValueError, match='must be a multiple'):
        historical_module._aggregate_spot_dollar_klines(
            data,
            2_500_000,
            1_000_000,
        )


def test_get_spot_dollar_klines_resets_aggregate_groups_by_day() -> None:
    starts = [
        datetime(2020, 1, 1, 22, tzinfo=timezone.utc),
        datetime(2020, 1, 1, 23, tzinfo=timezone.utc),
        datetime(2020, 1, 2, 0, tzinfo=timezone.utc),
        datetime(2020, 1, 2, 1, tzinfo=timezone.utc),
    ]

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / '1m-dollar.parquet'
        _dollar_frame(1_000_000).with_columns([
            pl.Series('start_datetime', starts),
            pl.Series(
                'end_datetime',
                [dt + timedelta(minutes=30) for dt in starts],
            ),
        ]).write_parquet(path)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                HistoricalData,
                'DEFAULT_SPOT_DOLLAR_KLINES_DATASET_REPO',
                str(path),
            )
            historical = HistoricalData()
            data = historical.get_spot_dollar_klines(dollar_bar_size=3_000_000)

    assert data.height == 2
    assert data['datetime'].to_list() == [starts[0], starts[2]]
    assert data['liquidity_sum'].to_list() == [2_000_000.0, 2_000_000.0]


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
            'vaquum/binance_btcusdt_240M_dollar_klines'
        ) == (
            'https://example.com/vaquum/binance_btcusdt_240M_dollar_klines/'
            'latest.parquet'
        )
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


_ARROW_BASE_NS = 1_577_836_800_000_000_000  # 2020-01-01 00:00:00 UTC, nanoseconds
_ARROW_HOUR_NS = 3_600_000_000_000


def _write_arrow_bar_file(
    path: Path,
    *,
    rows: int = 1000,
    batch_rows: int | None = None,
    compression: str = 'uncompressed',
) -> None:
    ts = [_ARROW_BASE_NS + index * _ARROW_HOUR_NS for index in range(rows)]
    frame = pl.DataFrame({
        'ts': pl.Series(ts, dtype=pl.Int64),
        'open': [float(index) for index in range(rows)],
        'high': [float(index) + 1.0 for index in range(rows)],
        'low': [float(index) - 1.0 for index in range(rows)],
        'close': [float(index) + 0.5 for index in range(rows)],
        'volume': [float(index) * 2.0 for index in range(rows)],
    }).rechunk()
    frame.write_ipc(
        str(path),
        compression=compression,
        record_batch_size=batch_rows if batch_rows is not None else frame.height,
    )


def test_get_arrow_file_zero_copy_single_batch(tmp_path: Path) -> None:
    path = tmp_path / 'time_1h.arrow'
    _write_arrow_bar_file(path, rows=1000)

    historical = HistoricalData()
    data = historical.get_arrow_file(str(path))

    assert isinstance(data, pl.DataFrame)
    assert data.n_chunks() == 1
    # to_numpy(allow_copy=False) raises if a copy is needed; reaching the assert
    # proves the columns are views straight onto the memory map.
    assert not data['ts'].to_numpy(allow_copy=False).flags.writeable
    assert not data['close'].to_numpy(allow_copy=False).flags.writeable
    assert 'datetime' in data.columns
    assert isinstance(data.schema['datetime'], pl.Datetime)
    assert data.columns == historical.data_columns


def test_get_arrow_file_date_range_is_zero_copy_slice(tmp_path: Path) -> None:
    path = tmp_path / 'time_1h.arrow'
    _write_arrow_bar_file(path, rows=1000)

    historical = HistoricalData()
    windowed = historical.get_arrow_file(
        str(path),
        start_date_limit='2020-01-02',
        end_date_limit='2020-01-02',
    )

    assert windowed.height == 24  # one full day of hourly bars
    assert windowed.n_chunks() == 1
    assert not windowed['close'].to_numpy(allow_copy=False).flags.writeable
    assert windowed['datetime'][0] == datetime(2020, 1, 2, tzinfo=timezone.utc)
    assert windowed['datetime'][-1] == datetime(
        2020, 1, 2, 23, tzinfo=timezone.utc
    )


def test_get_arrow_file_row_count_limit_zero_copy(tmp_path: Path) -> None:
    path = tmp_path / 'time_1h.arrow'
    _write_arrow_bar_file(path, rows=1000)

    historical = HistoricalData()
    full = historical.get_arrow_file(str(path))
    tail = historical.get_arrow_file(str(path), row_count_limit=10)
    legacy = historical.get_arrow_file(str(path), n_rows=10)

    assert tail.height == 10
    assert tail['ts'].to_list() == full['ts'].tail(10).to_list()
    assert legacy['ts'].to_list() == full['ts'].tail(10).to_list()
    assert not tail['close'].to_numpy(allow_copy=False).flags.writeable


def test_get_arrow_file_rejects_non_single_batch(tmp_path: Path) -> None:
    path = tmp_path / 'multi.arrow'
    _write_arrow_bar_file(path, rows=1000, batch_rows=100)

    historical = HistoricalData()
    with pytest.raises(ValueError, match='single Arrow record batch'):
        historical.get_arrow_file(str(path))


def test_get_arrow_file_rejects_compressed(tmp_path: Path) -> None:
    # A compressed single-batch file cannot be memory-mapped, so it must raise
    # rather than silently decompressing into RAM and dropping the zero-copy
    # guarantee (it passes a naive n_chunks() == 1 check).
    path = tmp_path / 'compressed.arrow'
    _write_arrow_bar_file(path, rows=1000, compression='lz4')

    historical = HistoricalData()
    with pytest.raises(ValueError, match='compressed'):
        historical.get_arrow_file(str(path))


def test_get_arrow_file_view_outlives_call(tmp_path: Path) -> None:
    import gc

    path = tmp_path / 'time_1h.arrow'
    _write_arrow_bar_file(path, rows=1000)

    historical = HistoricalData()
    data = historical.get_arrow_file(str(path))
    ts = data['ts'].to_numpy(allow_copy=False)
    # the intermediate map/reader scope has returned; the frame (held on the
    # instance via _store) owns the mapping, so the view is still valid.
    gc.collect()
    assert int(ts[0]) == _ARROW_BASE_NS
    assert int(ts[-1]) == _ARROW_BASE_NS + 999 * _ARROW_HOUR_NS


def test_get_arrow_file_rejects_row_limit_with_closed_date_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'time_1h.arrow'
    _write_arrow_bar_file(path, rows=100)

    historical = HistoricalData()
    with pytest.raises(ValueError, match='row_count_limit must be None'):
        historical.get_arrow_file(
            str(path),
            row_count_limit=1,
            start_date_limit='2020-01-01',
            end_date_limit='2020-01-02',
        )


def test_get_arrow_file_date_range_on_datetime_only_file(tmp_path: Path) -> None:
    # A generic Arrow file with a `datetime` column and no integer `ts` index:
    # the date range falls back to a datetime filter, because the zero-copy
    # searchsorted path is guarded by an integer-dtype check on `ts`.
    path = tmp_path / 'datetime_only.arrow'
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame({
        'datetime': [start + timedelta(hours=index) for index in range(1000)],
        'close': [float(index) for index in range(1000)],
    }).rechunk()
    frame.write_ipc(str(path), compression='uncompressed', record_batch_size=frame.height)

    historical = HistoricalData()
    windowed = historical.get_arrow_file(
        str(path),
        start_date_limit='2020-01-02',
        end_date_limit='2020-01-02',
    )

    assert windowed.height == 24
    assert windowed['datetime'][0] == datetime(2020, 1, 2, tzinfo=timezone.utc)
    assert windowed['datetime'][-1] == datetime(2020, 1, 2, 23, tzinfo=timezone.utc)


def test_read_any_file_caches_huggingface_dataset_snapshots() -> None:
    buffer = BytesIO()
    _spot_frame(7200).write_parquet(buffer)
    payload = buffer.getvalue()
    network_calls: list[str] = []

    def fake_read_remote_bytes(url: str) -> bytes:
        network_calls.append(url)
        return payload

    url_a = 'https://huggingface.co/datasets/vaquum/test_repo/resolve/main/snapshot_a.parquet'
    url_b = 'https://huggingface.co/datasets/vaquum/test_repo/resolve/main/snapshot_b.parquet'

    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / 'datasets'
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(historical_module, '_DATASET_CACHE_DIR', cache_dir)
            monkeypatch.setattr(historical_module, '_read_remote_bytes', fake_read_remote_bytes)

            first = historical_module._read_any_file(url_a)
            second = historical_module._read_any_file(url_a)
            repo_dir = cache_dir / 'vaquum--test_repo'

            assert network_calls == [url_a]
            assert first.equals(second)
            assert sorted(item.name for item in repo_dir.iterdir()) == ['snapshot_a.parquet']

            historical_module._read_any_file(url_b)

            assert network_calls == [url_a, url_b]
            assert sorted(item.name for item in repo_dir.iterdir()) == ['snapshot_b.parquet']


def test_read_any_file_does_not_cache_non_huggingface_urls() -> None:
    buffer = BytesIO()
    _spot_frame(7200).write_parquet(buffer)
    payload = buffer.getvalue()
    network_calls: list[str] = []

    def fake_read_remote_bytes(url: str) -> bytes:
        network_calls.append(url)
        return payload

    url = 'https://example.com/data/snapshot.parquet'

    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / 'datasets'
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(historical_module, '_DATASET_CACHE_DIR', cache_dir)
            monkeypatch.setattr(historical_module, '_read_remote_bytes', fake_read_remote_bytes)

            historical_module._read_any_file(url)
            historical_module._read_any_file(url)

            assert network_calls == [url, url]
            assert not cache_dir.exists()
