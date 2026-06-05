from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Final
from urllib.parse import urlparse
import zipfile

import numpy as np
import polars as pl
import requests

from limen.data._internal.binance_file_to_polars import binance_file_to_polars


_SUPPORTED_DATETIME_FORMATS: Final[tuple[str, ...]] = (
    '%Y-%m-%d',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%dT%H:%M:%S',
)
_REMOTE_TIMEOUT_SECONDS: Final[int] = 60
_DEFAULT_SPOT_DATASET_REPO: Final[str] = 'vaquum/binance_btcusdt_1m_klines'
_DEFAULT_SPOT_DOLLAR_DATASET_REPO: Final[str] = (
    'vaquum/binance_btcusdt_1M_dollar_klines'
)
_SPOT_DATASET_REPOS_BY_KLINE_SIZE: Final[dict[int, str]] = {
    60: 'vaquum/binance_btcusdt_1m_klines',
    900: 'vaquum/binance_btcusdt_15m_klines',
    1800: 'vaquum/binance_btcusdt_30m_klines',
    3600: 'vaquum/binance_btcusdt_1h_klines',
    7200: 'vaquum/binance_btcusdt_2h_klines',
    14400: 'vaquum/binance_btcusdt_4h_klines',
}
_SPOT_DOLLAR_DATASET_REPOS_BY_BAR_SIZE: Final[dict[int, str]] = {
    1_000_000: 'vaquum/binance_btcusdt_1M_dollar_klines',
    15_000_000: 'vaquum/binance_btcusdt_15M_dollar_klines',
    30_000_000: 'vaquum/binance_btcusdt_30M_dollar_klines',
    60_000_000: 'vaquum/binance_btcusdt_60M_dollar_klines',
    120_000_000: 'vaquum/binance_btcusdt_120M_dollar_klines',
    240_000_000: 'vaquum/binance_btcusdt_240M_dollar_klines',
}
_HUGGINGFACE_DATASET_REPOS: Final[frozenset[str]] = frozenset({
    _DEFAULT_SPOT_DATASET_REPO,
    _DEFAULT_SPOT_DOLLAR_DATASET_REPO,
    *_SPOT_DATASET_REPOS_BY_KLINE_SIZE.values(),
    *_SPOT_DOLLAR_DATASET_REPOS_BY_BAR_SIZE.values(),
})
_HUGGINGFACE_DATASET_REPO_PART_COUNT: Final[int] = 3
_MIN_ROWS_TO_INFER_INTERVAL: Final[int] = 2
_DATETIME_REPAIR_YEAR_CUTOFF: Final[int] = 2000


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {'http', 'https'}


def _normalize_datetime_literal(
    value: str | None,
    field_name: str,
    *,
    end_of_day: bool = False,
) -> str | None:
    if value is None:
        return None

    for fmt in _SUPPORTED_DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(value, fmt)
            if end_of_day and fmt == '%Y-%m-%d':
                parsed = parsed.replace(hour=23, minute=59, second=59)
            return parsed.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue

    raise ValueError(
        f"{field_name} must match one of: YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, "
        'YYYY-MM-DDTHH:MM:SS.'
    )


def _validate_positive_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None

    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int.")

    if value < 1:
        raise ValueError(f"{field_name} must be at least 1.")

    return value


def _resolve_row_count_limit(
    row_count_limit: int | None,
    n_rows: int | None,
) -> int | None:
    row_count_limit = _validate_positive_int(row_count_limit, 'row_count_limit')
    n_rows = _validate_positive_int(n_rows, 'n_rows')

    if row_count_limit is not None and n_rows is not None:
        raise ValueError('Only one of row_count_limit and n_rows may be set.')

    return row_count_limit if row_count_limit is not None else n_rows


def _parse_utc_datetime(value: str) -> datetime:
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)


def _slice_arrow_to_date_range(
    data: pl.DataFrame,
    start_date_limit: str | None,
    end_date_limit: str | None,
) -> pl.DataFrame:
    if start_date_limit is None and end_date_limit is None:
        return data

    if 'ts' in data.columns and data.schema['ts'].is_integer():
        # An integer `ts` is the sorted nanosecond index, so a date range is a
        # contiguous slice found by binary search -- the result stays a
        # zero-copy view rather than a materialized filter. A non-integer `ts`
        # (e.g. a Datetime column) falls through to the datetime filter below.
        index = data['ts'].to_numpy(allow_copy=False)
        start = 0
        if start_date_limit is not None:
            bound = int(_parse_utc_datetime(start_date_limit).timestamp())
            start = int(np.searchsorted(index, bound * 1_000_000_000, side='left'))
        stop = index.shape[0]
        if end_date_limit is not None:
            bound = int(_parse_utc_datetime(end_date_limit).timestamp())
            stop = int(np.searchsorted(index, bound * 1_000_000_000, side='right'))
        return data.slice(start, max(stop - start, 0))

    if 'datetime' in data.columns:
        if start_date_limit is not None:
            data = data.filter(
                pl.col('datetime') >= pl.lit(_parse_utc_datetime(start_date_limit))
            )
        if end_date_limit is not None:
            data = data.filter(
                pl.col('datetime') <= pl.lit(_parse_utc_datetime(end_date_limit))
            )
        return data

    raise ValueError(
        "A date range requires a 'ts' (Int64 nanoseconds) or 'datetime' column."
    )


def _validate_columns(df: pl.DataFrame, columns: list[str] | None) -> pl.DataFrame:
    if columns is None:
        return df

    if len(columns) != df.width:
        raise ValueError(
            f"Expected {df.width} column names, got {len(columns)}."
        )

    df.columns = columns
    return df


def _read_remote_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=_REMOTE_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def _read_csv_source(
    file_path_or_url: str,
    *,
    has_header: bool,
) -> pl.DataFrame:
    if _is_url(file_path_or_url):
        return pl.read_csv(
            BytesIO(_read_remote_bytes(file_path_or_url)),
            has_header=has_header,
            try_parse_dates=True,
        )

    return pl.read_csv(
        file_path_or_url,
        has_header=has_header,
        try_parse_dates=True,
    )


def _read_zip_source(
    file_path_or_url: str,
    *,
    has_header: bool,
) -> pl.DataFrame:
    if _is_url(file_path_or_url):
        zip_bytes = BytesIO(_read_remote_bytes(file_path_or_url))
        archive = zipfile.ZipFile(zip_bytes)
    else:
        archive = zipfile.ZipFile(file_path_or_url)

    with archive:
        csv_filename = next(
            (name for name in archive.namelist() if name.lower().endswith('.csv')),
            None,
        )
        if csv_filename is None:
            raise ValueError(f"No CSV file found inside archive: {file_path_or_url}")

        with archive.open(csv_filename) as csv_file:
            return pl.read_csv(csv_file, has_header=has_header, try_parse_dates=True)


def _resolve_huggingface_latest_file(repo_id: str) -> str:
    metadata_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/latest.json"
    response = requests.get(metadata_url, timeout=_REMOTE_TIMEOUT_SECONDS)
    response.raise_for_status()
    metadata = response.json()
    file_name = metadata['file_name']
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_name}"


def _repo_id_from_huggingface_url(file_path_or_url: str) -> str | None:
    if not file_path_or_url.startswith('https://huggingface.co/datasets/'):
        return None

    path_parts = [part for part in urlparse(file_path_or_url).path.split('/') if part]
    if (
        len(path_parts) < _HUGGINGFACE_DATASET_REPO_PART_COUNT
        or path_parts[0] != 'datasets'
    ):
        return None

    if 'resolve' in path_parts or 'blob' in path_parts:
        return None

    return '/'.join(path_parts[1:3])


def _resolve_file_path_or_url(file_path_or_url: str) -> str:
    if file_path_or_url in _HUGGINGFACE_DATASET_REPOS:
        return _resolve_huggingface_latest_file(file_path_or_url)

    repo_id = _repo_id_from_huggingface_url(file_path_or_url)
    if repo_id is not None:
        return _resolve_huggingface_latest_file(repo_id)

    return file_path_or_url


def _spot_dataset_repo_for_kline_size(kline_size: int, configured_repo: str) -> str:
    if configured_repo != _DEFAULT_SPOT_DATASET_REPO:
        return configured_repo

    return _SPOT_DATASET_REPOS_BY_KLINE_SIZE.get(kline_size, configured_repo)


def _spot_dollar_dataset_repo_for_bar_size(
    dollar_bar_size: int,
    configured_repo: str,
) -> tuple[str, int | None]:
    if configured_repo != _DEFAULT_SPOT_DOLLAR_DATASET_REPO:
        return configured_repo, None

    if dollar_bar_size in _SPOT_DOLLAR_DATASET_REPOS_BY_BAR_SIZE:
        return (
            _SPOT_DOLLAR_DATASET_REPOS_BY_BAR_SIZE[dollar_bar_size],
            dollar_bar_size,
        )

    source_sizes = [
        source_size
        for source_size in _SPOT_DOLLAR_DATASET_REPOS_BY_BAR_SIZE
        if source_size <= dollar_bar_size and dollar_bar_size % source_size == 0
    ]
    if not source_sizes:
        return configured_repo, 1_000_000

    source_size = max(source_sizes)
    return _SPOT_DOLLAR_DATASET_REPOS_BY_BAR_SIZE[source_size], source_size


def _read_any_file(
    file_path_or_url: str,
    *,
    has_header: bool = True,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    resolved_source = _resolve_file_path_or_url(file_path_or_url)
    source_path = urlparse(resolved_source).path if _is_url(resolved_source) else resolved_source
    suffix = Path(source_path).suffix.lower()

    if suffix == '.parquet':
        df = pl.read_parquet(resolved_source)
    elif suffix == '.csv':
        df = _read_csv_source(resolved_source, has_header=has_header)
    elif suffix == '.zip':
        df = _read_zip_source(resolved_source, has_header=has_header)
    else:
        raise ValueError(
            f"Unsupported file type for {file_path_or_url}. Supported extensions: "
            '.parquet, .csv, .zip.'
        )

    df = _validate_columns(df, columns)
    return df


def _normalize_datetime_column_by_name(
    df: pl.DataFrame,
    column_name: str,
) -> pl.DataFrame:
    if column_name not in df.columns:
        return df

    datetime_dtype = df.schema[column_name]

    if datetime_dtype == pl.Utf8:
        return df.with_columns(
            pl.col(column_name)
            .str.to_datetime(strict=False, time_unit='ms')
            .dt.replace_time_zone('UTC')
            .alias(column_name)
        )

    if datetime_dtype == pl.Date:
        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Datetime('ms'))
            .dt.replace_time_zone('UTC')
            .alias(column_name)
        )

    if isinstance(datetime_dtype, pl.Datetime):
        expr = pl.col(column_name).dt.cast_time_unit('ms')
        if datetime_dtype.time_zone is None:
            expr = expr.dt.replace_time_zone('UTC')
        else:
            expr = expr.dt.convert_time_zone('UTC')

        return df.with_columns(expr.alias(column_name))

    return df


def _normalize_datetime_column(df: pl.DataFrame) -> pl.DataFrame:
    return _normalize_datetime_column_by_name(df, 'datetime')


def _repair_second_encoded_datetime_column(
    df: pl.DataFrame,
    column_name: str,
) -> pl.DataFrame:
    if column_name not in df.columns or not isinstance(df.schema[column_name], pl.Datetime):
        return df

    return df.with_columns(
        pl.when(pl.col(column_name).dt.year() < _DATETIME_REPAIR_YEAR_CUTOFF)
        .then(
            (pl.col(column_name).dt.epoch('ms') * 1000)
            .cast(pl.Datetime('ms', time_zone='UTC'))
        )
        .otherwise(pl.col(column_name))
        .alias(column_name)
    )


def _normalize_timestamp_column(df: pl.DataFrame) -> pl.DataFrame:
    if 'timestamp' not in df.columns:
        return df

    timestamp_expr = pl.col('timestamp').cast(pl.Int64, strict=False)
    df = df.with_columns(
        pl.when(timestamp_expr < 10 ** 13)
        .then(timestamp_expr)
        .otherwise(timestamp_expr // 1000)
        .cast(pl.UInt64)
        .alias('timestamp')
    )

    if 'datetime' not in df.columns:
        df = df.with_columns(
            pl.col('timestamp')
            .cast(pl.Datetime('ms', time_zone='UTC'))
            .alias('datetime')
        )

    return df


def _normalize_generic_frame(df: pl.DataFrame) -> pl.DataFrame:
    df = _normalize_timestamp_column(df)
    df = _normalize_datetime_column(df)

    if 'datetime' in df.columns:
        df = df.sort('datetime')

    return df


def _base_interval_seconds(data: pl.DataFrame) -> int:
    if 'datetime' not in data.columns or data.height < _MIN_ROWS_TO_INFER_INTERVAL:
        raise ValueError('At least two datetime rows are required to infer base interval.')

    intervals = data.select(
        pl.col('datetime').diff().dt.total_seconds().alias('base_interval_seconds')
    ).drop_nulls()
    min_diff_seconds = (
        intervals
        .filter(pl.col('base_interval_seconds') > 0)
        .select(pl.col('base_interval_seconds').min())
        .item()
    )

    if min_diff_seconds is None:
        raise ValueError('Could not infer a positive base interval from datetime values.')

    return int(min_diff_seconds)


def _round_spot_kline_columns(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns([
        pl.col('mean').round(5),
        pl.col('std').round(6),
        pl.col('volume').round(9),
        pl.col('liquidity_sum').round(1),
        pl.col('maker_liquidity').round(1),
    ])


def _canonical_spot_kline_columns() -> list[str]:
    return [
        'datetime',
        'open',
        'high',
        'low',
        'close',
        'mean',
        'std',
        'volume',
        'maker_ratio',
        'no_of_trades',
        'open_liquidity',
        'high_liquidity',
        'low_liquidity',
        'close_liquidity',
        'liquidity_sum',
        'maker_volume',
        'maker_liquidity',
    ]


def _aggregate_spot_klines(data: pl.DataFrame, kline_size: int) -> pl.DataFrame:
    required_columns = {
        'datetime',
        'open',
        'high',
        'low',
        'close',
        'mean',
        'std',
        'volume',
        'maker_ratio',
        'no_of_trades',
        'open_liquidity',
        'high_liquidity',
        'low_liquidity',
        'close_liquidity',
        'liquidity_sum',
        'maker_volume',
        'maker_liquidity',
    }
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            'Spot kline aggregation requires these columns in the source file: '
            + ', '.join(missing_columns)
        )

    data = data.sort('datetime')
    base_interval = _base_interval_seconds(data)

    if kline_size < base_interval:
        raise ValueError(
            f"kline_size={kline_size} is smaller than the source file interval "
            f"({base_interval} seconds). Sub-base aggregation is not supported."
        )

    if kline_size % base_interval != 0:
        raise ValueError(
            f"kline_size={kline_size} must be a multiple of the source file interval "
            f"({base_interval} seconds)."
        )

    canonical_columns = _canonical_spot_kline_columns()

    if kline_size == base_interval:
        return _round_spot_kline_columns(data.select(canonical_columns))

    bucket_ms = kline_size * 1000
    weighted = data.with_columns([
        (
            pl.col('datetime')
            .dt.epoch('ms')
            .floordiv(bucket_ms)
            .mul(bucket_ms)
            .cast(pl.Datetime('ms', time_zone='UTC'))
        ).alias('_bucket_datetime'),
        (
            pl.col('mean') * pl.col('no_of_trades').cast(pl.Float64)
        ).alias('_mean_weighted_sum'),
        (
            (pl.col('std').pow(2) + pl.col('mean').pow(2))
            * pl.col('no_of_trades').cast(pl.Float64)
        ).alias('_sum_of_squares'),
        (
            pl.col('maker_ratio') * pl.col('no_of_trades').cast(pl.Float64)
        ).alias('_maker_ratio_weighted_sum'),
    ])

    no_of_trades_sum = pl.col('no_of_trades').sum().cast(pl.Float64)
    mean_expr = pl.col('_mean_weighted_sum').sum() / no_of_trades_sum
    variance_expr = (
        pl.col('_sum_of_squares').sum() / no_of_trades_sum
    ) - mean_expr.pow(2)

    aggregated = (
        weighted
        .group_by('_bucket_datetime', maintain_order=True)
        .agg([
            pl.col('open').first().alias('open'),
            pl.col('high').max().alias('high'),
            pl.col('low').min().alias('low'),
            pl.col('close').last().alias('close'),
            mean_expr.alias('mean'),
            pl.when(variance_expr < 0)
            .then(0.0)
            .otherwise(variance_expr)
            .sqrt()
            .alias('std'),
            pl.col('volume').sum().alias('volume'),
            (
                pl.col('_maker_ratio_weighted_sum').sum() / no_of_trades_sum
            ).alias('maker_ratio'),
            pl.col('no_of_trades').sum().alias('no_of_trades'),
            pl.col('open_liquidity').first().alias('open_liquidity'),
            pl.col('high_liquidity').max().alias('high_liquidity'),
            pl.col('low_liquidity').min().alias('low_liquidity'),
            pl.col('close_liquidity').last().alias('close_liquidity'),
            pl.col('liquidity_sum').sum().alias('liquidity_sum'),
            pl.col('maker_volume').sum().alias('maker_volume'),
            pl.col('maker_liquidity').sum().alias('maker_liquidity'),
        ])
        .rename({'_bucket_datetime': 'datetime'})
        .select(canonical_columns)
        .sort('datetime')
    )

    return _round_spot_kline_columns(aggregated)


def _normalize_spot_dollar_klines(data: pl.DataFrame) -> pl.DataFrame:
    for column_name in ('start_datetime', 'end_datetime', 'datetime'):
        data = _normalize_datetime_column_by_name(data, column_name)
        data = _repair_second_encoded_datetime_column(data, column_name)

    if 'datetime' not in data.columns and 'start_datetime' in data.columns:
        data = data.with_columns(pl.col('start_datetime').alias('datetime'))

    required_columns = set(_canonical_spot_kline_columns())
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            'Spot dollar kline aggregation requires these columns in the source file: '
            + ', '.join(missing_columns)
        )

    return (
        data
        .select(_canonical_spot_kline_columns())
        .sort('datetime')
    )


def _with_spot_dollar_groups(
    data: pl.DataFrame,
    dollar_bar_size: int,
) -> pl.DataFrame:
    return (
        data
        .sort('datetime')
        .with_columns(pl.col('datetime').dt.date().alias('_dollar_day'))
        .with_columns(
            (
                pl.col('liquidity_sum').cum_sum().over('_dollar_day')
                - pl.col('liquidity_sum')
            ).alias('_day_liquidity_before')
        )
        .with_columns(
            (
                (pl.col('_day_liquidity_before') / float(dollar_bar_size))
                .floor()
                .cast(pl.UInt64)
            ).alias('_dollar_group')
        )
    )


def _aggregate_spot_dollar_klines(
    data: pl.DataFrame,
    dollar_bar_size: int,
    source_dollar_bar_size: int | None,
) -> pl.DataFrame:
    if source_dollar_bar_size is not None:
        if dollar_bar_size < source_dollar_bar_size:
            raise ValueError(
                f"dollar_bar_size={dollar_bar_size} is smaller than the source "
                f"file dollar bar size ({source_dollar_bar_size}). Sub-base "
                'aggregation is not supported.'
            )

        if dollar_bar_size % source_dollar_bar_size != 0:
            raise ValueError(
                f"dollar_bar_size={dollar_bar_size} must be a multiple of the "
                f"source file dollar bar size ({source_dollar_bar_size})."
            )

        if dollar_bar_size == source_dollar_bar_size:
            return _round_spot_kline_columns(data)

    grouped = _with_spot_dollar_groups(data, dollar_bar_size)

    weighted = grouped.with_columns([
        (
            pl.col('mean') * pl.col('no_of_trades').cast(pl.Float64)
        ).alias('_mean_weighted_sum'),
        (
            (pl.col('std').pow(2) + pl.col('mean').pow(2))
            * pl.col('no_of_trades').cast(pl.Float64)
        ).alias('_sum_of_squares'),
        (
            pl.col('maker_ratio') * pl.col('no_of_trades').cast(pl.Float64)
        ).alias('_maker_ratio_weighted_sum'),
    ])

    no_of_trades_sum = pl.col('no_of_trades').sum().cast(pl.Float64)
    mean_expr = pl.col('_mean_weighted_sum').sum() / no_of_trades_sum
    variance_expr = (
        pl.col('_sum_of_squares').sum() / no_of_trades_sum
    ) - mean_expr.pow(2)

    aggregated = (
        weighted
        .group_by(['_dollar_day', '_dollar_group'], maintain_order=True)
        .agg([
            pl.col('datetime').first().alias('datetime'),
            pl.col('open').first().alias('open'),
            pl.col('high').max().alias('high'),
            pl.col('low').min().alias('low'),
            pl.col('close').last().alias('close'),
            mean_expr.alias('mean'),
            pl.when(variance_expr < 0)
            .then(0.0)
            .otherwise(variance_expr)
            .sqrt()
            .alias('std'),
            pl.col('volume').sum().alias('volume'),
            (
                pl.col('_maker_ratio_weighted_sum').sum() / no_of_trades_sum
            ).alias('maker_ratio'),
            pl.col('no_of_trades').sum().alias('no_of_trades'),
            pl.col('open_liquidity').first().alias('open_liquidity'),
            pl.col('high_liquidity').max().alias('high_liquidity'),
            pl.col('low_liquidity').min().alias('low_liquidity'),
            pl.col('close_liquidity').last().alias('close_liquidity'),
            pl.col('liquidity_sum').sum().alias('liquidity_sum'),
            pl.col('maker_volume').sum().alias('maker_volume'),
            pl.col('maker_liquidity').sum().alias('maker_liquidity'),
        ])
        .select(_canonical_spot_kline_columns())
        .sort('datetime')
    )

    return _round_spot_kline_columns(aggregated)


class HistoricalData:

    '''Stateful file-backed data access surface for Limen.'''

    DEFAULT_SPOT_KLINES_DATASET_REPO: Final[str] = _DEFAULT_SPOT_DATASET_REPO
    DEFAULT_SPOT_DOLLAR_KLINES_DATASET_REPO: Final[str] = (
        _DEFAULT_SPOT_DOLLAR_DATASET_REPO
    )

    def __init__(self) -> None:
        self.data = pl.DataFrame()
        self.data_columns: list[str] = []

    def _store(self, data: pl.DataFrame) -> pl.DataFrame:
        self.data = data
        self.data_columns = data.columns
        return data

    def get_binance_file(
        self,
        file_url: str,
        has_header: bool = False,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:

        '''Load a Binance archive file and normalize timestamp/datetime columns.'''

        data = binance_file_to_polars(file_url, has_header=has_header)
        data = _validate_columns(data, columns)
        data = _normalize_generic_frame(data)
        return self._store(data)

    def get_any_file(
        self,
        file_path_or_url: str,
        has_header: bool = True,
        columns: list[str] | None = None,
        row_count_limit: int | None = None,
        *,
        n_rows: int | None = None,
    ) -> pl.DataFrame:

        '''Load a local path or URL into Polars.

        Supported file types are `.parquet`, `.csv`, and `.zip`. Row limits
        return the latest rows. `n_rows` is accepted as a legacy alias.
        '''

        row_count_limit = _resolve_row_count_limit(row_count_limit, n_rows)

        data = _read_any_file(
            file_path_or_url,
            has_header=has_header,
            columns=columns,
        )
        data = _normalize_generic_frame(data)
        if row_count_limit is not None:
            data = data.tail(row_count_limit)

        return self._store(data)

    def get_spot_klines(
        self,
        row_count_limit: int | None = None,
        kline_size: int = 60,
        start_date_limit: str | None = None,
        end_date_limit: str | None = None,
        *,
        n_rows: int | None = None,
    ) -> pl.DataFrame:

        '''Load BTCUSDT spot klines from a file and aggregate upward when needed.

        By default this resolves the latest snapshot from the Hugging Face
        BTCUSDT 1m, 15m, 30m, 1h, 2h, or 4h kline dataset when `kline_size`
        matches. Other intervals use the 1m dataset and aggregate upward. Row limits
        return the latest rows. `n_rows` is accepted as a legacy alias.
        '''

        row_count_limit = _resolve_row_count_limit(row_count_limit, n_rows)
        kline_size = _validate_positive_int(kline_size, 'kline_size') or 60
        start_date_limit = _normalize_datetime_literal(start_date_limit, 'start_date_limit')
        end_date_limit = _normalize_datetime_literal(
            end_date_limit,
            'end_date_limit',
            end_of_day=True,
        )
        if (
            start_date_limit is not None
            and end_date_limit is not None
            and row_count_limit is not None
        ):
            raise ValueError(
                'row_count_limit must be None when both start_date_limit '
                'and end_date_limit are set.'
            )

        dataset_repo = _spot_dataset_repo_for_kline_size(
            kline_size,
            self.DEFAULT_SPOT_KLINES_DATASET_REPO,
        )
        base_data = _read_any_file(dataset_repo, has_header=True)
        base_data = _normalize_generic_frame(base_data)

        if start_date_limit is not None:
            start_datetime = datetime.strptime(
                start_date_limit, '%Y-%m-%d %H:%M:%S'
            ).replace(tzinfo=timezone.utc)
            base_data = base_data.filter(
                pl.col('datetime') >= pl.lit(start_datetime)
            )

        if end_date_limit is not None:
            end_datetime = datetime.strptime(
                end_date_limit, '%Y-%m-%d %H:%M:%S'
            ).replace(tzinfo=timezone.utc)
            base_data = base_data.filter(
                pl.col('datetime') <= pl.lit(end_datetime)
            )

        data = _aggregate_spot_klines(base_data, kline_size)

        if row_count_limit is not None:
            data = data.tail(row_count_limit)

        return self._store(data)

    def get_spot_dollar_klines(
        self,
        row_count_limit: int | None = None,
        dollar_bar_size: int = 1_000_000,
        start_date_limit: str | None = None,
        end_date_limit: str | None = None,
        *,
        n_rows: int | None = None,
    ) -> pl.DataFrame:

        '''Load BTCUSDT spot dollar klines from Hugging Face snapshots.

        Native Vaquum dollar-bar datasets are used when `dollar_bar_size`
        matches a published snapshot. Other supported multiples use the
        largest available lower-resolution dollar-bar source and aggregate
        upward. Row limits return the latest rows. `n_rows` is accepted as a
        legacy alias.
        '''

        row_count_limit = _resolve_row_count_limit(row_count_limit, n_rows)
        dollar_bar_size = (
            _validate_positive_int(dollar_bar_size, 'dollar_bar_size') or 1_000_000
        )
        start_date_limit = _normalize_datetime_literal(start_date_limit, 'start_date_limit')
        end_date_limit = _normalize_datetime_literal(
            end_date_limit,
            'end_date_limit',
            end_of_day=True,
        )
        if (
            start_date_limit is not None
            and end_date_limit is not None
            and row_count_limit is not None
        ):
            raise ValueError(
                'row_count_limit must be None when both start_date_limit '
                'and end_date_limit are set.'
            )

        dataset_repo, source_dollar_bar_size = _spot_dollar_dataset_repo_for_bar_size(
            dollar_bar_size,
            self.DEFAULT_SPOT_DOLLAR_KLINES_DATASET_REPO,
        )
        base_data = _read_any_file(dataset_repo, has_header=True)
        base_data = _normalize_spot_dollar_klines(base_data)

        if start_date_limit is not None:
            start_datetime = datetime.strptime(
                start_date_limit, '%Y-%m-%d %H:%M:%S'
            ).replace(tzinfo=timezone.utc)
            base_data = base_data.filter(
                pl.col('datetime') >= pl.lit(start_datetime)
            )

        if end_date_limit is not None:
            end_datetime = datetime.strptime(
                end_date_limit, '%Y-%m-%d %H:%M:%S'
            ).replace(tzinfo=timezone.utc)
            base_data = base_data.filter(
                pl.col('datetime') <= pl.lit(end_datetime)
            )

        data = _aggregate_spot_dollar_klines(
            base_data,
            dollar_bar_size,
            source_dollar_bar_size,
        )

        if row_count_limit is not None:
            data = data.tail(row_count_limit)

        return self._store(data)

    def get_arrow_file(
        self,
        file_path: str,
        row_count_limit: int | None = None,
        start_date_limit: str | None = None,
        end_date_limit: str | None = None,
        *,
        n_rows: int | None = None,
    ) -> pl.DataFrame:

        '''Zero-copy read of a local single-batch Arrow IPC file.

        Memory-maps `file_path` and returns a Polars DataFrame whose columns are
        views onto the mapping: `df[col].to_numpy(allow_copy=False)` succeeds and
        shares the mapped memory -- no deserialization, no buffer copy. Built for
        the tdw Arrow bar store (`/opt/arrow/<series>/latest.arrow`): a single
        uncompressed record batch with a strictly increasing `Int64`-nanosecond
        `ts` index. When `ts` is present, a `datetime` column is added as a
        zero-cost reinterpret of it, since the rest of Limen keys on `datetime`.

        Date range and row limits stay zero-copy: a date range maps to a
        contiguous slice of the sorted index, and `row_count_limit` returns the
        latest rows as a slice. Input mirrors `get_spot_klines`. The memory map's
        lifetime is owned by the returned frame (kept on the instance via
        `_store`); keep that frame referenced while holding zero-copy views.

        Raises if the file is not a single record batch, since it then cannot be
        served zero-copy. `n_rows` is accepted as a legacy alias.
        '''

        row_count_limit = _resolve_row_count_limit(row_count_limit, n_rows)
        start_date_limit = _normalize_datetime_literal(
            start_date_limit, 'start_date_limit'
        )
        end_date_limit = _normalize_datetime_literal(
            end_date_limit, 'end_date_limit', end_of_day=True
        )
        if (
            start_date_limit is not None
            and end_date_limit is not None
            and row_count_limit is not None
        ):
            raise ValueError(
                'row_count_limit must be None when both start_date_limit '
                'and end_date_limit are set.'
            )

        data = pl.read_ipc(file_path, memory_map=True, rechunk=False)
        if data.n_chunks() != 1:
            raise ValueError(
                f"{file_path} is not a single Arrow record batch "
                f"(n_chunks={data.n_chunks()}); it cannot be served zero-copy."
            )

        if (
            'datetime' not in data.columns
            and 'ts' in data.columns
            and data.schema['ts'].is_integer()
        ):
            data = data.with_columns(
                pl.col('ts')
                .cast(pl.Datetime('ns', time_zone='UTC'))
                .alias('datetime')
            )

        data = _slice_arrow_to_date_range(data, start_date_limit, end_date_limit)

        if row_count_limit is not None:
            data = data.tail(row_count_limit)

        return self._store(data)
