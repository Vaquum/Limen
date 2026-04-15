from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Final
from urllib.parse import urlparse
import zipfile

import polars as pl
import requests

from limen.data._internal.binance_file_to_polars import binance_file_to_polars


_SUPPORTED_DATETIME_FORMATS: Final[tuple[str, ...]] = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)
_REMOTE_TIMEOUT_SECONDS: Final[int] = 60
_DEFAULT_SPOT_DATASET_REPO: Final[str] = "vaquum/binance_btcusdt_1m_klines"
_DEFAULT_TEST_FILE_URL: Final[str] = (
    "https://raw.githubusercontent.com/Vaquum/Limen/refs/heads/main/"
    "datasets/klines_2h_2020_2025.csv"
)
_HUGGINGFACE_DATASET_REPO_PART_COUNT: Final[int] = 3
_MIN_ROWS_TO_INFER_INTERVAL: Final[int] = 2


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _normalize_datetime_literal(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None

    for fmt in _SUPPORTED_DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    raise ValueError(
        f"{field_name} must match one of: YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, "
        "YYYY-MM-DDTHH:MM:SS."
    )


def _validate_positive_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None

    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int.")

    if value < 1:
        raise ValueError(f"{field_name} must be at least 1.")

    return value


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
            (name for name in archive.namelist() if name.lower().endswith(".csv")),
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
    file_name = metadata["file_name"]
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_name}"


def _repo_id_from_huggingface_url(file_path_or_url: str) -> str | None:
    if not file_path_or_url.startswith("https://huggingface.co/datasets/"):
        return None

    path_parts = [part for part in urlparse(file_path_or_url).path.split("/") if part]
    if (
        len(path_parts) < _HUGGINGFACE_DATASET_REPO_PART_COUNT
        or path_parts[0] != "datasets"
    ):
        return None

    if "resolve" in path_parts or "blob" in path_parts:
        return None

    return "/".join(path_parts[1:3])


def _resolve_file_path_or_url(file_path_or_url: str) -> str:
    if file_path_or_url == _DEFAULT_SPOT_DATASET_REPO:
        return _resolve_huggingface_latest_file(_DEFAULT_SPOT_DATASET_REPO)

    repo_id = _repo_id_from_huggingface_url(file_path_or_url)
    if repo_id is not None:
        return _resolve_huggingface_latest_file(repo_id)

    return file_path_or_url


def _read_any_file(
    file_path_or_url: str,
    *,
    has_header: bool = True,
    columns: list[str] | None = None,
    n_rows: int | None = None,
) -> pl.DataFrame:
    n_rows = _validate_positive_int(n_rows, "n_rows")
    resolved_source = _resolve_file_path_or_url(file_path_or_url)
    source_path = urlparse(resolved_source).path if _is_url(resolved_source) else resolved_source
    suffix = Path(source_path).suffix.lower()

    if suffix == ".parquet":
        df = pl.read_parquet(resolved_source)
    elif suffix == ".csv":
        df = _read_csv_source(resolved_source, has_header=has_header)
    elif suffix == ".zip":
        df = _read_zip_source(resolved_source, has_header=has_header)
    else:
        raise ValueError(
            f"Unsupported file type for {file_path_or_url}. Supported extensions: "
            ".parquet, .csv, .zip."
        )

    df = _validate_columns(df, columns)
    if n_rows is not None:
        df = df.head(n_rows)

    return df


def _normalize_datetime_column(df: pl.DataFrame) -> pl.DataFrame:
    if "datetime" not in df.columns:
        return df

    datetime_dtype = df.schema["datetime"]

    if datetime_dtype == pl.Utf8:
        return df.with_columns(
            pl.col("datetime")
            .str.to_datetime(strict=False, time_unit="ms")
            .dt.replace_time_zone("UTC")
            .alias("datetime")
        )

    if datetime_dtype == pl.Date:
        return df.with_columns(
            pl.col("datetime")
            .cast(pl.Datetime("ms"))
            .dt.replace_time_zone("UTC")
            .alias("datetime")
        )

    if isinstance(datetime_dtype, pl.Datetime):
        expr = pl.col("datetime").dt.cast_time_unit("ms")
        if datetime_dtype.time_zone is None:
            expr = expr.dt.replace_time_zone("UTC")
        else:
            expr = expr.dt.convert_time_zone("UTC")

        return df.with_columns(expr.alias("datetime"))

    return df


def _normalize_timestamp_column(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" not in df.columns:
        return df

    timestamp_expr = pl.col("timestamp").cast(pl.Int64, strict=False)
    df = df.with_columns(
        pl.when(timestamp_expr < 10 ** 13)
        .then(timestamp_expr)
        .otherwise(timestamp_expr // 1000)
        .cast(pl.UInt64)
        .alias("timestamp")
    )

    if "datetime" not in df.columns:
        df = df.with_columns(
            pl.col("timestamp")
            .cast(pl.Datetime("ms", time_zone="UTC"))
            .alias("datetime")
        )

    return df


def _normalize_generic_frame(df: pl.DataFrame) -> pl.DataFrame:
    df = _normalize_timestamp_column(df)
    df = _normalize_datetime_column(df)

    if "datetime" in df.columns:
        df = df.sort("datetime")

    return df


def _base_interval_seconds(data: pl.DataFrame) -> int:
    if "datetime" not in data.columns or data.height < _MIN_ROWS_TO_INFER_INTERVAL:
        raise ValueError("At least two datetime rows are required to infer base interval.")

    intervals = data.select(
        pl.col("datetime").diff().dt.total_seconds().alias("base_interval_seconds")
    ).drop_nulls()
    min_diff_seconds = (
        intervals
        .filter(pl.col("base_interval_seconds") > 0)
        .select(pl.col("base_interval_seconds").min())
        .item()
    )

    if min_diff_seconds is None:
        raise ValueError("Could not infer a positive base interval from datetime values.")

    return int(min_diff_seconds)


def _round_spot_kline_columns(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns([
        pl.col("mean").round(5),
        pl.col("std").round(6),
        pl.col("volume").round(9),
        pl.col("liquidity_sum").round(1),
        pl.col("maker_liquidity").round(1),
    ])


def _aggregate_spot_klines(data: pl.DataFrame, kline_size: int) -> pl.DataFrame:
    required_columns = {
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "mean",
        "std",
        "volume",
        "maker_ratio",
        "no_of_trades",
        "open_liquidity",
        "high_liquidity",
        "low_liquidity",
        "close_liquidity",
        "liquidity_sum",
        "maker_volume",
        "maker_liquidity",
    }
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "Spot kline aggregation requires these columns in the source file: "
            + ", ".join(missing_columns)
        )

    data = data.sort("datetime")
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

    canonical_columns = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "mean",
        "std",
        "volume",
        "maker_ratio",
        "no_of_trades",
        "open_liquidity",
        "high_liquidity",
        "low_liquidity",
        "close_liquidity",
        "liquidity_sum",
        "maker_volume",
        "maker_liquidity",
    ]

    if kline_size == base_interval:
        return _round_spot_kline_columns(data.select(canonical_columns))

    bucket_ms = kline_size * 1000
    weighted = data.with_columns([
        (
            pl.col("datetime")
            .dt.epoch("ms")
            .floordiv(bucket_ms)
            .mul(bucket_ms)
            .cast(pl.Datetime("ms", time_zone="UTC"))
        ).alias("_bucket_datetime"),
        (
            pl.col("mean") * pl.col("no_of_trades").cast(pl.Float64)
        ).alias("_mean_weighted_sum"),
        (
            (pl.col("std").pow(2) + pl.col("mean").pow(2))
            * pl.col("no_of_trades").cast(pl.Float64)
        ).alias("_sum_of_squares"),
        (
            pl.col("maker_ratio") * pl.col("no_of_trades").cast(pl.Float64)
        ).alias("_maker_ratio_weighted_sum"),
    ])

    no_of_trades_sum = pl.col("no_of_trades").sum().cast(pl.Float64)
    mean_expr = pl.col("_mean_weighted_sum").sum() / no_of_trades_sum
    variance_expr = (
        pl.col("_sum_of_squares").sum() / no_of_trades_sum
    ) - mean_expr.pow(2)

    aggregated = (
        weighted
        .group_by("_bucket_datetime", maintain_order=True)
        .agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            mean_expr.alias("mean"),
            pl.when(variance_expr < 0)
            .then(0.0)
            .otherwise(variance_expr)
            .sqrt()
            .alias("std"),
            pl.col("volume").sum().alias("volume"),
            (
                pl.col("_maker_ratio_weighted_sum").sum() / no_of_trades_sum
            ).alias("maker_ratio"),
            pl.col("no_of_trades").sum().alias("no_of_trades"),
            pl.col("open_liquidity").first().alias("open_liquidity"),
            pl.col("high_liquidity").max().alias("high_liquidity"),
            pl.col("low_liquidity").min().alias("low_liquidity"),
            pl.col("close_liquidity").last().alias("close_liquidity"),
            pl.col("liquidity_sum").sum().alias("liquidity_sum"),
            pl.col("maker_volume").sum().alias("maker_volume"),
            pl.col("maker_liquidity").sum().alias("maker_liquidity"),
        ])
        .rename({"_bucket_datetime": "datetime"})
        .select(canonical_columns)
        .sort("datetime")
    )

    return _round_spot_kline_columns(aggregated)


class HistoricalData:

    """Stateful file-backed data access surface for Limen."""

    DEFAULT_SPOT_KLINES_DATASET_REPO: Final[str] = _DEFAULT_SPOT_DATASET_REPO
    DEFAULT_TEST_FILE_URL: Final[str] = _DEFAULT_TEST_FILE_URL
    DEFAULT_TEST_FILE_PATH: Final[Path] = (
        Path(__file__).resolve().parents[2] / "datasets" / "klines_2h_2020_2025.csv"
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

        """Load a Binance archive file and normalize timestamp/datetime columns."""

        data = binance_file_to_polars(file_url, has_header=has_header)
        data = _validate_columns(data, columns)
        data = _normalize_generic_frame(data)
        return self._store(data)

    def get_any_file(
        self,
        file_path_or_url: str,
        has_header: bool = True,
        columns: list[str] | None = None,
        n_rows: int | None = None,
    ) -> pl.DataFrame:

        """Load a local path or URL into Polars.

        Supported file types are `.parquet`, `.csv`, and `.zip`.
        """

        data = _read_any_file(
            file_path_or_url,
            has_header=has_header,
            columns=columns,
            n_rows=n_rows,
        )
        data = _normalize_generic_frame(data)
        return self._store(data)

    def get_spot_klines(
        self,
        n_rows: int | None = None,
        kline_size: int = 60,
        start_date_limit: str | None = None,
        file_path_or_url: str | None = None,
    ) -> pl.DataFrame:

        """Load BTCUSDT spot klines from a file and aggregate upward when needed.

        By default this resolves the latest daily snapshot from the Hugging Face
        dataset repo `vaquum/binance_btcusdt_1m_klines`.
        """

        n_rows = _validate_positive_int(n_rows, "n_rows")
        kline_size = _validate_positive_int(kline_size, "kline_size") or 60
        start_date_limit = _normalize_datetime_literal(start_date_limit, "start_date_limit")

        source = file_path_or_url or self.DEFAULT_SPOT_KLINES_DATASET_REPO
        base_data = _read_any_file(source, has_header=True)
        base_data = _normalize_generic_frame(base_data)

        if start_date_limit is not None:
            start_datetime = datetime.strptime(
                start_date_limit, "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
            base_data = base_data.filter(
                pl.col("datetime") >= pl.lit(start_datetime)
            )

        data = _aggregate_spot_klines(base_data, kline_size)

        if n_rows is not None:
            data = data.head(n_rows)

        return self._store(data)
