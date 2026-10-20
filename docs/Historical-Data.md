# Historical Data

Begin with a fixed set of observations. The two projects use recorded files with declared digests; a hosted dataset is an acquisition source whose current snapshot may differ. `HistoricalData` provides five file-backed retrieval methods:

- `get_spot_klines()`
- `get_spot_dollar_klines()`
- `get_binance_file()`
- `get_any_file()`
- `get_arrow_file()`

All five return `polars.DataFrame`, and each call also updates `historical.data` and `historical.data_columns`.

Install `vaquum-limen[data]` when using Hugging Face parquet datasets or Arrow IPC paths. The base package does not install PyArrow.

## Prerequisites

- Python 3.10-3.13
- `pip install "vaquum-limen[data]"` for every built-in hosted or Arrow data path on this page
- network access for Hugging Face or remote URL examples; local files need no network

## Fix the observations before research

The [Python project's](examples/python-research/README.md) scenarios verify the complete recorded file before selecting a row prefix. Python Research can use ordinary file I/O; it does not require a Manifest or `HistoricalData`.

<!-- example: data-python-recorded source=docs/examples/python-research/scenarios.py region=recorded_input scenario=bars interface=python role=executed -->
```python
def recorded(data_dir, name, rows):
    source = data_dir / name
    if hashlib.sha256(source.read_bytes()).hexdigest() != INPUTS[name]:
        raise ValueError('Recorded input digest differs: ' + name)
    return pl.read_parquet(source, n_rows=rows)
```

The [Compiled project](examples/compiled-research/README.md) names the recorded quarter-hour source in its selection Manifest:

<!-- example: data-compiled-recorded source=docs/examples/compiled-research/selection.yaml region=recorded_source scenario=selection interface=compiled role=executed -->
```yaml
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_15m_20250101_20250531.parquet
```

Its driver verifies the input digest and resolves the documented `--data-dir` argument into this file declaration. The archived source and exact commands preserve that resolution. Both projects acquire their matching recorded inputs from the project bundle; neither requires a live dataset download during research proof.

`HistoricalData` remains stateful for Manifest compatibility. Each retrieval method returns the loaded frame and stores it on the instance as `data`, with column names in `data_columns`.

## Current Surface

| Method | Backend | Returns | Use case |
|---|---|---|---|
| `get_spot_klines()` | Hugging Face BTCUSDT parquet datasets | BTCUSDT spot klines as `pl.DataFrame` | standard experiment input |
| `get_spot_dollar_klines()` | Hugging Face BTCUSDT dollar-bar parquet datasets | BTCUSDT spot dollar bars as `pl.DataFrame` | event-time experiments |
| `get_binance_file()` | direct Binance ZIP/CSV archive | normalized Binance file contents as `pl.DataFrame` | source-native Binance trade files |
| `get_any_file()` | local path or URL (`.parquet`, `.csv`, `.zip`) | loaded file contents as `pl.DataFrame` | test fixtures, local research files, remote datasets |
| `get_arrow_file()` | local Arrow IPC file | memory-mapped Arrow contents as `pl.DataFrame` | zero-copy local Arrow reads when buffer layout permits |

## `get_spot_klines()`

`get_spot_klines()` reads from the BTCUSDT datasets published on Hugging Face.

By default it reads native [1m](https://huggingface.co/datasets/vaquum/binance_btcusdt_1m_klines), [15m](https://huggingface.co/datasets/vaquum/binance_btcusdt_15m_klines), [30m](https://huggingface.co/datasets/vaquum/binance_btcusdt_30m_klines), [1h](https://huggingface.co/datasets/vaquum/binance_btcusdt_1h_klines), [2h](https://huggingface.co/datasets/vaquum/binance_btcusdt_2h_klines), or [4h](https://huggingface.co/datasets/vaquum/binance_btcusdt_4h_klines) datasets for matching `kline_size` values, and otherwise the 1m dataset before aggregating upward.

Dataset snapshot file names are treated as the cache identity. Limen does not package upstream dataset bytes, licenses, or hash manifests inside the wheel/sdist; release packaging proves code and metadata integrity, while dataset publishers remain authoritative for dataset size, license, and content integrity. Cached snapshots live under `~/.cache/limen/datasets` and can be large; delete that directory to reclaim local disk or force a refetch.

<!-- reference: data-api-get-spot-klines source=limen/data/historical_data.py region=signature:HistoricalData.get_spot_klines role=signature -->
```python
def get_spot_klines(self, row_count_limit: int | None=None, kline_size: int=60, start_date_limit: str | None=None, end_date_limit: str | None=None, *, n_rows: int | None=None) -> pl.DataFrame:
    ...
```

Important rules:

- sub-1-minute klines are not supported
- `kline_size` must be a multiple of the source file interval
- `row_count_limit` returns the latest rows after date filtering and aggregation
- date-only `end_date_limit` values include the full named day
- `start_date_limit` and `end_date_limit` may define a closed window only when `row_count_limit` is unset

Returned columns:

- `datetime`, `open`, `high`, `low`, `close`
- `mean`, `std`
- `volume`, `maker_ratio`, `no_of_trades`
- `open_liquidity`, `high_liquidity`, `low_liquidity`, `close_liquidity`
- `liquidity_sum`, `maker_volume`, `maker_liquidity`

## `get_spot_dollar_klines()`

`get_spot_dollar_klines()` reads BTCUSDT dollar bars from Vaquum Hugging Face datasets.

It reads native [1M](https://huggingface.co/datasets/vaquum/binance_btcusdt_1M_dollar_klines), [15M](https://huggingface.co/datasets/vaquum/binance_btcusdt_15M_dollar_klines), [30M](https://huggingface.co/datasets/vaquum/binance_btcusdt_30M_dollar_klines), [60M](https://huggingface.co/datasets/vaquum/binance_btcusdt_60M_dollar_klines), [120M](https://huggingface.co/datasets/vaquum/binance_btcusdt_120M_dollar_klines), or [240M](https://huggingface.co/datasets/vaquum/binance_btcusdt_240M_dollar_klines) datasets for matching `dollar_bar_size` values. Other multiples use the largest available lower dollar-bar source and aggregate upward.

<!-- reference: data-api-get-spot-dollar-klines source=limen/data/historical_data.py region=signature:HistoricalData.get_spot_dollar_klines role=signature -->
```python
def get_spot_dollar_klines(self, row_count_limit: int | None=None, dollar_bar_size: int=1000000, start_date_limit: str | None=None, end_date_limit: str | None=None, *, n_rows: int | None=None) -> pl.DataFrame:
    ...
```

Important rules:

- `dollar_bar_size` must be a positive integer
- supported native sizes are resolved directly instead of deriving from the 1-minute kline file
- non-native sizes must be multiples of the selected source dollar-bar size
- returned `datetime` is the dollar bar start time
- returned columns match `get_spot_klines()` for manifest compatibility

## `get_binance_file()`

`get_binance_file()` keeps the same role as before: load a Binance archive directly and normalize its `timestamp` / `datetime` columns.

<!-- reference: data-api-get-binance-file source=limen/data/historical_data.py region=signature:HistoricalData.get_binance_file role=signature -->
```python
def get_binance_file(self, file_url: str, has_header: bool=False, columns: list[str] | None=None) -> pl.DataFrame:
    ...
```

Use this for Binance source files rather than the curated kline dataset.

## `get_any_file()`

`get_any_file()` is the generic file ingestion path. It accepts a local path or URL and currently supports:

- `.parquet`
- `.csv`
- `.zip`

<!-- reference: data-api-get-any-file source=limen/data/historical_data.py region=signature:HistoricalData.get_any_file role=signature -->
```python
def get_any_file(self, file_path_or_url: str, has_header: bool=True, columns: list[str] | None=None, row_count_limit: int | None=None, *, n_rows: int | None=None) -> pl.DataFrame:
    ...
```

It is the right choice for:

- local fixtures in tests
- repo-hosted CSV files
- remote parquet snapshots
- manifest test data sources

## `get_arrow_file()`

<!-- reference: data-api-get-arrow-file source=limen/data/historical_data.py region=signature:HistoricalData.get_arrow_file role=signature -->
```python
def get_arrow_file(self, file_path: str, row_count_limit: int | None=None, start_date_limit: str | None=None, end_date_limit: str | None=None, *, n_rows: int | None=None) -> pl.DataFrame:
    ...
```

The local Arrow IPC file must contain one uncompressed record batch. A strictly increasing Int64 nanosecond `ts` index permits date filtering and latest-row limits through contiguous slices; `datetime` is added from `ts` when absent. A datetime-only date filter materializes. Multi-batch and compressed files are refused instead of claiming zero-copy access. Keep the returned frame referenced while retaining views into its mapped buffers.

## Manifest Integration

Use `HistoricalData.get_any_file` when the experiment names a specific recorded local file or URL, as the Compiled project does above. Hosted `get_spot_klines` and `get_spot_dollar_klines` resolve the provider's available snapshots. Preserve the actual input identity when reproducing a run; a dataset name alone cannot identify its bytes. A bounded, coarser test source can shorten preparation diagnostics, but it changes the population and cannot stand in for proof of the full experiment.

## Choosing The Right Surface

- Use `get_spot_klines()` for standard Limen experiments and manifest test sources that should stay on the public BTCUSDT path.
- Use `get_spot_dollar_klines()` when market activity, not wall-clock time, should define each row.
- Use `get_binance_file()` for direct Binance archives.
- Use `get_any_file()` for local fixtures, URLs, and generic file-backed ingestion.
- Use `get_arrow_file()` for local Arrow IPC files when memory mapping and zero-copy validation matter.

## Read Next

- [Data Bars](Data-Bars.md)
- [Single-File Decoder](Single-File-Decoder.md)
- [Experiment Manifest](Experiment-Manifest.md)
