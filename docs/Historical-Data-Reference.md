# Historical data API reference

`HistoricalData` retrieves recorded files and hosted datasets, returns a Polars frame and retains the loaded data on its instance. The [Python project](examples/python-research/README.md) can use ordinary file I/O; the [Compiled project](examples/compiled-research/README.md) declares a source method and parameters in YAML. The [observation chapter](Historical-Data.md) fixes the population before either research path.

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
def get_spot_klines(self, row_count_limit: int | None=None, kline_size: int=60, start_date_limit: str | None=None, end_date_limit: str | None=None, *, n_rows: int | None=None, lookback_days: float | None=None) -> pl.DataFrame:
    ...
```

Important rules:

- sub-1-minute klines are not supported
- `kline_size` must be a multiple of the source file interval
- `row_count_limit` returns the latest rows after date filtering and aggregation
- date-only `end_date_limit` values include the full named day
- `start_date_limit` and `end_date_limit` may define a closed window only when `row_count_limit` is unset

Aggregated bars use UTC buckets anchored at the Unix epoch, with `datetime` labelling the bucket start. Date filtering happens before aggregation, so fetch edges can contain partial buckets. [Manifest date windows](Experiment-Manifest.md) require aligned ends for directly declared native spot sources; use a day-dividing interval for ordinary YAML date bounds, or choose window ends on the interval's epoch grid.

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
def get_spot_dollar_klines(self, row_count_limit: int | None=None, dollar_bar_size: int=1000000, start_date_limit: str | None=None, end_date_limit: str | None=None, *, n_rows: int | None=None, lookback_days: float | None=None) -> pl.DataFrame:
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
def get_any_file(self, file_path_or_url: str, has_header: bool=True, columns: list[str] | None=None, row_count_limit: int | None=None, *, n_rows: int | None=None, expected_sha256: str | None=None) -> pl.DataFrame:
    ...
```

`expected_sha256` pins a local Parquet file. Limen verifies its bytes before decoding those same bytes, on both initial loading and CLI resume. The recovery YAML declares this digest; a changed source fails before another candidate can fit.

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

## Recorded format equivalence

The bundle's `recorded-hourly.csv`, `recorded-hourly.zip` and single-batch uncompressed `recorded-hourly.arrow` preserve the first 1,440 hourly observations exactly. ZIP contains that CSV. They are encodings of the pinned source, not new market observations. The project tests compare UTC nanoseconds and every OHLCV value with the original Parquet population, including the latest-row limit and retained instance state.

### Python Research

<!-- example: sources-python-formats source=docs/examples/python-research/source_cases.py region=source_formats scenario=search interface=python role=executed -->
```python
def read_source_formats(data_dir):
    report = {}
    for suffix in ('csv', 'zip', 'arrow'):
        path = data_dir / ('recorded-hourly.' + suffix)
        historical = HistoricalData()
        frame = (historical.get_arrow_file(str(path)) if suffix == 'arrow' else historical.get_any_file(str(path)))
        latest = (historical.get_arrow_file(str(path), row_count_limit=12) if suffix == 'arrow' else historical.get_any_file(str(path), row_count_limit=12))
        if not latest.equals(frame.tail(12)) or historical.data is not latest:
            raise ValueError('Local source changed row-limit or retained-frame semantics')
        report[suffix] = {'rows': frame.height, 'columns': frame.columns, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'timestamps_ns': frame['datetime'].dt.epoch('ns').to_list(),
                         'values': frame.select('open', 'high', 'low', 'close', 'volume').to_dicts()}
    return report
```

### Compiled Research

Each complete source variant names the appropriate retrieval method and the bundled file. No conversion step is required:

<!-- example: sources-compiled-formats source=docs/examples/compiled-research/run.sh region=source_formats scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar \
  compiled-research/source_csv.yaml \
  compiled-research/source_zip.yaml \
  compiled-research/source_arrow.yaml
```

A hosted file remains a current acquisition source. The unchanged canonical research uses its local pinned bytes; it does not silently replace them with the latest hosted snapshot. Recorded trade archives and observed-cost histories retain their separate input schemas and provenance.
