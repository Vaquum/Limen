# Historical Data

A fixed set of observations establishes the research population. The two projects use recorded files with declared digests; a hosted dataset supplies a snapshot whose current contents can differ. The [complete source API](Historical-Data-Reference.md) defines acquisition paths for caller-supplied observations and hosted sources.

## Decide which observations remain final

Before researching a protected final window, choose the Compiled project's separate `final/` or `final-event/` branch and reserve its declared dates. The complete branch starts with `limen reserve`, runs its YAML and finishes with `limen assess --reservation`; follow [final assessment](Final-Holdout.md) for those exact commands. Do not run exploratory cases against that branch's reserved observations first. The project ledger records local known exposure; downloading a file or starting a new Git directory does not establish a global never-observed claim.

Python Research excludes final observations in its caller-owned preparation and selection code. It does not acquire a native reservation merely by supplying a cutoff. Both projects state the population each result measures.

## Prerequisites

- Python 3.10-3.13
- `pip install "vaquum-limen[data]"` for every built-in hosted or Arrow data path on this page
- network access for Hugging Face or remote URL examples; local files need no network

## Source choice determines origin and resolution

`HistoricalData` provides hosted BTCUSDT spot candles and dollar bars, Binance archives, local or HTTP(S) Parquet/CSV/ZIP files, and a specialized local Arrow IPC path. Ordinary Python Research can also supply observations acquired through caller-owned code.

| Source path | Native choices or contract | Research consequence |
|---|---|---|
| Hosted spot candles | 1-minute, 15-minute, 30-minute, 1-hour, 2-hour, and 4-hour BTCUSDT observations | Supported larger intervals aggregate an admitted native source. |
| Hosted dollar bars | 1, 15, 30, 60, 120, and 240 million USDT of quote turnover | Activity determines elapsed time between observations. |
| Local or remote file | Parquet, CSV, and ZIP with declared columns and timestamps | Source identity, schema, units, and clock semantics come from the supplied data. |
| Binance archive | Declared archive format and recorded timestamp convention | Archive ingestion preserves the distinction between candles and individual trades. |
| Local Arrow IPC | Uncompressed, single-record-batch file | Memory mapping has a narrower file contract than the general loader. |

Hosted acquisition resolves and caches source snapshots, with retried requests and atomic cache writes. A changing hosted source therefore needs a retained snapshot identity for a repeated research comparison. `expected_sha256` pins local Parquet bytes under its specific loader contract. The [source reference](Historical-Data-Reference.md) defines acquisition arguments, aggregation, filtering, and cache behavior.

## Population selection preserves an observation policy

Date bounds, latest-row limits, and elapsed-day lookbacks answer different population questions. Acquisition includes the entire date-only end day. Experiment date partitions use half-open UTC intervals. The distinction matters when the same date appears in acquisition and split declarations.

Sequential and date-based partitions retain chronology. Seeded random partitions provide disjoint samples under their declared design. Walk-forward adds anchored or rolling training, purge, and embargo. Pre-split selectors can retain a trailing elapsed-time window or one complete contiguous block of a specified regime. Their [utility contracts](Utilities.md#temporal-perturbation-selectors) define selection and unavailable cases.

The selection policy belongs beside the result. A latest-row limit depends on the source endpoint, while a fixed date window names its endpoints directly. A regime-selected block changes the assessed population and requires the recorded selection identity.

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

The bundle pins the source by SHA-256. The YAML reads its literal `inputs/` path from the bundle root. Both projects use the matching recorded-input bundle; research proof requires no live dataset download.

`HistoricalData` remains stateful for Manifest compatibility. Each retrieval method returns the loaded frame and stores it on the instance as `data`, with column names in `data_columns`.

## The file-format diagnostics preserve the same observations

The Python format case reads the bundle's equivalent CSV, ZIP, and Arrow files through `HistoricalData`. It checks that a 12-row limit returns the final 12 observations and that the stateful loader retains that returned frame:

<!-- example: historical-python-formats source=docs/examples/python-research/source_cases.py region=source_formats scenario=search interface=python role=executed -->
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

The Compiled format cases execute the corresponding literal file declarations:

<!-- example: historical-compiled-formats source=docs/examples/compiled-research/run.sh region=source_formats scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar \
  compiled-research/source_csv.yaml \
  compiled-research/source_zip.yaml \
  compiled-research/source_arrow.yaml
```

<!-- result: historical-python-formats interface=python scenario=search fields=input_formats.csv.rows,input_formats.zip.rows,input_formats.arrow.rows -->

Independent checks reconcile timestamps and OHLCV values across formats. The native Compiled checks also require matching candidate identities and measured accuracy. File equivalence preserves the recorded observations; feature preparation and research partitions still follow their own declarations.

## Carry the declared population forward

The selection examples use quarter-hour BTCUSDT observations. Their `datetime` is the start of the interval in UTC; prices are USDT per BTC and volume is BTC. A close can be used only after the interval completes. The Python inference adapter records that 900-second availability delay explicitly. Native execution declares its source interval with the trading policy.

Python selects 1,920 recorded rows: the first 960 may fit state, the next 480 may select it and the final 480 supply later decisions. Feature warm-up and forward labels remove rows within each partition. Compiled selection instead uses its literal half-open calendar dates; it is a different population. Later variants name their changed windows explicitly.

Keep the acquired `inputs.json`, raw files and publication identity together. Funding and retrospective observed-cost inputs carry separate source clocks and provenance; their reported availability must not be inferred from a bar timestamp. The three supplied target-label files are deterministic software labels derived within chronological partitions, rather than observed venue trades.

[Historical data API](Historical-Data-Reference.md) describes local Parquet/CSV/ZIP, hosted sources, Binance archives and Arrow IPC, including filtering and cache behavior. Those acquisition methods do not make a changing hosted snapshot equivalent to the pinned project input. Continue to [observation and label preparation](Data-Bars.md) with the unchanged file or frame.
