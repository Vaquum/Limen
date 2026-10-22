# Historical Data

Begin with a fixed set of observations. The two projects use recorded files with declared digests; a hosted dataset is an acquisition source whose current snapshot may differ. The [complete source API](Historical-Data-Reference.md) remains available when your own observations require a different acquisition path.

## Decide which observations remain final

Before researching a protected final window, choose the Compiled project's separate `final/` or `final-event/` branch and reserve its declared dates. The complete branch starts with `limen reserve`, runs its YAML and finishes with `limen assess --reservation`; follow [final assessment](Final-Holdout.md) for those exact commands. Do not run exploratory cases against that branch's reserved observations first. The project ledger records local known exposure; downloading a file or starting a new Git directory does not establish a global never-observed claim.

Python Research excludes final observations in its caller-owned preparation and selection code. It does not acquire a native reservation merely by supplying a cutoff. Both projects state the population each result measures.

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

The bundle pins the source by SHA-256. The YAML reads its literal `inputs/` path from the bundle root. Both projects use the matching recorded-input bundle; research proof requires no live dataset download.

`HistoricalData` remains stateful for Manifest compatibility. Each retrieval method returns the loaded frame and stores it on the instance as `data`, with column names in `data_columns`.

## Carry the declared population forward

The selection examples use quarter-hour BTCUSDT observations. Their `datetime` is the start of the interval in UTC; prices are USDT per BTC and volume is BTC. A close can be used only after the interval completes. The Python inference adapter records that 900-second availability delay explicitly. Native execution declares its source interval with the trading policy.

Python selects 1,920 recorded rows: the first 960 may fit state, the next 480 may select it and the final 480 supply later decisions. Feature warm-up and forward labels remove rows within each partition. Compiled selection instead uses its literal half-open calendar dates; it is a different population. Later variants name their changed windows explicitly.

Keep the acquired `inputs.json`, raw files and publication identity together. Funding and retrospective observed-cost inputs carry separate source clocks and provenance; their reported availability must not be inferred from a bar timestamp. The three supplied target-label files are deterministic software labels derived within chronological partitions, rather than observed venue trades.

[Historical data API](Historical-Data-Reference.md) describes local Parquet/CSV/ZIP, hosted sources, Binance archives and Arrow IPC, including filtering and cache behavior. Those acquisition methods do not make a changing hosted snapshot equivalent to the pinned project input. Continue to [observation and label preparation](Data-Bars.md) with the unchanged file or frame.
