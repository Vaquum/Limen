# `limen.data`

> Fetch historical Binance market data (klines, spot trades, futures trades, agg-trades) into a Polars DataFrame.

## Responsibilities

Provides a single `HistoricalData` class that wraps all remote and file-based data-access calls and populates `self.data` for downstream use.

Does **not** own feature engineering, bar formation, or any data transformation — it delivers raw OHLCV-style frames only.

## Key concepts

- **HistoricalData** – stateful data-access class; each `get_*` method writes results to `self.data` (a `pl.DataFrame`) and `self.data_columns`
- **binance_file_to_polars** (`_internal/`) – downloads a Binance vision CSV/ZIP file by URL and parses it into Polars
- **generic_endpoints** (`_internal/`) – low-level wrappers around the Vaquum data API for klines and raw trade tables
- **standard_bars** (`bars/`) – bar-formation utilities consumed by the manifest pipeline
- **splits** (`utils/`) – sequential train/val/test split helpers used by `Manifest.prepare_data()`

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `HistoricalData` | `historical_data.py` | Instantiate (optionally with `auth_token`), then call one `get_*` method |
| `get_spot_klines()` | `historical_data.py` | Fetches OHLCV klines for Binance spot |
| `get_futures_klines()` | `historical_data.py` | Fetches OHLCV klines for Binance futures |
| `get_spot_trades()` | `historical_data.py` | Fetches raw spot trade records |
| `get_futures_trades()` | `historical_data.py` | Fetches raw futures trade records |
| `get_spot_agg_trades()` | `historical_data.py` | Fetches aggregated spot trades |
| `get_binance_file()` | `historical_data.py` | Loads a Binance vision file directly from its URL |
| `_get_data_for_test()` | `historical_data.py` | Loads a local CSV sample; used only in SFD test runs |

## Dependencies

- **Internal:** `limen.data._internal` (binance file parser, API endpoints), `limen.data.bars`, `limen.data.utils`
- **External:** `polars`, `pandas` (test helper only)

## Quick orientation

```text
data/
├── historical_data.py     # Public HistoricalData class
├── _internal/
│   ├── binance_file_to_polars.py   # URL → pl.DataFrame parser
│   └── generic_endpoints.py        # Raw API query helpers
├── bars/
│   └── standard_bars.py            # Bar formation (used by Manifest)
└── utils/
    ├── compute_data_bars.py         # Bar computation utilities
    ├── random_slice.py              # Random window sampling
    └── splits.py                    # Sequential train/val/test splitting
```

## Gotchas / things to know

- All `get_*` methods are side-effectful: they write to `self.data` rather than returning a value (except `get_futures_trades`, which also returns the DataFrame)
- `timestamp` is normalised to milliseconds and cast to `pl.Datetime('ms')`; a `datetime` column is always added
- `_get_data_for_test()` reads from `datasets/klines_2h_2020_2025.csv` relative to the working directory — it is only for test harnesses and must not be used in production
- `auth_token` is required for endpoints that hit the private Vaquum API; leave `None` for file-based or public calls
