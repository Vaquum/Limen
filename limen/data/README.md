# `limen.data`

> Fetch, normalise, and prepare historical Binance market data for use in experiments.

## Responsibilities

Owns all data ingestion (klines, trades, agg-trades — spot and futures), bar formation from raw tick/trade data, train/val/test splitting, and the helper utilities that glue raw data to the prep pipeline.
Does **not** own feature engineering, indicators, or model training.

## Key concepts

- **HistoricalData** – stateful class; each `get_*` call populates `self.data` with a Polars DataFrame and `self.data_columns` with the column list.
- **Bars** – `standard_bars.py` aggregates tick/trade data into fixed-time or volume bars; accessed via `compute_data_bars`.
- **Splits** – `splits.py` provides `split_sequential` (ordered train/val/test proportions) and `split_random` (random sampling), plus `split_data_to_prep_output` which converts split DataFrames into the `data_dict` format consumed by the experiment loop.
- **`_internal`** – private helpers that handle Binance file download (`binance_file_to_polars`) and generic database / API queries (`generic_endpoints`); not part of the public API.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `HistoricalData` | `historical_data.py` | Instantiate with optional `auth_token`; call `get_spot_klines()` etc. to populate `.data` |
| `compute_data_bars()` | `utils/compute_data_bars.py` | Called by `Manifest.set_bar_formation()` to convert raw data into OHLCV bars |
| `split_sequential()` | `utils/splits.py` | Used internally by `Manifest.prepare_data()` to partition data |
| `split_data_to_prep_output()` | `utils/splits.py` | Converts a list of split DataFrames into the `data_dict` used by model functions |

## Dependencies

- **Internal:** none — this is a leaf module
- **External:** `polars`, `pandas` (test helper only)

## Quick orientation
```text
data/
├── historical_data.py     # HistoricalData class (klines, trades, agg-trades)
├── _internal/
│   ├── binance_file_to_polars.py   # Download & parse Binance CSV/ZIP files
│   └── generic_endpoints.py        # DB/API query helpers
├── bars/
│   └── standard_bars.py            # Bar formation (time/volume aggregation)
└── utils/
    ├── compute_data_bars.py         # Public bar-formation entry point
    ├── splits.py                    # Train/val/test split helpers
    └── random_slice.py              # Random window slicing utility
```

## Gotchas / things to know

- `HistoricalData._get_data_for_test()` reads from `datasets/klines_2h_2020_2025.csv`; this file must exist relative to the working directory when `LOOP_ENV=test`.
- `get_binance_file()` normalises millisecond timestamps to seconds automatically if the value exceeds `10^13`.
- The `auth_token` parameter is forwarded to `generic_endpoints` for authenticated database queries; leave `None` for public Binance file access.
- `split_data_to_prep_output` expects all splits to share the same column schema.
