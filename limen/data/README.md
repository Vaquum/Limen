# `limen.data`

> Fetch raw market data, form optional bars, and hand experiment-ready splits to the rest of Limen.

## Canonical docs

- [Historical Data](../../docs/Historical-Data.md)
- [Data Bars](../../docs/Data-Bars.md)

## What this package owns

Owns raw data access, optional threshold-bar formation, train/validation/test splitting, and the helpers that turn raw frames into the `data_dict` schema used by model code.
Does **not** own indicators, higher-level features, manifests, or model training.

## Key entry points

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `HistoricalData` | You need spot or futures klines, trades, or agg-trades | The main public class exported by `limen.data` |
| `compute_data_bars()` | You want to aggregate kline rows into threshold bars before feature engineering | Used by manifests through `set_bar_formation()` |
| `split_sequential()` | You need ordered train/validation/test windows | Used by manifest-driven prep |
| `split_data_to_prep_output()` | You need the standard `data_dict` structure | Converts split frames into model-ready keys like `x_train` and `y_test` |

## Adjacent modules

- `limen.experiment` consumes this package through manifests and the Universal Experiment Loop.
- `limen.indicators` and `limen.features` run after data retrieval and optional bar formation.
- `limen.utils.data_dict_to_numpy` is commonly used one layer downstream inside model functions.

## Quick orientation

```text
data/
├── historical_data.py              # HistoricalData class
├── _internal/
│   ├── binance_file_to_polars.py   # Binance file download and parsing
│   └── generic_endpoints.py        # ClickHouse-backed query helpers
├── bars/
│   └── standard_bars.py            # Threshold bar implementation
└── utils/
    ├── compute_data_bars.py        # Public bar-formation entry point
    ├── splits.py                   # Train/validation/test split helpers
    └── random_slice.py             # Random window slicing helper
```

## Things to know

- `HistoricalData` is stateful. Each `get_*` call mutates `self.data` and `self.data_columns`.
- `get_binance_file()` normalizes millisecond timestamps automatically when the source file stores them as large integers.
- `auth_token` is forwarded to the ClickHouse-backed endpoint helpers. Leave it `None` when using direct Binance file access.
- Test-mode retrieval depends on local dataset fixtures when `LOOP_ENV=test`.

## Read next

- [Historical Data](../../docs/Historical-Data.md)
- [Data Bars](../../docs/Data-Bars.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
