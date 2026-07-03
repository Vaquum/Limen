# `limen.data.bars`

> Threshold-bar constructors that resample kline rows into information-driven bars.

## Canonical docs

- [Data Bars](../../../docs/Data-Bars.md)

## What this package owns

Owns the standard threshold-bar implementations that aggregate raw kline rows once a running quantity crosses a threshold.
Does **not** own raw data retrieval (owned by `limen.data.HistoricalData`) or the public `compute_data_bars()` entry point (owned by `limen.data.utils`).

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `volume_bars` | Emit a bar every fixed traded volume | Threshold on cumulative base volume |
| `trade_bars` | Emit a bar every fixed trade count | Threshold on cumulative trade count |
| `liquidity_bars` | Emit a bar every fixed quote-value turnover | Threshold on cumulative quote volume |

## Adjacent modules

- `limen.data.utils.compute_data_bars` is the public wrapper manifests call; it dispatches to these functions.
- `limen.data.HistoricalData` supplies the raw kline frames these bars aggregate.

## Quick orientation

```text
bars/
└── standard_bars.py   # volume_bars, trade_bars, liquidity_bars
```

## Things to know

- These are the low-level bar builders. Manifests should go through `compute_data_bars()` rather than importing them directly.
- Each function consumes and returns a `pl.DataFrame`; bars are formed in row order, so the input must already be time-sorted.
- A trailing partial bar (below the threshold at end of data) is not emitted.

## Read next

- [Data Bars](../../../docs/Data-Bars.md)
- [Historical Data](../../../docs/Historical-Data.md)
