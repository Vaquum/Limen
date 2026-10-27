# `limen.data.bars`

> Fixed activity bars over klines and adaptive directional-volume bars over trade points.

## Canonical docs

- [Data Bars](../../../docs/Data-Bars.md)

## In the research projects

Both projects run `bars`. The [Python project](../../../docs/examples/python-research/README.md) checks complete-bar boundaries and conserved volume directly; [Compiled project](../../../docs/examples/compiled-research/README.md) applies the same formation functions separately after each chronological split.

## What this package owns

Owns fixed activity thresholds over kline rows and adaptive imbalance/run thresholds over individual trade points.
Does **not** own raw data retrieval (owned by `limen.data.HistoricalData`) or the public `compute_data_bars()` entry point (owned by `limen.data.utils`).

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `volume_bars` | Emit a bar every fixed traded volume | Threshold on cumulative base volume |
| `trade_bars` | Emit a bar every fixed trade count | Threshold on cumulative trade count |
| `liquidity_bars` | Emit a bar every fixed quote-value turnover | Threshold on cumulative quote volume |
| `imbalance_bars`, `run_bars` | Adaptive directional-volume sampling | Individual native trade points; see [algorithm](../../../docs/Data-Bars.md#adaptive-trade-point-bars) |

## Adjacent modules

- `limen.data.utils.compute_data_bars` is the public wrapper manifests call; it dispatches to these functions.
- `limen.data.HistoricalData` supplies the raw kline frames these bars aggregate.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `standard_bars.py` | volume_bars, trade_bars, liquidity_bars |

<!-- reference: package-data-bars-layout source=limen/data/bars/README.md region=layout:package role=layout -->
```text
bars/
├── README.md
├── __init__.py
├── _bar_aggregation.py
├── adaptive_bars.py
└── standard_bars.py
```

## Things to know

- These are the low-level bar builders. Manifests should go through `compute_data_bars()` rather than importing them directly.
- Each function consumes and returns a `pl.DataFrame`; bars are formed in row order, so the input must already be time-sorted.
- A trailing partial bar (below the threshold at end of data) is not emitted.

## Read next

- [Data Bars](../../../docs/Data-Bars.md)
- [Historical Data](../../../docs/Historical-Data.md)
