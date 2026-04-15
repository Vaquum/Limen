# Data Bars

Limen currently supports threshold-based bar formation over existing kline data. This is an optional preprocessing step inside a manifest: you start from regular time-based klines, then aggregate consecutive rows until a volume, trade-count, or liquidity threshold is reached.

Use bars when time-based candles are not the best surface for the behavior you want to study. Skip them when fixed-interval klines already match the rhythm of the research problem.

## Current Scope

The implemented bar surface today is:

- `volume_bars`
- `trade_bars`
- `liquidity_bars`

Limen does not currently expose imbalance bars, run bars, or tick bars as a supported public surface.

## When Bars Help

Bars usually help when you want each row to represent a more comparable amount of market activity instead of a fixed amount of clock time.

Typical good fits:

- high-volatility periods where fixed-time candles contain very uneven activity
- strategies that care more about activity intensity than about wall-clock spacing
- research where volume or liquidity concentration matters more than elapsed time

Typical poor fits:

- experiments where explicit time-of-day structure matters
- workflows that depend on regular calendar spacing
- cases where the added aggregation step makes the experiment harder to interpret than the base klines

## Shared Output Schema

All supported bar functions return a `pl.DataFrame` with this shared schema:

| Column | Meaning |
|---|---|
| `datetime` | start time of the aggregated bar |
| `open`, `high`, `low`, `close` | OHLC values of the aggregated bar |
| `volume` | cumulative volume inside the bar |
| `no_of_trades` | cumulative trade count inside the bar |
| `liquidity_sum` | cumulative liquidity inside the bar |
| `maker_ratio` | trade-count-weighted maker ratio |
| `maker_volume` | cumulative maker volume |
| `maker_liquidity` | cumulative maker liquidity |
| `mean` | trade-count-weighted mean price |
| `bar_count` | number of source klines merged into the bar |
| `base_interval` | source kline interval in seconds |

Your source dataframe must already contain the columns needed to compute the chosen bar type. In practice that means using kline-style input with fields such as `volume`, `no_of_trades`, and `liquidity_sum`.

## Supported Functions

### `volume_bars(data, volume_threshold)`

Aggregate rows until cumulative volume reaches `volume_threshold`.

### `trade_bars(data, trade_threshold)`

Aggregate rows until cumulative trade count reaches `trade_threshold`.

### `liquidity_bars(data, liquidity_threshold)`

Aggregate rows until cumulative liquidity reaches `liquidity_threshold`.

## Manifest Usage

Bar formation is configured through `Manifest.set_bar_formation()` and is applied separately inside each split. That matters because Limen's manifest pipeline is split-first by design.

```python
from limen.data import HistoricalData
from limen.data.utils import compute_data_bars
from limen.experiment import Manifest

def params():
    return {
        'bar_type': ['base', 'volume', 'trade'],
        'volume_threshold': [50_000, 100_000],
        'trade_threshold': [2_000, 5_000],
    }

def manifest():
    return (
        Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
        )
        .set_test_data_source(
            method=HistoricalData.get_any_file,
            params={
                'file_path_or_url': HistoricalData.DEFAULT_TEST_FILE_URL,
                'n_rows': 5000,
            },
        )
        .set_bar_formation(
            compute_data_bars,
            bar_type='bar_type',
            volume_threshold='volume_threshold',
            trade_threshold='trade_threshold',
        )
        .set_required_bar_columns([
            'datetime',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'no_of_trades',
            'liquidity_sum',
        ])
    )
```

Two important details:

- `bar_type` must be present in `round_params` if you want the bar step to switch between bar modes.
- `set_required_bar_columns()` is an assertion layer. It verifies that the bar step still leaves the downstream columns your experiment needs.

## `compute_data_bars()`

`limen.data.utils.compute_data_bars()` is Limen's convenience router for manifest-driven bar selection.

It currently supports these `bar_type` values:

- `base`
- `trade`
- `volume`
- `liquidity`

`base` returns the input data unchanged. The other values dispatch to the corresponding threshold-bar function and require the matching threshold parameter.

## How Bars Fit Into The Manifest Pipeline

Inside a manifest-driven experiment, the order is:

1. fetch raw input data
2. optionally apply a pre-split selector
3. split into train, validation, and test
4. apply bar formation inside each split
5. run indicators, features, targets, and scaling on the resulting bars

This keeps train-only fitting and test-only evaluation aligned with the actual post-bar data seen by each fold.

## Read Next

- Continue to [Single File Decoder](Single-File-Decoder.md) if you are deciding how to package the experiment.
- Continue to [Experiment Manifest](Experiment-Manifest.md) for the full declarative pipeline that bar formation plugs into.
- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) to run the resulting experiment.
