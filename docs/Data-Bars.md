# Data bars

Limen currently supports threshold-based bar formation over existing kline data. This optional manifest preprocessing step starts from regular time-based klines, then aggregates consecutive rows until a volume, trade-count, or liquidity threshold is reached.

Use bars when time-based candles do not match the studied behavior. Skip bars when fixed-interval klines already match the research rhythm.

## Prerequisites

- a kline-like Polars frame with the columns required by the chosen bar type
- `vaquum-limen[data]` when the source frame comes from `HistoricalData`
- a declared research reason for replacing fixed-time rows with activity-threshold rows

## Current scope

The implemented bar surface today is:

- `volume_bars`
- `trade_bars`
- `liquidity_bars`

Limen does not currently expose imbalance bars, run bars, or tick bars as a supported public surface.

## When bars help

Bars help when each row should represent a comparable amount of market activity instead of a fixed amount of clock time.

Suitable fits:

- high-volatility periods where fixed-time candles contain very uneven activity
- strategies that depend more on activity intensity than wall-clock spacing
- research where volume or liquidity concentration matters more than elapsed time

Unsuitable fits:

- experiments where explicit time-of-day structure matters
- workflows that depend on regular calendar spacing
- cases where the added aggregation step makes the experiment harder to interpret than the base klines

## Shared output schema

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

The source dataframe must already contain the columns needed to compute the chosen bar type. Kline-style input must include fields such as `volume`, `no_of_trades`, and `liquidity_sum`.

## Supported functions

### `volume_bars(data, volume_threshold)`

Aggregate rows until cumulative volume reaches `volume_threshold`.

### `trade_bars(data, trade_threshold)`

Aggregate rows until cumulative trade count reaches `trade_threshold`.

### `liquidity_bars(data, liquidity_threshold)`

Aggregate rows until cumulative liquidity reaches `liquidity_threshold`.

## Form bars in either project

The [Python project's](examples/python-research/README.md) `bars` scenario forms each of the three supported bar types from 384 recorded hourly rows. It derives a diagnostic threshold from that bounded population, writes the complete bars, and records the omitted incomplete tail. This is a bar-formation diagnostic; a model comparison must still choose fitting and assessment populations separately.

<!-- example: bars-python-formation source=docs/examples/python-research/scenarios.py region=bars scenario=bars interface=python role=executed -->
```python
def bars(data_dir, output):
    raw = recorded(data_dir, 'spot_1h_20240101_20241231.parquet', 384)
    results = []
    for kind, parameter, column in [('volume', 'volume_threshold', 'volume'), ('trade', 'trade_threshold', 'no_of_trades'), ('liquidity', 'liquidity_threshold', 'liquidity_sum')]:
        median = raw[column].median()
        if not isinstance(median, (int, float)):
            raise ValueError('Bar formation requires a numeric recorded threshold source')
        threshold = float(median) * 4
        if kind == 'trade':
            threshold = int(threshold)
        formed = compute_data_bars(raw, bar_type=kind, **{parameter: threshold})
        formed.write_parquet(output / (kind + '.parquet'))
        results.append({'kind': kind, 'threshold': threshold, 'rows': formed.height, 'source_rows_in_complete_bars': int(formed['bar_count'].sum()), 'volume': float(formed['volume'].sum()), 'first_open': formed['open'][0], 'last_close': formed['close'][-1]})
    return {'source_rows': raw.height, 'formed': results, 'incomplete_tail': 'omitted; do not treat it as a completed bar', 'trade_profile': trade_profile(data_dir, output)}
```

The [Compiled project's](examples/compiled-research/README.md) `bars` scenario runs three declared parameter choices. Its Manifest forms bars separately after each chronological split:

<!-- example: bars-compiled-formation source=docs/examples/compiled-research/bars.yaml region=bar_formation scenario=bars interface=compiled role=executed -->
```yaml
    bar_formation:
      func: limen.data.utils.compute_data_bars
      params:
        bar_type: '{bar_type}'
        volume_threshold: '{volume_threshold}'
        trade_threshold: '{trade_threshold}'
        liquidity_threshold: '{liquidity_threshold}'
```

The complete configuration declares `bar_type` as volume, trade or liquidity, with volume threshold `1000.0`, trade threshold `100000`, and liquidity threshold `10000000.0`. The scenario tests verify all three completed rounds; the Python diagnostic independently checks complete-bar boundaries and conserved volume. Both also process the same short recorded trade slice for [CFR](Conserved-Flux-Renormalization.md), without claiming it supplies a full ML population.

`Manifest.set_bar_formation()` attaches the bar step. `set_required_bar_columns()` asserts that the resulting frame retains the columns required downstream. Parameter references in the bar declaration must resolve from the current round.

## `compute_data_bars()`

`limen.data.utils.compute_data_bars()` is Limen's convenience router for manifest-driven bar selection.

It currently supports these `bar_type` values:

- `base`
- `trade`
- `volume`
- `liquidity`

`base` returns the input data unchanged. The other values dispatch to the corresponding threshold-bar function and require the matching threshold parameter. Unknown `bar_type` values raise `ValueError`; they are not silently treated as `base`.

## How bars fit into the manifest pipeline

Inside a manifest-driven experiment, the order is:

1. fetch raw input data
2. optionally apply a pre-split selector
3. split into train, validation, and test
4. apply bar formation inside each split
5. run indicators, features, targets, and scaling on the resulting bars

This keeps train-only fitting and test-only evaluation aligned with the actual post-bar data seen by each fold.

## Read next

- Continue to [Single-File Decoder](Single-File-Decoder.md) for experiment packaging.
- Continue to [Experiment Manifest](Experiment-Manifest.md) for the full declarative pipeline that bar formation plugs into.
- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) to run the resulting experiment.
