# Conserved Flux Renormalization

Conserved Flux Renormalization (CFR) is a trade-data feature that summarizes how traded value and trade-size entropy behave across multiple time scales inside each bar.

In practical Limen terms, CFR turns raw trades into:

- a kline-like frame
- plus six additional columns that describe multi-scale flow stability and entropy behavior

The [Python project](examples/python-research/README.md) runs this CFR diagnostic in its `bars` scenario: 90 recorded seconds of BTCUSDT trades produce three 30-second bars and their profile. The [Compiled project](examples/compiled-research/README.md) separately exercises threshold bars on hourly observations; it does not run this trade-data diagnostic. Three diagnostic bars do not establish a trained CFR strategy.

## Prerequisites

- trade-level `datetime`, `price`, and `quantity` columns
- enough events for the chosen bar interval and scale set
- a downstream use for the formed bars and six CFR diagnostic columns

## Input And Output

### Input

`conserved_flux_renormalization()` expects a trade dataframe with at least:

- `datetime`
- `price`
- `quantity`

### Output

It returns a kline-style dataframe containing:

- `datetime`
- `open`, `high`, `low`, `close`
- `volume`
- `value_sum`
- `vwap`
- `flux_rel_std_mean`
- `flux_rel_std_var`
- `entropy_mean`
- `entropy_var`
- `Δflux_rms`
- `Δentropy_rms`

## Usage

Use the complete project entry points and recorded input bundle. Both calls below execute the same shared helper; the input loader checks its recorded trade-file digest before forming the profile.

### Python Research

<!-- example: cfr-python-profile source=docs/examples/recorded_inputs.py region=trade_profile scenario=bars interface=python role=executed -->
```python
def trade_profile(data_dir, output):
    historical = HistoricalData()
    historical.get_any_file(str(checked_path(data_dir, TRADES)), has_header=False,
                            columns=['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker', '_null'])
    trades = historical.data
    if not isinstance(trades, pl.DataFrame):
        raise ValueError('The recorded trade source did not produce a dataframe')
    profile = conserved_flux_renormalization(trades, kline_interval='30s', base_window_s=5, levels=3)
    profile.write_parquet(output / 'trade-profile.parquet')
    rows = [{**row, 'datetime': row['datetime'].isoformat()} for row in profile.to_dicts()]
    return {'trade_rows': trades.height, 'bars': rows, 'input_sha256': HASHES[TRADES], 'population': '90 recorded seconds for flow diagnostics; no compiled training population is asserted'}
```

### Compiled Research

The recorded trade slice supplies three CFR diagnostic bars. The Python example above measures that transformation; it supplies no supervised training population. The Compiled project's [bar declaration](examples/compiled-research/bars.yaml) separately exercises volume, trade and liquidity bars on the recorded hourly source.

The assertions independently reconstruct each bar's OHLC, volume, traded value and VWAP from the original trades. They check that the complete population is accounted for and all six diagnostics are finite and nonnegative. They do not turn the observed deviation measures into predictive or acceptance thresholds.

The current API is:

<!-- reference: cfr-public-api source=limen/features/conserved_flux_renormalization.py region=signature:conserved_flux_renormalization role=signature -->
```python
def conserved_flux_renormalization(trades_df: pl.DataFrame, *, kline_interval: str='1h', base_window_s: int=60, levels: int=6) -> pl.DataFrame:
    ...
```

### Parameters

| Parameter | Meaning |
|---|---|
| `trades_df` | raw trade dataframe |
| `kline_interval` | bar interval for the output frame, such as `'1h'` |
| `base_window_s` | smallest internal coarse-graining window in seconds |
| `levels` | number of renormalization levels to compute |

## What The CFR Columns Mean

### `flux_rel_std_mean`

Mean relative variability of traded-value flux across the renormalization levels.

### `flux_rel_std_var`

Variance of that flux-variability ladder across levels.

### `entropy_mean`

Mean trade-size entropy across the renormalization levels.

### `entropy_var`

Variance of that entropy ladder across levels.

### `Δflux_rms`

Root-mean-square deviation from the flat flux-variability reference ladder.

Higher values mean one or more scales dominate the traded-value flow instead of the flow looking scale-stable.

### `Δentropy_rms`

Root-mean-square deviation from the one-bit-per-octave entropy reference ladder.

Higher values mean trade-size diversity changes unevenly across scales.

## When CFR fits

Suitable fits:

- anomaly detection in trade flow
- regime features for microstructure-heavy experiments
- identifying bars where traded value is concentrated on a subset of scales
- identifying bars where trade-size diversity behaves unusually

Poor fits:

- workflows where only OHLCV-level information is available
- experiments that never touch raw trade data
- cases where lower-dimensional activity features already capture the target behavior

## The Intuition

CFR aggregates the recorded trades over progressively longer windows. The worked profile uses three levels within each 30-second bar:

| Level | Recorded example window |
|---|---|
| 0 | 5 seconds |
| 1 | 10 seconds |
| 2 | 20 seconds |

At each scale it measures two things:

- how variable the traded-value flux is
- how diverse the trade sizes are

Instead of returning the entire multi-scale ladder, CFR compresses it into a small set of summary features that can be used directly in downstream research.

## Interpreting The Deviation Metrics

The anomaly-style outputs are:

- `Δflux_rms`
- `Δentropy_rms`

They measure how far the observed scale ladder is from an idealized reference shape.

As a rough intuition:

- high `Δflux_rms` suggests bursty or scale-concentrated flow
- high `Δentropy_rms` suggests an uneven or patchy size mix across scales
- when both are elevated, the bar is more likely to be structurally unusual

These are interpretation aids, not hard universal thresholds.

## Read Next

- Continue to [Features](Features.md) for the broader feature layer CFR belongs to.
- Continue to [Historical Data](Historical-Data.md) for the trade-data surfaces CFR expects upstream.
