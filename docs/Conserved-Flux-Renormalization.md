# Conserved Flux Renormalization

Conserved Flux Renormalization (CFR) is a trade-data feature that summarizes how traded value and trade-size entropy behave across multiple time scales inside each bar.

In practical Limen terms, CFR turns raw trades into:

- a kline-like frame
- plus six additional columns that describe multi-scale flow stability and entropy behavior

CFR provides a compact diagnostic for scale-stable or anomalous trade flow inside each bar.

## Prerequisites

- trade-level `datetime`, `price`, and `quantity` columns
- enough events for the chosen bar interval and scale set
- a downstream plan for the synthetic kline and six CFR diagnostic columns

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

```python
from limen.features.conserved_flux_renormalization import conserved_flux_renormalization

cfr_df = conserved_flux_renormalization(
    trades_df,
    kline_interval='1h',
    base_window_s=60,
    levels=6,
)
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

The conserved-flux idea treats traded value as conserved "stuff." Renormalization then asks:

CFR measures how that flow changes under repeated time-scale aggregation.

The function starts from a base window such as 60 seconds and repeatedly coarse-grains the trade stream:

```text
60s -> 120s -> 240s -> 480s
```

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
