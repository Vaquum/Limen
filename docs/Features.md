# Features

Features are more complex than Indicators, and often involve further refining Indicators or combining several Indicators into a single Feature. 

## `loop.features`

### `conserved_flux_renormalization`

Compute multi-scale, conserved-flux features and their deviation scores for each k-line—turning raw trade ticks into a six-value fingerprint that flags hours where the dollar flow or trade-size entropy breaks scale-invariant behaviour.

Read more in: [Conserved Flux Renormalization](Conserved-Flux-Renormalization.md)

#### Args

| Parameter        | Type            | Description                         |
|------------------|-----------------|-------------------------------------|
| `trades_df`      | `pl.DataFrame`  | The trades DataFrame.               |
| `kline_interval` | `str`           | The kline interval.                 |
| `base_window_s`  | `int`           | The base window size.               |
| `levels`         | `int`           | The number of levels to compute.    |

#### Returns

`pl.DataFrame`: A klines DataFrame with the CFR features

| Column               | Type        | Brief description (1-line)                                           |
|----------------------|-------------|-----------------------------------------------------------------------|
| `datetime`           | datetime[ms] | Start timestamp of the k-line window (bucket-aligned).               |
| `open`               | float64     | First trade price in the window.                                     |
| `high`               | float64     | Highest trade price in the window.                                   |
| `low`                | float64     | Lowest trade price in the window.                                    |
| `close`              | float64     | Last trade price in the window.                                      |
| `volume`             | float64     | Total BTC traded in the window.                                      |
| `value_sum`          | float64     | Dollar notional traded ∑ (price × quantity).                          |
| `vwap`               | float64     | Volume-weighted average price (value_sum / volume).                  |
| `flux_rel_std_mean`  | float64     | Mean relative σ/μ of value-flux across the 6 nested scales.           |
| `flux_rel_std_var`   | float64     | Variance of that σ/μ ladder (how one scale dominates).                |
| `entropy_mean`       | float64     | Mean Shannon entropy (bits) of trade-size mix across scales.          |
| `entropy_var`        | float64     | Variance of the entropy ladder (patchiness of size mix).              |
| `Δflux_rms`          | float64     | RMS gap from an ideal *flat* flux ladder (0 = scale-neutral flow).    |
| `Δentropy_rms`       | float64     | RMS gap from a perfect 1-bit-per-octave entropy drop.                 |