# Indicator performance

Limen 5.29.1 reduces the calculation cost of the existing Hilbert, MAMA, CCI, three candlestick and five regression indicators. Public indicator signatures, defaults, validation, output order/dtypes and warmup/null/NaN behavior are preserved.

## Reproduce

```bash
python -m tests.benchmark_indicators \
  --input /path/to/btcusdt_15m_kline_20200101_to_20261003.parquet \
  --reference fe27656c \
  --output /tmp/limen-indicator-measurements
python -m pytest tests/test_indicator_performance.py
```

The input is the recorded BTCUSDT 15-minute snapshot from `vaquum/binance_btcusdt_15m_klines`, with its first 20,000 and 83,136 rows used unchanged for both implementations. The benchmark fails on unequal outputs/schemas or a latency regression. It reuses measured baseline results only when the input, reference, runtime versions, hardware, repeat count and thread settings match; candidate code changes invalidate candidate results.

- Baseline: `fe27656cfa6973185c5e789080fbf61e169e0d5a` (5.29.0).
- Measured candidate: `ee3c99b57a6368802bb6953ca203878c067558aa` (5.29.1); implementation SHA-256 `92ac33c7239c7ac761789b90c2446aea4bb3a329756e8e2c7aabd6e5e9a159d4`.
- Date: 2026-10-17; processor: Apple M1 Max; platform: macOS-26.6.2-arm64-arm-64bit.
- Runtime: Python 3.12.12, NumPy 2.5.1, Polars 1.42.1.
- Threads: `POLARS_MAX_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `BLIS_NUM_THREADS` are all `1`.
- Snapshot SHA-256: `04cdc6b715daa4ad32b2594fcb33a2c2f3aea100eb17e131182391880a46c443`.
- 20,000-row OHLCV SHA-256: `e6d78685cf5eb2dbfa6936f13ddec32268ab5147fc2edc86572adc1cbe03a70e`.
- 83,136-row OHLCV SHA-256: `dc9b556fb8d2ea5a7ac1988f40c069bcaf57f8b063a9694273358222f20dd3a0`.

OHLCV digests hash NumPy Float64 bytes in `open, high, low, close, volume` column order. [Raw evidence](https://github.com/Vaquum/Limen/blob/codex/930-indicator-performance/tests/fixtures/indicator_equivalence_benchmark.json) contains every timing sample, import duration, schema and memory measurement.

## Calculation latency

Milliseconds, before / after; median of three untraced calls in a fresh process per indicator, size and implementation. Conversion, calculation, allocation and output-column construction are timed together.

| Indicator | 20,000 rows | 83,136 rows | Speedup at 83,136 |
|---|---:|---:|---:|
| `ht_sine` | 469.70 / 55.08 | 1964.22 / 234.08 | 8.4× |
| `ht_trendmode` | 321.56 / 72.76 | 1364.49 / 383.39 | 3.6× |
| `ht_dcphase` | 204.13 / 59.34 | 744.04 / 362.69 | 2.1× |
| `ht_phasor` | 133.65 / 32.35 | 773.12 / 112.36 | 6.9× |
| `ht_trendline` | 110.83 / 46.95 | 466.77 / 199.14 | 2.3× |
| `ht_dcperiod` | 62.94 / 25.54 | 271.47 / 104.78 | 2.6× |
| `mama` | 149.55 / 35.05 | 2482.94 / 243.24 | 10.2× |
| `cci` | 64.39 / 0.50 | 617.08 / 1.73 | 357.4× |
| `cdladvancedblock` | 168.14 / 1.26 | 2054.59 / 5.09 | 403.9× |
| `cdl3whitesoldiers` | 124.69 / 1.17 | 474.63 / 6.42 | 74.0× |
| `cdlstalledpattern` | 80.87 / 0.87 | 319.62 / 4.10 | 78.0× |
| `linearreg` | 77.15 / 0.42 | 225.90 / 1.20 | 188.5× |
| `linearreg_angle` | 79.02 / 2.13 | 213.18 / 12.81 | 16.6× |
| `linearreg_intercept` | 83.27 / 0.32 | 220.38 / 1.31 | 167.7× |
| `linearreg_slope` | 69.80 / 0.29 | 521.29 / 0.97 | 534.8× |
| `tsf` | 52.01 / 0.89 | 202.51 / 1.82 | 111.2× |

## Peak memory

MiB, before / after. RSS is the isolated process high-water mark after the untraced calls, including imports and input data. Traced allocation peaks come from a separate warm call and cover Python/NumPy allocations; Polars native allocations are instead reflected in RSS. Tracing time is excluded from latency measurements.

| Indicator | Traced, 20,000 | Traced, 83,136 | RSS, 20,000 | RSS, 83,136 |
|---|---:|---:|---:|---:|
| `ht_sine` | 0.47 / 0.92 | 1.91 / 3.81 | 85.41 / 83.94 | 97.73 / 104.27 |
| `ht_trendmode` | 0.16 / 0.77 | 0.64 / 3.18 | 84.31 / 83.81 | 94.59 / 99.64 |
| `ht_dcphase` | 0.16 / 0.77 | 0.64 / 3.18 | 84.05 / 84.19 | 94.56 / 99.72 |
| `ht_phasor` | 0.47 / 0.92 | 1.91 / 3.81 | 84.22 / 83.53 | 97.34 / 101.11 |
| `ht_trendline` | 0.16 / 0.77 | 0.64 / 3.18 | 84.28 / 83.67 | 95.12 / 98.34 |
| `ht_dcperiod` | 0.16 / 0.77 | 0.64 / 3.18 | 84.09 / 84.03 | 94.67 / 98.36 |
| `mama` | 0.47 / 0.92 | 1.91 / 3.81 | 83.84 / 83.67 | 96.34 / 101.59 |
| `cci` | 0.46 / 1.07 | 1.27 / 4.05 | 83.77 / 84.08 | 97.16 / 98.55 |
| `cdladvancedblock` | 0.09 / 2.34 | 0.33 / 9.68 | 84.03 / 83.67 | 94.52 / 105.86 |
| `cdl3whitesoldiers` | 0.08 / 1.95 | 0.33 / 8.10 | 84.08 / 83.52 | 94.67 / 110.77 |
| `cdlstalledpattern` | 0.08 / 1.78 | 0.33 / 6.83 | 83.86 / 83.72 | 94.70 / 104.86 |
| `linearreg` | 0.16 / 0.92 | 0.64 / 3.81 | 84.00 / 84.14 | 94.95 / 98.81 |
| `linearreg_angle` | 0.16 / 0.92 | 0.64 / 3.81 | 83.81 / 83.92 | 94.75 / 100.48 |
| `linearreg_intercept` | 0.16 / 0.92 | 0.64 / 3.81 | 84.00 / 83.53 | 94.92 / 99.56 |
| `linearreg_slope` | 0.16 / 0.92 | 0.64 / 3.81 | 84.16 / 83.95 | 94.94 / 97.73 |
| `tsf` | 0.16 / 0.92 | 0.64 / 3.81 | 84.02 / 84.22 | 94.39 / 100.47 |

The vector kernels use O(rows) working memory, never a rows-by-period matrix. Recursive kernels copy prices into Python doubles once per call. The measured latency includes these costs; the table records the additional working memory explicitly.

## First calls and imports

There is no JIT or runtime native compilation. First calls include lazy phase-weight construction; subsequent calls reuse at most 50 small weight tuples. Imports and input loading finish before calculation timing starts. Each side uses a fresh interpreter, while filesystem caches and unrelated desktop workloads remain uncontrolled.

| Indicator | First call, 20,000 ms | First call, 83,136 ms |
|---|---:|---:|
| `ht_sine` | 432.75 / 53.58 | 2135.74 / 344.96 |
| `ht_trendmode` | 374.19 / 76.39 | 1244.85 / 357.93 |
| `ht_dcphase` | 186.71 / 62.62 | 737.48 / 303.09 |
| `ht_phasor` | 143.93 / 35.41 | 664.08 / 137.09 |
| `ht_trendline` | 136.93 / 43.77 | 490.20 / 188.71 |
| `ht_dcperiod` | 63.91 / 25.30 | 301.36 / 104.89 |
| `mama` | 152.58 / 37.07 | 1276.52 / 176.19 |
| `cci` | 64.99 / 1.00 | 810.01 / 5.08 |
| `cdladvancedblock` | 162.48 / 2.56 | 1374.76 / 7.17 |
| `cdl3whitesoldiers` | 117.63 / 2.40 | 520.13 / 10.41 |
| `cdlstalledpattern` | 81.90 / 1.63 | 316.45 / 6.33 |
| `linearreg` | 81.32 / 0.77 | 228.83 / 2.42 |
| `linearreg_angle` | 85.77 / 2.43 | 212.78 / 9.42 |
| `linearreg_intercept` | 89.25 / 0.76 | 217.81 / 2.24 |
| `linearreg_slope` | 56.21 / 0.70 | 212.11 / 2.07 |
| `tsf` | 54.29 / 2.09 | 200.61 / 2.37 |

Median indicator-package import time across the isolated processes was 28.25 / 14.93 ms. No dependency or import-time calculation was added. Raw evidence includes dependency imports separately; these import numbers are observations under filesystem caching, not a claimed import optimization.

## Implementation and equivalence

- Paired Hilbert/MAMA outputs share one batch calculation; their original implementations ran it twice.
- Hilbert recurrence order is unchanged. Python doubles and indexed state remove NumPy scalar/dictionary dispatch, while cached scalar trigonometric weights preserve the original `math.sin`/`math.cos` values.
- CCI and regression accumulate each independent window in the original chronological order using NumPy slices.
- Candlestick kernels seed the original moving-total recurrence and run its increments through sequential `cumsum`; this preserves rounding, ties and NaN propagation that a new rolling mean could change.
- Initial screening covered every candlestick and regression indicator. Advance Block, Three White Soldiers and Stalled Pattern were the three largest candlestick costs; all five regression/forecast functions were optimized. The raw evidence retains every initial screening time.

The 409 contract cases compare the candidate with the original source on the same runtime, with exact numerical equality and no relaxed tolerance. Both eager and lazy frames are checked, including empty/short histories, warmup boundaries, real-derived flat/tied prices, Float32/integer inputs, every OHLC missing-value position, alternate price columns, MAMA limits and periods 2, 7, 30 and 100,000. The longest input contains 100,002 recorded bars.

The original Hilbert implementation produces different flat-price phase results across the macOS and Linux runtimes, where tiny floating-point differences reach zero-sensitive phase branches. Therefore, a macOS-only output snapshot is not a portable oracle. `indicator_equivalence_reference.zip` contains the original source with only its internal import namespace relocated; committed SHA-256 values verify every original file. Tests compare both implementations within each runtime instead of accepting a numerical discrepancy.

CI runs exact equivalence and a three-sample relative latency check for every changed indicator on 6,000 recorded bars. CCI must take less than half the baseline time; each remaining calculation must take less than 90%. The standalone 20k/83k benchmark additionally rejects any measured latency regression.
