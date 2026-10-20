# Indicator performance evidence

Changes to an indicator must preserve the values and clocks that a research decision uses. The [Python project](../examples/python-research/README.md) owns a decimal momentum expression; the [Compiled project](../examples/compiled-research/README.md) declares the package's percent-valued rate of change. Re-run their selection and inference assertions when changing shared preparation. Calculation timings answer a separate question from research results.

## Prerequisites

Use a Limen development checkout with its recorded fixtures and test dependencies. The historical measurements below identify their source revisions, data, interpreter and thread settings. Keep those identities when interpreting a replay; a timing measured on another host is another observation.

The [indicator reference](../Indicators.md) retains function names, default output columns and project usage. This appendix retains the implementation measurements and reproduction procedures.

## Performance measurements

Limen 5.29.2 reduces the calculation cost of the existing Hilbert, MAMA, CCI, three candlestick and five regression indicators. Public indicator signatures, defaults, validation, output order/dtypes and warmup/null/NaN behavior are preserved.

### Reproduce

```bash
python -m tests.benchmark_indicators \
  --input /path/to/btcusdt_15m_kline_20200101_to_20261003.parquet \
  --reference fe27656c \
  --output /tmp/limen-indicator-measurements
python -m pytest tests/test_indicator_performance.py
```

The input is the recorded BTCUSDT 15-minute snapshot from `vaquum/binance_btcusdt_15m_klines`, with its first 20,000 and 83,136 rows used unchanged for both implementations. The benchmark fails on unequal outputs/schemas or a latency regression. It reuses measured baseline results only when the input, resolved reference commit, runtime versions, hardware, repeat count and thread settings match; candidate code changes invalidate candidate results.

- Baseline: `fe27656cfa6973185c5e789080fbf61e169e0d5a` (5.29.0).
- Measured candidate: `6242252168794ad84b0eeb2e0119033b94556bc7` (5.29.2); implementation SHA-256 `81c82f8c0b5c9cd2a609573f0bce8d3f5e83a2fa7ff2738aba709f839b95c248`.
- Date: 2026-10-17; processor: Apple M1 Max; platform: macOS-26.6.2-arm64-arm-64bit.
- Runtime: Python 3.12.12, NumPy 2.5.1, Polars 1.42.1.
- Threads: `POLARS_MAX_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `BLIS_NUM_THREADS` are all `1`.
- Snapshot SHA-256: `04cdc6b715daa4ad32b2594fcb33a2c2f3aea100eb17e131182391880a46c443`.
- 20,000-row OHLCV SHA-256: `e6d78685cf5eb2dbfa6936f13ddec32268ab5147fc2edc86572adc1cbe03a70e`.
- 83,136-row OHLCV SHA-256: `dc9b556fb8d2ea5a7ac1988f40c069bcaf57f8b063a9694273358222f20dd3a0`.

OHLCV digests hash NumPy Float64 bytes in `open, high, low, close, volume` column order. [Raw evidence](https://github.com/Vaquum/Limen/blob/codex/930-indicator-performance/tests/fixtures/indicator_equivalence_benchmark.json) contains every timing sample, import duration, schema and memory measurement.

### Calculation latency

Milliseconds, before / after; median of three untraced calls in a fresh process per indicator, size and implementation. Conversion, calculation, allocation and output-column construction are timed together.

| Indicator | 20,000 rows | 83,136 rows | Speedup at 83,136 |
|---|---:|---:|---:|
| `ht_sine` | 469.70 / 58.62 | 1964.22 / 300.44 | 6.5× |
| `ht_trendmode` | 321.56 / 93.42 | 1364.49 / 388.47 | 3.5× |
| `ht_dcphase` | 204.13 / 60.46 | 744.04 / 248.78 | 3.0× |
| `ht_phasor` | 133.65 / 34.09 | 773.12 / 170.09 | 4.5× |
| `ht_trendline` | 110.83 / 45.84 | 466.77 / 216.22 | 2.2× |
| `ht_dcperiod` | 62.94 / 30.30 | 271.47 / 183.97 | 1.5× |
| `mama` | 149.55 / 36.85 | 2482.94 / 209.74 | 11.8× |
| `cci` | 64.39 / 0.55 | 617.08 / 1.92 | 320.6× |
| `cdladvancedblock` | 168.14 / 1.44 | 2054.59 / 5.35 | 384.4× |
| `cdl3whitesoldiers` | 124.69 / 1.68 | 474.63 / 6.75 | 70.3× |
| `cdlstalledpattern` | 80.87 / 4.61 | 319.62 / 3.46 | 92.5× |
| `linearreg` | 77.15 / 0.56 | 225.90 / 6.37 | 35.5× |
| `linearreg_angle` | 79.02 / 4.48 | 213.18 / 9.68 | 22.0× |
| `linearreg_intercept` | 83.27 / 0.48 | 220.38 / 1.11 | 199.1× |
| `linearreg_slope` | 69.80 / 0.31 | 521.29 / 1.06 | 493.8× |
| `tsf` | 52.01 / 0.57 | 202.51 / 1.47 | 137.9× |

### Peak memory

MiB, before / after. RSS is the isolated process high-water mark after the untraced calls, including imports and input data. Traced allocation peaks come from a separate warm call and cover Python/NumPy allocations; Polars native allocations are instead reflected in RSS. Tracing time is excluded from latency measurements.

| Indicator | Traced, 20,000 | Traced, 83,136 | RSS, 20,000 | RSS, 83,136 |
|---|---:|---:|---:|---:|
| `ht_sine` | 0.47 / 0.92 | 1.91 / 3.81 | 85.41 / 83.70 | 97.73 / 102.47 |
| `ht_trendmode` | 0.16 / 0.77 | 0.64 / 3.18 | 84.31 / 84.09 | 94.59 / 98.67 |
| `ht_dcphase` | 0.16 / 0.77 | 0.64 / 3.18 | 84.05 / 83.88 | 94.56 / 99.66 |
| `ht_phasor` | 0.47 / 0.92 | 1.91 / 3.81 | 84.22 / 83.62 | 97.34 / 100.95 |
| `ht_trendline` | 0.16 / 0.77 | 0.64 / 3.18 | 84.28 / 83.81 | 95.12 / 99.95 |
| `ht_dcperiod` | 0.16 / 0.77 | 0.64 / 3.18 | 84.09 / 83.66 | 94.67 / 98.77 |
| `mama` | 0.47 / 0.92 | 1.91 / 3.81 | 83.84 / 83.59 | 96.34 / 100.08 |
| `cci` | 0.46 / 1.07 | 1.27 / 4.05 | 83.77 / 84.19 | 97.16 / 99.64 |
| `cdladvancedblock` | 0.09 / 2.34 | 0.33 / 9.68 | 84.03 / 83.88 | 94.52 / 109.42 |
| `cdl3whitesoldiers` | 0.08 / 1.95 | 0.33 / 8.10 | 84.08 / 83.98 | 94.67 / 107.80 |
| `cdlstalledpattern` | 0.08 / 1.78 | 0.33 / 6.83 | 83.86 / 84.36 | 94.70 / 102.50 |
| `linearreg` | 0.16 / 0.92 | 0.64 / 3.81 | 84.00 / 83.64 | 94.95 / 101.78 |
| `linearreg_angle` | 0.16 / 0.92 | 0.64 / 3.81 | 83.81 / 83.77 | 94.75 / 100.62 |
| `linearreg_intercept` | 0.16 / 0.92 | 0.64 / 3.81 | 84.00 / 83.66 | 94.92 / 98.84 |
| `linearreg_slope` | 0.16 / 0.92 | 0.64 / 3.81 | 84.16 / 83.61 | 94.94 / 98.03 |
| `tsf` | 0.16 / 0.92 | 0.64 / 3.81 | 84.02 / 83.88 | 94.39 / 101.97 |

The vector kernels use O(rows) working memory, never a rows-by-period matrix. Recursive kernels copy prices into Python doubles once per call. The measured latency includes these costs; the table records the additional working memory explicitly.

### First calls and imports

There is no JIT or runtime native compilation. First calls include lazy phase-weight construction; subsequent calls reuse at most 50 small weight tuples. Imports and input loading finish before calculation timing starts. Each side uses a fresh interpreter, while filesystem caches and unrelated desktop workloads remain uncontrolled.

| Indicator | First call, 20,000 ms | First call, 83,136 ms |
|---|---:|---:|
| `ht_sine` | 432.75 / 57.42 | 2135.74 / 310.57 |
| `ht_trendmode` | 374.19 / 93.14 | 1244.85 / 351.21 |
| `ht_dcphase` | 186.71 / 58.56 | 737.48 / 302.99 |
| `ht_phasor` | 143.93 / 43.65 | 664.08 / 130.78 |
| `ht_trendline` | 136.93 / 54.13 | 490.20 / 284.51 |
| `ht_dcperiod` | 63.91 / 43.97 | 301.36 / 148.84 |
| `mama` | 152.58 / 42.93 | 1276.52 / 166.03 |
| `cci` | 64.99 / 1.09 | 810.01 / 7.26 |
| `cdladvancedblock` | 162.48 / 3.42 | 1374.76 / 9.31 |
| `cdl3whitesoldiers` | 117.63 / 5.10 | 520.13 / 8.73 |
| `cdlstalledpattern` | 81.90 / 2.35 | 316.45 / 6.11 |
| `linearreg` | 81.32 / 0.76 | 228.83 / 5.00 |
| `linearreg_angle` | 85.77 / 3.79 | 212.78 / 10.53 |
| `linearreg_intercept` | 89.25 / 2.09 | 217.81 / 3.11 |
| `linearreg_slope` | 56.21 / 0.73 | 212.11 / 2.07 |
| `tsf` | 54.29 / 0.91 | 200.61 / 3.43 |

Median indicator-package import time across the isolated processes was 28.25 / 23.57 ms. No dependency or import-time calculation was added. Raw evidence includes dependency imports separately; these import numbers are observations under filesystem caching, not a claimed import optimization.

### Implementation and equivalence

- Paired Hilbert/MAMA outputs share one batch calculation; their original implementations ran it twice.
- Hilbert recurrence order is unchanged. Python doubles and indexed state remove NumPy scalar/dictionary dispatch, while cached scalar trigonometric weights preserve the original `math.sin`/`math.cos` values.
- CCI and regression accumulate each independent window in the original chronological order using NumPy slices; large-period, short-output histories use sequential native cumulative sums.
- Candlestick kernels seed the original moving-total recurrence and run its increments through sequential `cumsum`; this preserves rounding, ties and NaN propagation that a new rolling mean could change.
- Initial screening covered every candlestick and regression indicator. Advance Block, Three White Soldiers and Stalled Pattern were the three largest candlestick costs; all five regression/forecast functions were optimized. The raw evidence retains every initial screening time.

The 409 contract cases compare the candidate with the original source on the same runtime, with exact numerical equality and no relaxed tolerance. Both eager and lazy frames are checked, including empty/short histories, warmup boundaries, real-derived flat/tied prices, Float32/integer inputs, every OHLC missing-value position, alternate price columns, MAMA limits and periods 2, 7, 30 and 100,000. The longest input contains 100,002 recorded bars.

The original Hilbert implementation produces different flat-price phase results across the macOS and Linux runtimes, where tiny floating-point differences reach zero-sensitive phase branches. Therefore, a macOS-only output snapshot is not a portable oracle. `indicator_equivalence_reference.zip` contains the original source with only its internal import namespace relocated; committed SHA-256 values verify every original file. Tests compare both implementations within each runtime instead of accepting a numerical discrepancy.

CI runs covered exact equivalence and a three-sample untraced relative latency check for every changed indicator on 3,000 recorded bars. CCI must take less than half the baseline time; each remaining calculation must take less than 90%. The standalone 20k/83k benchmark additionally rejects any measured latency regression.

### Remaining candlestick calculations

The remaining 44 candlestick indicators convert input arrays to Python doubles once before their existing scalar loops. Small moving-total states use Python double lists. Pattern predicates, seeds and recurrence order stay unchanged; no native pattern rewrite or dependency is introduced.

| Rows | Sum of standalone CPU ms, before → after | Sum of standalone wall ms, before → after | Largest peak RSS increase MiB |
|---|---:|---:|---:|
| 20,000 | 1799.4 → 654.6 | 3527.8 → 3256.5 | 0.8 |
| 83,136 | 7190.9 → 2588.0 | 8778.7 → 4111.3 | 15.5 |

These totals add independent calls; they are not a production-workload benchmark. Measurements use real BTCUSDT 15-minute bars on M1 Max, Python 3.12.12, NumPy 2.5.1 and Polars 1.42.1, with numerical-library thread limits set to one. Wall values are medians of three warm calls and include desktop contention. Input conversion adds bounded list storage; the table reports its observed RSS cost.

All 88 size/indicator comparisons preserve exact outputs and schemas. CI also compares eager/lazy results on short, empty, flat, tied, missing, Float32/integer and alternate-column inputs and checks each indicator's CPU cost against the frozen original source. [Per-indicator first/warm timings, CPU, imports, RSS, input digests and schemas](https://github.com/Vaquum/Limen/blob/main/tests/fixtures/candles_performance_benchmark.json) retain the full evidence.

Each input list uses approximately 32 bytes per bar (a slot plus a Python double), so four inputs can add about 436 MiB on 3.57 million rows. This is an O(rows) speed/memory trade-off; the 83k RSS observations are not a bound for larger histories. Numerical outputs remain exact; NumPy scalar floating-point warning emission is not part of that numerical contract.

CI structurally compares every predicate, arithmetic update and output assignment with the frozen source, allowing only input unboxing, small-state lists and explicit Float64 multiplication of penetration parameters. This protects patterns absent from the available real input without adding fabricated detections. CPU timing samples batch complete calls until the baseline consumes at least 30 ms.

Reproduce all 44 pairs at both sizes using the committed real-data prefix:

```sh
LIMEN_BENCHMARK_CASES=tests/fixtures/candles_performance_cases.json \
python tests/fixtures/candles_performance_replay.py \
  --input tests/fixtures/indicator_equivalence_long.parquet \
  --reference 728a6eefc19cde2cb0fa3f5312b185e2877bc330 \
  --output /tmp/limen-candles-replay --repeats 3
```

The replay supplies named cases to the existing benchmark; it adds no package runtime code. Each side runs in a fresh process with all six numerical thread limits set to one. RSS is sampled after warm calls, before the separate allocation-tracing probe; tracing does not enter timing. The committed input has the same 20k/83k OHLCV digests as the original cached snapshot, although its whole-file hash differs.

In replay reports, `steady_ms` is median CPU time (`steady_clock: process_time`); `steady_wall_ms` and `samples_ms` retain the wall observations.

### Rolling extrema and Ultimate Oscillator

MIDPRICE and MIDPOINT compare windows in the original order using array slices; few-output, long-window cases retain scalar comparisons. Ultimate Oscillator computes each pressure/range term once before the original accumulation loops. This preserves first-element NaNs, ties, warmups and arithmetic order.

| Calculation | Rows | Wall ms, before → after | CPU ms, before → after | Peak RSS MiB, before → after |
|---|---:|---:|---:|---:|
| `midprice` | 20,000 | 92.01 → 0.61 | 75.49 → 0.61 | 85.3 → 83.6 |
| `midpoint` | 20,000 | 69.84 → 0.63 | 28.83 → 0.63 | 84.1 → 83.9 |
| `midprice168` | 20,000 | 1847.92 → 14.21 | 919.89 → 9.45 | 83.7 → 83.6 |
| `midpoint168` | 20,000 | 490.62 → 10.04 | 304.63 → 6.86 | 83.8 → 83.6 |
| `ultosc` | 20,000 | 110.69 → 9.19 | 74.80 → 8.31 | 84.2 → 83.4 |
| `midprice` | 83,136 | 498.19 → 2.92 | 312.87 → 2.54 | 95.1 → 100.0 |
| `midpoint` | 83,136 | 234.71 → 2.13 | 129.12 → 2.11 | 94.5 → 98.8 |
| `midprice168` | 83,136 | 6445.98 → 70.65 | 3836.22 → 29.07 | 95.6 → 101.4 |
| `midpoint168` | 83,136 | 2207.32 → 26.08 | 1310.78 → 25.14 | 95.3 → 101.1 |
| `ultosc` | 83,136 | 287.63 → 43.40 | 277.20 → 37.68 | 95.1 → 108.4 |

A `168` suffix denotes period 168; other periods use their public defaults. Measurements use real BTCUSDT bars on M1 Max, Python 3.12.12, NumPy 2.5.1 and Polars 1.42.1, with numerical-library thread limits set to one. Wall timings are medians of three warm calls under desktop contention. RSS includes imports and loaded input. All ten comparisons preserve every output and schema exactly; regression tests also cover eager/lazy execution, missing values, flat/tied inputs and full windows at period 100,000. [Raw samples, imports, input digests and schemas](https://github.com/Vaquum/Limen/blob/main/tests/fixtures/extrema_performance_benchmark.json) record the evidence.

Extrema seed each output from its oldest element and apply strict comparisons in chronological order: a seed NaN persists, while later NaNs do not replace the seed. ULTOSC keeps its original sequential sums and epsilon branches; high/low NaN totals can produce a permanent zero tail, an existing difference from TA-Lib preserved by this performance change. Working storage is O(rows); RSS observations are diagnostic, not a memory-reduction claim.

Reproduce the five parameter sets at both sizes through the shared replay, using the committed real-data prefix:

```sh
LIMEN_BENCHMARK_CASES=tests/fixtures/extrema_performance_cases.json \
python tests/fixtures/candles_performance_replay.py \
  --input tests/fixtures/indicator_equivalence_long.parquet \
  --reference 728a6eefc19cde2cb0fa3f5312b185e2877bc330 \
  --output /tmp/limen-extrema-replay --repeats 3
```

The replay uses the existing benchmark. Each implementation runs in a fresh process with six numerical thread limits pinned to one. RSS is sampled before the separate allocation-tracing probe. CPU medians measure execution cost; desktop wall values remain indicative.

### Shared moving averages

SMA keeps its original seed and ordered add/subtract recurrence in a native cumulative scan. APO/PPO reuse MAMA only when their first output does not overwrite the input column; MACDEXT reuses it when fast and slow averages receive the same segment. No cross-call cache is added.

| Calculation | Rows | CPU ms, before → after | Indicative wall ms, before → after | Peak RSS MiB, before → after |
|---|---:|---:|---:|---:|
| `sma` | 20,000 | 5.21 → 0.46 | 47.85 → 0.50 | 85.1 → 84.2 |
| `sma(period=168)` | 20,000 | 5.00 → 0.46 | 6.56 → 0.57 | 84.0 → 84.4 |
| `apo` | 20,000 | 12.60 → 1.30 | 26.36 → 10.61 | 84.2 → 83.9 |
| `ppo` | 20,000 | 10.53 → 1.44 | 12.35 → 1.49 | 84.2 → 83.8 |
| `apo(ma_type=7)` | 20,000 | 75.47 → 33.63 | 227.05 → 41.04 | 84.2 → 84.1 |
| `ppo(ma_type=7)` | 20,000 | 72.78 → 35.79 | 143.48 → 54.00 | 84.0 → 83.8 |
| `macdext(fast_ma_type=7, slow_ma_type=7)` | 20,000 | 80.72 → 38.66 | 127.91 → 87.62 | 84.4 → 84.2 |
| `sma` | 83,136 | 20.10 → 1.07 | 23.96 → 2.19 | 94.5 → 98.3 |
| `sma(period=168)` | 83,136 | 21.31 → 1.20 | 28.81 → 1.67 | 94.7 → 98.1 |
| `apo` | 83,136 | 45.74 → 3.14 | 88.30 → 4.96 | 95.3 → 101.6 |
| `ppo` | 83,136 | 45.62 → 2.89 | 202.22 → 3.49 | 96.6 → 100.8 |
| `apo(ma_type=7)` | 83,136 | 301.21 → 151.92 | 1061.10 → 200.47 | 98.9 → 100.6 |
| `ppo(ma_type=7)` | 83,136 | 303.94 → 152.55 | 1915.80 → 291.99 | 100.4 → 101.4 |
| `macdext(fast_ma_type=7, slow_ma_type=7)` | 83,136 | 325.87 → 155.27 | 1060.79 → 218.33 | 104.3 → 107.0 |

Other parameters use their public defaults. Measurements use real BTCUSDT bars on M1 Max, Python 3.12.12, NumPy 2.5.1 and Polars 1.42.1, with numerical-library thread limits set to one. Wall values are medians of three warm calls under desktop contention. RSS includes imports and input loading.

Every output and schema compares exactly in all fourteen size/calculation pairs. CI also checks eager/lazy execution, all APO/PPO moving-average types, missing values, ties, maximum-period accumulation, input-column collisions and MAMA invocation counts. [Raw first/warm samples, CPU, imports, RSS, input digests and schemas](https://github.com/Vaquum/Limen/blob/main/tests/fixtures/smoothing_performance_benchmark.json) preserve the evidence.

CPU medians are the performance measure; contended desktop wall samples are indicative only. The scan uses O(rows) extra storage (up to 5 MiB RSS observed at 83,136 rows). SMA retains permanent NaN propagation. NumPy floating-point warning wording is not part of the numerical contract. MAMA reuse applies only to type 7; types 1–6 and 8 keep their existing calculations.

Reproduce the seven parameter sets at both sizes with the committed real-data prefix (the OHLCV hashes above match the original snapshot):

```sh
LIMEN_BENCHMARK_CASES=tests/fixtures/smoothing_performance_cases.json \
python tests/fixtures/candles_performance_replay.py \
  --input tests/fixtures/indicator_equivalence_long.parquet \
  --reference 728a6eefc19cde2cb0fa3f5312b185e2877bc330 \
  --output /tmp/limen-smoothing-replay --repeats 3
```

The replay uses the existing benchmark with named parameter sets. It runs each implementation in a fresh process, pins all six thread limits to one, checks every recorded output and schema, and includes input conversion in timing. RSS is the process high-water mark after warm calls; allocation tracing runs afterwards and does not enter timing or RSS.

In replay reports, `steady_ms` is median CPU time (`steady_clock: process_time`); `steady_wall_ms` and `samples_ms` retain the wall observations.
