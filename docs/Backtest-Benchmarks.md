# Backtest engine benchmarks

`python -m tests.benchmark_backtest_strategies` compares Limen's snapshot
backtester with vectorbt's Numba and Rust backends, backtesting.py and
Backtrader. It sends the same recorded prices and causal strategy signals to
each engine, checks every equity observation and fill against an independent
cash replay, and then checks the competitor output against Limen.

The benchmark measures execution correctness and runtime under a declared
long/flat contract. Trading results after fees are a separate native SFD
campaign. The eight fixed strategy definitions, parameters and split behavior
are canonical in [Common strategy baselines](Common-Strategies.md).

The [Python project](examples/python-research/README.md) declares its own trading
score and direct event-policy cases. The [Compiled project](examples/compiled-research/README.md)
records native snapshot and event outputs. The historical campaigns below
compare their named snapshot workloads under explicit parity rules; they do
not replace current project proof or establish event-execution parity.

## Prerequisites

- A Limen repository checkout, Python 3.12 and the recorded timestamped OHLCV parquet fixtures.
- Separate supported environments for Limen and vectorbt, configured below.
- A new output directory and a source tree that stays unchanged throughout measurement.

## Compared implementations

| CLI name | Implementation | Measured execution boundary |
|---|---|---|
| `limen` | `long_flat_strategy` | Native return kernel; the public `backtest_snapshot` path is also timed separately. |
| `vectorbt` | `Portfolio.from_signals`, `engine="numba"` | Portfolio construction with the Numba backend explicitly selected. |
| `vectorbt_rust` | `Portfolio.from_signals`, `engine="rust"` | The same portfolio API with the optional Rust backend explicitly selected. |
| `backtesting_py` | `FractionalBacktest.run` | Fractional-unit execution, including the API's native statistics. |
| `backtrader` | `Cerebro.run` | Preloaded execution with `runonce=True` and standard observers disabled. |

These are five engine configurations across four projects. They cover a native
return kernel, vectorized portfolio construction and two event-loop frameworks.
The selection does not establish a popularity ranking or exhaust the available
backtesting software. The [vectorbt portfolio API](https://vectorbt.dev/api/portfolio/base/),
[backtesting.py API](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html)
and [Backtrader Cerebro API](https://www.backtrader.com/docu/cerebro/) describe
the public boundaries used here.

## Reproduce the environment

Run from a Limen repository checkout. The benchmark is repository tooling,
under `tests/`, and is not installed as a separate package command. Use Python
3.12 and keep both virtual environments outside the checkout.

The following comparison-package versions match the implementation setup:

| Step | Command |
|---|---|
| 1 | `python3.12 -m venv ../limen-benchmark-env` |
| 2 | `../limen-benchmark-env/bin/python -m pip install -e '.[data]' 'numpy==2.5.4' 'pandas==2.3.3' 'polars==1.44.2' 'backtesting==0.6.6' 'backtrader==1.9.78.123'` |
| 3 | `python3.12 -m venv ../vectorbt-benchmark-env` |
| 4 | `../vectorbt-benchmark-env/bin/python -m pip install 'numpy==2.5.4' 'pandas==3.0.6' 'vectorbt[rust]==1.1.2' 'numba==0.68.0' 'llvmlite==0.50.0'` |

To reproduce the complete recorded package stacks, use the published
[Limen environment pins](benchmarks/common-strategies-2026-10-18/limen-requirements.txt)
and [vectorbt environment pins](benchmarks/common-strategies-2026-10-18/vectorbt-requirements.txt)
in place of the installation commands above. They pin every installed
distribution except the editable Limen checkout, which is installed separately
from the measured source revision:

| Step | Command |
|---|---|
| 1 | `../limen-benchmark-env/bin/python -m pip install -r docs/benchmarks/common-strategies-2026-10-18/limen-requirements.txt` |
| 2 | `../limen-benchmark-env/bin/python -m pip install --no-deps -e .` |
| 3 | `../vectorbt-benchmark-env/bin/python -m pip install -r docs/benchmarks/common-strategies-2026-10-18/vectorbt-requirements.txt` |

The recorded reference-engine environment pins pandas 2.3.3; vectorbt 1.1.2 declares
`pandas>=3.0.3,<4`. Separate environments preserve that recorded comparison. Limen itself no longer requires pandas at runtime.
The controller prepares NumPy archives once, and competitor workers read those
archives without importing Limen. No Limen install is needed in the vectorbt
environment. The installed vectorbt Rust extra must match vectorbt's version;
see the [official installation instructions](https://vectorbt.dev/getting-started/installation/).
The adapter requests a specific backend and rejects an unavailable or
incompatible selection instead of substituting another engine.

Preserve the full dependency inventories alongside a published result:

| Step | Command |
|---|---|
| 1 | `../limen-benchmark-env/bin/python -m pip freeze --all > ../limen-benchmark-env/requirements.lock.txt` |
| 2 | `../vectorbt-benchmark-env/bin/python -m pip freeze --all > ../vectorbt-benchmark-env/requirements.lock.txt` |

The setup commands pin the principal comparison packages. Transitive packages,
the Python patch release and the operating system also belong to a reproducible
environment; the report records runtime versions and platform information, and
the dependency inventories retain the rest. Keep the exact source commit and
record any uncommitted changes. The controller hashes the implementation and
rejects a campaign if source code changes during measurement.
Differences in dataframe-library versions are part of these supported software
stacks and can affect setup and API costs.

## Run every engine and strategy

This command runs all eight strategies against all five configurations on both
bundled timestamped market fixtures:

| Step | Command |
|---|---|
| 1 | `../limen-benchmark-env/bin/python -m tests.benchmark_backtest_strategies --inputs tests/fixtures/spot_1h_20240101_20241231.parquet tests/fixtures/spot_15m_20250101_20250531.parquet --engines limen vectorbt vectorbt_rust backtesting_py backtrader --engine-python vectorbt=../vectorbt-benchmark-env/bin/python --engine-python vectorbt_rust=../vectorbt-benchmark-env/bin/python --repeats 7 --output ../backtest-engine-results` |

The output directory must be new or empty. Every completed worker updates the
reports, so a failed competitor remains visible. An unavailable optional engine
is marked `unavailable`; a path mismatch is `parity_failed`, and a worker error
is `error`. A competitor receives `unranked` status if the Limen baseline did
not verify. A campaign with a mismatch or execution error exits unsuccessfully.
A campaign with unavailable optional engines preserves that status explicitly.

For a smaller diagnostic run, use one recorded input and add `--rows 600`.
`--rows` is the number of evaluated observations, so 600 requires 601 actual
source rows. `--strategies` accepts any of the eight catalogue names, and
`--engines` accepts the names in the table above. Limen always runs first.
The minimum repeat count is three; a percentile from so few samples is only a
descriptive statistic.

### Compare the strategy matrix separately

The `--batch` workload compares one vectorbt portfolio matrix with serial Limen
calls for the same strategy columns. Run it in its own output directory:

| Step | Command |
|---|---|
| 1 | `../limen-benchmark-env/bin/python -m tests.benchmark_backtest_strategies --inputs tests/fixtures/spot_1h_20240101_20241231.parquet tests/fixtures/spot_15m_20250101_20250531.parquet --batch --engines limen vectorbt vectorbt_rust --engine-python vectorbt=../vectorbt-benchmark-env/bin/python --engine-python vectorbt_rust=../vectorbt-benchmark-env/bin/python --repeats 7 --output ../backtest-matrix-results` |

All eight catalogue strategies are selected by default. Each column has its
own initial cash and account history; vectorbt uses `cash_sharing=False` and
no grouping. The matrix is eight independent scenarios, not a combined trading
portfolio. The same equity-and-fill checks apply separately to every column.
Backtesting.py and Backtrader are covered by the individual-strategy workload;
they are not admitted to this matrix adapter.

Matrix timings cover the complete selected group. Reports retain
`portfolio_count`, and JSON stores each strategy's outcomes under
`metrics.by_strategy`; the CSV has one timing row per engine/input matrix.
The result archive preserves each column independently. Keep matrix timing
separate from individual-strategy timing: one matrix API call and eight serial
calls amortize setup differently. The public Limen snapshot summary is timed
in the individual-strategy campaign, not added to the matrix measurement.

### Recorded-data contract

Inputs are parquet files with `datetime`, `open`, `high`, `low`, `close` and
`volume`. Timestamps must use a Polars Datetime type and increase strictly
without duplicates. OHLC values must be finite and positive, volume must be
finite and nonnegative, and high/low must contain the recorded open and close.
Required nulls and inconsistent ranges reject the input.

The controller retains each input file's SHA-256 digest, actual timestamps and
row counts, the prepared-array digest and the original strategy-signal digest.
Each worker also reports `evaluation_rows` and `processed_rows`, exposing which
adapters process the settlement row.
With no row limit it uses the file's observations, reserving the final real row
for settlement. It does not repeat data, invent timestamps, append a copied
price or pad a short file. The larger indicator-equivalence fixture is excluded
because it has no recorded timestamps; its size alone cannot justify using it
for a chronology-checked engine comparison.

## Common execution contract

| Dimension | Contract |
|---|---|
| Market input | One recorded instrument per scenario. |
| Starting account | 1,000,000 cash units. |
| Position | All available capital long or entirely flat; no shorts or leverage. |
| Quantity | Fractional units; no shared account across independent strategies. |
| Signal | The catalogue's completed-bar desired position; warm-up remains flat. |
| Timing | Signal at close `t` controls exposure from close `t` to close `t+1`. Limen uses `execution_lag_bars=1`. |
| Fills | Idealized unconstrained fills at the recorded signal-bar close. |
| Costs | Zero fees, fixed fees, spread and slippage. |
| Ending account | Liquidate at the last evaluated close; process one further real source row where the engine needs it to settle the queued order. |
| Population | Every evaluated row, including initial warm-up and flat observations. |

The independent reference maintains cash and quantity directly. It does not
reuse Limen's return kernel. An entry buys `cash / close`, an exit converts the
held units back to cash, and each evaluated close marks the account. The
common return path derives from those equity marks. This idealized close-fill
contract does not model exchange latency, queue position, partial fills or
volume participation.

### Why one additional real bar is retained

The last evaluated row requests flat. Backtesting.py and Backtrader process
that queued close order on the next iteration while pricing it at the recorded
creation close. The extra source row enables that iteration. Its price return
is excluded from evaluation, and it is excluded from strategy preparation.
Limen receives only the evaluated rows and applies its native final exit;
the other adapters may process the extra row and then truncate their reported
account path to the common evaluation window.

The benchmark verifies the final fill's row and price. It therefore detects an
engine closing one bar early even when the final account value looks plausible.
Backtesting.py uses `trade_on_close=True` with `finalize_trades=False`; the
adapter closes the position explicitly and uses the next real row for
settlement. Backtrader uses its documented
[close-execution broker setting](https://www.backtrader.com/docu/broker/).

### Fractional quantities and the first equity observation

Backtesting.py's [FractionalBacktest](https://kernc.github.io/backtesting.py/doc/backtesting/lib.html)
uses `fractional_unit=1e-12`. Its allocation fraction is the floating-point
value immediately below one, avoiding an integer-size interpretation while
deploying effectively all available cash. Backtrader likewise requests the
representable quantity just below `cash / close`, avoiding a rounding-induced
cash rejection. Quantity rounding still has to satisfy the equity tolerance.

The backtesting.py adapter records the actual broker equity in `Strategy.init`.
Its native account curve leaves the first slot unrecorded and backfills it from
the following observation. Normalization replaces only that unrecorded first
slot with the broker observation, preventing the first earned return from
appearing one row early. Subsequent equity observations and all trade records
come from the engine. The worker's `settings` record this normalization.

## Correctness gates and timing scopes

Every engine must match the independent reference on the complete equity path,
held-bar flags, fill rows, fill sides and fill prices. The controller then
compares each verified competitor against Limen's saved output. Equity uses
relative tolerance `1e-10` and absolute tolerance `1e-6` in account currency;
fill prices use relative tolerance `1e-12` and absolute tolerance `1e-8`.
Held states, fill rows and fill sides must match exactly. Each repeated run is
checked again. Matching only terminal return or aggregate trade count is
insufficient.

| Report field | Included work |
|---|---|
| `parquet_read_validate_ms` | Recorded file loading and input validation in the controller. |
| `signal_preparation` | First and repeated calls to the shared pure strategy function. |
| `adapter_preparation_ms` | Framework imports and conversion/setup performed while constructing the adapter. |
| `cold.engine_ms` / `warm.engine_ms` | The declared engine API call. |
| `normalization_ms` | Extracting the engine's equity and fill records and computing the common summary. |
| `common_outputs_ms` | Engine execution plus normalization and the common summary. |
| `cpu_ms` | Process CPU consumed by engine execution and common output extraction. |
| `native_snapshot` | Limen's public snapshot validation, execution and complete native bar-metric summary, timed separately. |
| `process_wall_ms` | Worker process lifetime, including startup, verification and artifact writes. |
| `peak_rss_mib` | Process high-water resident memory, including imports and preparation. |

Each engine/scenario runs in a fresh worker with a separate initially empty
Numba cache directory. The first execution sample is consequently distinct
from repeated execution in that worker. Imports and adapter setup are reported
separately from that first execution; `process_wall_ms` is the broader
process-level measurement. Warm timing retains every raw sample plus the
median and sample p95. A warm repeat executes the engine again; it does not
return a retained portfolio object.

`native_snapshot.first_call_ms` is the first public snapshot API call after
that worker has already exercised the return kernel and its repeated runs.
It is not a fresh-process cold snapshot measurement.

The runner limits supported BLAS/OpenMP, Polars, Numba and Rayon thread settings
to one and records them. It also records logical CPU count and available CPU
affinity. Memory is a process high-water observation, not a count of allocations
made by the timed engine call. The separate native snapshot measurement can
raise its own process peak, which is recorded separately.

Public APIs do different amounts of additional work. For example,
backtesting.py's run computes native statistics, while Limen's return kernel
does not. Report the engine and common-output timings together, with Limen's
full snapshot timing alongside them. No single ratio makes these APIs identical
workloads or establishes a universal speed ranking.

## Artifacts and measured results

The output root contains `report.json` and `report.csv`. JSON is the complete
record: it includes source and code provenance, dependency versions, thread
settings, selected strategy parameters, timings and correctness statuses.
CSV is a compact per-scenario/per-engine view suitable for tabulation.

Each scenario directory contains `prepared.npz`, the per-engine JSON report,
and the per-engine NumPy result archive. Result archives retain equity,
held-bar states and fill row/side/price arrays. Preserve these files together
when sharing a result; they make timing claims auditable against the actual
financial output.

A measured comparison should identify the evaluated row count, instrument and
date window, source/code digests, Python and engine versions, hardware,
repeat count, correctness status, and the exact timing scope. Publish failed
or unavailable cases alongside successful ones. Only `verified` cases support
performance comparisons under this contract.

### 2026-10-18 campaign

Both campaigns completed: **80 of 80 individual cases** passed verification
(two inputs, eight strategies and five engines), and **six of six matrix
cases** passed verification (two inputs and three engines, with all eight
independent portfolio columns checked in each case). Every warm repeat also
passed the same financial-output checks. The largest recorded absolute equity
error against the independent cash replay was `1.132e-7` account-currency
units; the report records this error on each worker's first result.

The individual and matrix campaigns use Limen 5.34.0 at source revision
`8640fd3bc378413d6fde4afa97fce188f60ad94f`, Python 3.12.14 and the package
versions pinned above. The runtime is Linux 6.18.55 on x86-64 with glibc 2.39,
nine reported logical CPUs and affinity covering those nine CPUs. The CPU
model is unavailable; the reported processor value is only `x86_64`.
Supported thread settings are fixed to one as recorded in each report.

Both campaigns recorded the same implementation digest,
`962f8208ce303c37a034a437301820a1e951ed72b447a00ebb2a8296d5d393ba`.
The individual report records a clean checkout. The matrix report records
the documentation and evidence edits made while the individual campaign
ran; runtime source stayed unchanged. The aggregate reports preserve those
working-tree observations without alteration.

Both inputs are recorded BTCUSDT spot bars. The final real bar of each
fixture supplies terminal settlement and is excluded from evaluation. All
timestamps below are UTC; full input-file and signal digests are retained
in the reports.

| Input | Evaluated bars | First evaluated bar | Last evaluated bar | Settlement bar |
|---|---:|---|---|---|
| Hourly 2024 | 8,783 | 2024-01-01 00:00 | 2024-12-31 22:00 | 2024-12-31 23:00 |
| 15-minute 2025 | 14,495 | 2025-01-01 00:00 | 2025-05-31 23:30 | 2025-05-31 23:45 |

For individual strategies, the table aggregates in two stages: first take
the median of seven warm repetitions for each strategy, then take the median
of those eight strategy medians for each input and engine. Each range gives
the minimum and maximum of those same eight strategy medians. The ranges
describe strategy-to-strategy variation; they are not confidence intervals
or the minimum and maximum of all timing samples. Engine-API time and time to
produce the common equity, fills and summary remain separate columns.

| Input | Engine | API ms: median [min, max] | Common outputs ms: median [min, max] |
|---|---|---:|---:|
| Hourly 2024 | Limen kernel | 0.532 [0.409, 1.393] | 0.918 [0.725, 2.450] |
| Hourly 2024 | vectorbt Numba | 5.022 [4.586, 7.406] | 9.766 [8.642, 15.570] |
| Hourly 2024 | vectorbt Rust | 3.854 [3.350, 5.534] | 8.049 [7.235, 11.715] |
| Hourly 2024 | backtesting.py | 60.955 [54.995, 78.524] | 61.463 [55.304, 78.902] |
| Hourly 2024 | Backtrader | 1208.464 [984.100, 1523.669] | 1209.572 [985.203, 1524.841] |
| 15-minute 2025 | Limen kernel | 0.807 [0.745, 0.902] | 1.232 [1.152, 1.407] |
| 15-minute 2025 | vectorbt Numba | 5.687 [5.540, 6.407] | 12.013 [10.399, 15.561] |
| 15-minute 2025 | vectorbt Rust | 4.006 [3.379, 5.485] | 8.484 [7.670, 14.171] |
| 15-minute 2025 | backtesting.py | 85.010 [79.328, 165.000] | 85.605 [79.807, 166.420] |
| 15-minute 2025 | Backtrader | 1723.801 [1614.486, 2381.252] | 1727.248 [1615.651, 2383.199] |

### Complete public snapshot

Limen's full public snapshot uses the same two-stage aggregation and is
reported separately because it additionally computes its native bar-metric
summary. These are warm calls to `backtest_snapshot`; its first-call field
also follows the return-kernel runs and is not a fresh-process cold snapshot.

| Input | Full snapshot ms: median [min, max] |
|---|---:|
| Hourly 2024 | 3.537 [2.863, 14.907] |
| 15-minute 2025 | 4.846 [4.184, 6.903] |

### Eight-portfolio matrix

The matrix is a separate workload: each measurement executes all eight
independent strategy portfolios, and the table reports the median and
minimum/maximum of its seven whole-matrix repetitions. Limen makes eight
serial kernel calls; vectorbt makes one matrix portfolio call. The values
measure the entire eight-portfolio operation, without division by eight.

| Input | Engine | API ms: median [min, max] | Common outputs ms: median [min, max] |
|---|---|---:|---:|
| Hourly 2024 | Limen, eight serial calls | 4.415 [3.737, 7.315] | 6.658 [6.379, 10.200] |
| Hourly 2024 | vectorbt Numba matrix | 19.705 [16.878, 22.671] | 41.328 [30.911, 47.188] |
| Hourly 2024 | vectorbt Rust matrix | 12.423 [10.298, 15.983] | 26.734 [23.038, 30.765] |
| 15-minute 2025 | Limen, eight serial calls | 5.740 [5.085, 7.126] | 8.523 [7.981, 11.396] |
| 15-minute 2025 | vectorbt Numba matrix | 25.743 [23.558, 30.233] | 42.593 [41.257, 54.749] |
| 15-minute 2025 | vectorbt Rust matrix | 19.430 [18.506, 22.453] | 43.849 [38.705, 57.869] |

### Interpretation and evidence

Limen's common-output path had the lowest aggregate warm time in both
individual windows and both matrix windows measured here. Its full snapshot
required 3.537 and 4.846 ms at the same aggregate statistic, showing the
additional work behind the public research interface. Public APIs retain
their different native responsibilities, so these results support the stated
workloads and timing scopes. They do not establish a universal engine ranking.
Seven warm samples describe local timing variation and do not estimate tail
latency.

The following aggregate reports are byte-preserved copies of the completed
campaign outputs. They retain the first execution sample, all warm samples,
setup and memory measurements, package versions and correctness evidence.
The derived summary records unrounded table values and the exact aggregation
rules. Full per-scenario NumPy archives accompany the campaign output; the
compact documentation evidence contains the aggregate reports and summaries.

| Evidence | JSON | CSV |
|---|---|---|
| Individual campaign | [Full report](benchmarks/common-strategies-2026-10-18/single-report.json) | [Per-case rows](benchmarks/common-strategies-2026-10-18/single-report.csv) |
| Matrix campaign | [Full report](benchmarks/common-strategies-2026-10-18/matrix-report.json) | [Per-case rows](benchmarks/common-strategies-2026-10-18/matrix-report.csv) |
| Derived timing tables | [Summary and report hashes](benchmarks/common-strategies-2026-10-18/performance-summary.json) | [Unrounded summary](benchmarks/common-strategies-2026-10-18/performance-summary.csv) |

## Costed SFD research and current limits

Run the [common-strategy SFD](Common-Strategies.md#run-the-eight-strategy-grid)
separately for native strategy results with fees and slippage. Its defaults are
10 bps fee and 5 bps slippage per fill. UEL prepares its train, validation and
test partitions independently, so state and indicator warm-up restart at each
split. The engine benchmark instead prepares one continuous evaluated window
and uses zero costs.

The completed [2026-10-18 native run](Common-Strategies.md#recorded-costed-results-2026-10-18)
reports all eight strategies' costed returns, maximum drawdowns and trade
counts on the 1,598-row held-out test split. Those research results use a
different window and state-reset protocol from the speed comparison.

Zero costs are an explicit limit of cross-engine parity here. Limen's legacy
snapshot deducts entry fees using its deployed arithmetic; an affordable
cash-spot engine sizes the purchase after reserving fees. Giving both engines
the same nonzero fee number would not make those account models identical.
The benchmark therefore does not produce a cross-engine costed ranking.

The current comparison covers Limen's snapshot long/flat path. It does not
benchmark Limen's signed event-execution ledger, funding, elapsed exits,
take-profit/stop-loss behavior, margin, order-book execution or multi-asset
portfolio constraints. It also does not test GPU execution, large parameter
sweeps or optimizer scheduling. Its recorded windows and fixed rules support bounded
correctness and timing observations; they do not establish out-of-sample
profitability, capacity or future live returns.

## Backtest computation reuse

The computation-reuse comparison runs the baseline and candidate source trees in
separate processes against the same recorded market data and prepared signals.
It requires absolute numerical error no greater than `1e-10`, with zero relative
tolerance. Table schemas, row and column order, null masks, identifiers, clocks
and contract digests must agree. Snapshot regressions use the tighter `1e-12`
absolute tolerance and preserve the original CVaR tail summation order.

| Step | Command |
|---|---|
| 1 | `mkdir -p /tmp/limen-backtest-baseline` |
| 2 | `git archive ec254c73f0ff47cfcc2017e262458b2d5f671fb0 \| tar -x -C /tmp/limen-backtest-baseline` |
| 3 | `python -m tests.benchmark_backtest_reuse --baseline-root /tmp/limen-backtest-baseline --output "$(mktemp -d /tmp/limen-backtest-reuse.XXXXXX)/campaign"` |

The benchmark measures the snapshot ledger, complete snapshot evaluation,
repeated line features, full timer-driven event execution and forward trade
labels. First-call and warm timings are separate. The timer case retains every
audit state row; label evaluation retains both directions and all label metadata.
Source loading and shared signal preparation are excluded from each timed case.
The report records source-code hashes, input checksums, dependency versions and
the maximum absolute numerical error for every case.

### Recorded comparison: 2026-10-18

Baseline `ec254c73` and computation-reuse code `4754d096` ran on the same host with one native thread, NumPy 2.5.1 and Polars 1.42.1. These are medians of three warm samples; they describe the named workloads and do not predict a complete experiment sweep.

| Workload | Baseline | Reuse | Speedup |
|---|---:|---:|---:|
| Snapshot ledger, 8,784 bars | 1.177 ms | 0.699 ms | 1.69x |
| Complete snapshot, 8,784 bars | 1.597 ms | 1.103 ms | 1.45x |
| 32 rounds of two line features | 607.907 ms | 163.300 ms | 3.72x |
| Timer execution, all 43,201 state rows | 211.680 ms | 17.638 ms | 12.00x |
| 168 hourly bars with recorded funding | 10.115 ms | 2.616 ms | 3.87x |
| 256 forward candidates, TP/SL exits | 859.438 ms | 22.610 ms | 38.01x |

Every compared numeric output had maximum absolute error **0.0**, exceeding the required ten-decimal equivalence. Non-numeric output contracts also matched, including all timer ledger rows, both label directions, funding cashflows and provenance. See the [machine-readable report](benchmarks/backtest-reuse/report.json) and [CSV summary](benchmarks/backtest-reuse/summary.csv) for source/input hashes and all timing samples.

The standalone campaign fails on numerical/structural differences or any named warm speedup at or below 1.0. CI checks the committed report's acceptance, recorded source identities and current fixture hashes, independent baseline outputs and deterministic operation counts. Ratios remain host-dependent; funded timers and joint feature/target throughput have no measured speed claim.

A second campaign with identical production-code hashes also matched exactly. Its process peak RSS (including imports, inputs and output persistence) was 307.8 → 270.1 MiB for line features, 305.3 → 258.1 MiB for timer execution, and within 1.3 MiB of baseline for the other four cases. Cold timings and normalized RSS for all six cases are in the report's `resource_campaign`; these observations do not establish a universal memory or cold-latency bound.

Line-feature reuse is bounded to one invocation. Event plans retain validated source clocks and use content-addressed identities; changed predictions retain dynamic validation. Timer blocks reconstruct all original state rows. Unfunded forward candidates stop after their independent episode closes; candidates that remain open still inspect the remaining events. Funded candidates retain full execution, including custom adapter behavior.
