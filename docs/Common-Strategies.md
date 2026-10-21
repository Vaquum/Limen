# Common strategy baselines

`common_strategies` is a manifest-backed, rule-based Single-File Decoder (SFD)
that evaluates eight fixed long/flat baselines through Limen's existing
`RuleBasedManifest` and `RuleBasedStrategy`. It spans passive exposure, trend
following, momentum and mean reversion. These distinct holding patterns make
the catalogue useful both for strategy research and for checking backtest
engines against identical position requests.

The [Python project](examples/python-research/README.md) uses the same eight
signal rules in a caller-owned next-open-to-close comparison. The
[Compiled project](examples/compiled-research/README.md) executes their native
foundational SFD across all eight candidates. Their desired signals agree;
their trading policies differ and their returns must not be compared as if the
accounts were equivalent.

## Prerequisites

- A Limen checkout or installed package; use the `data` extra for automatic market-data fetching.
- Recorded, chronological bars with finite positive close prices; Donchian also needs high and low.
- A declared bar interval, evaluation window, execution convention and cost scenario.

The catalogue is a set of declared baseline choices. Its parameters have not
been selected for profitability on the bundled observations. All periods below
count input bars: a 20-bar lookback means 20 hours on hourly observations, and
five hours on 15-minute observations.

## Rules and defaults

Each rule produces a completed-bar desired position: `1` for long and `0` for
flat. The catalogue includes no short exposure, leverage, pyramiding or learned
parameters.

| Strategy parameter | Entry / long rule | Exit / flat rule | Initial flat rows |
|---|---|---|---:|
| `buy_and_hold` | Request long from the first completed bar | Remain long; the executor owns final liquidation | 0 |
| `sma_cross` | SMA(50) exceeds SMA(200) | SMA(50) is at or below SMA(200) | 199 |
| `ema_cross` | EMA(12) exceeds EMA(26) | EMA(12) is at or below EMA(26) | 25 |
| `macd` | MACD(12,26) exceeds its EMA(9) signal line | MACD is at or below its signal line | 33 |
| `rsi_mean_reversion` | Wilder RSI(14) is at or below 30 | Hold until RSI is at or above 70 | 14 |
| `bollinger_mean_reversion` | Close is at or below SMA(20) minus two population standard deviations | Hold until close is at or above SMA(20) | 19 |
| `donchian_breakout` | Close exceeds the maximum high of the previous 20 bars | Hold until close falls below the minimum low of the previous 10 bars | 20 |
| `time_series_momentum` | Close exceeds the close 20 bars earlier | Close is at or below that earlier close | 20 |

The SMA, EMA, MACD and momentum variants are regime rules: they can enter at the
first row with valid indicator history if the long condition already holds.
They do not require an earlier observed cross. The RSI, Bollinger and Donchian
variants maintain their position between their separate entry and exit events;
an entry is not a one-bar pulse. An exit wins if both events coincide.

EMA starts from an SMA seed and uses alpha `2 / (period + 1)`. RSI uses Wilder
smoothing with alpha `1 / period`. MACD uses Limen's TA-Lib-compatible
initialization. Bollinger standard deviation uses `ddof=0`. Both Donchian
channels exclude the current bar, so today's high cannot move the threshold
that today's close must break.

These families are described in the primary implementation and author sources
below. The exact trading variants and lookbacks in this table are Limen's
declared comparison settings. In particular, the momentum baseline is a
20-bar long/flat adaptation, and the Bollinger entry/exit rule is a selected
mean-reversion variant; neither is a reproduction of the source authors'
complete strategy.

## Run the eight-strategy grid

The installed CLI exposes the `common_strategies` template through `limen init`; see the [command reference](Command-Line-Interface.md). Its native foundational entry points are:

<!-- reference: common-foundational-params source=limen/sfd/foundational_sfd/common_strategies.py region=signature:params role=signature -->
```python
def params() -> dict[str, list[str]]:
    ...
```

<!-- reference: common-foundational-manifest source=limen/sfd/foundational_sfd/common_strategies.py region=signature:manifest role=signature -->
```python
def manifest() -> Manifest:
    ...
```

The template fetches real hourly observations for calendar year 2024 from the
configured Limen source. Its explicit date splits are January through September
for training, October for validation, and November through December for testing.
Fee is 10 basis points and slippage is 5 basis points on each entry and exit fill. These are explicit
scenario assumptions, not measured costs. Execution uses the existing snapshot
backtest's one-bar lag and final-exit convention; see [Backtest](Backtest.md).

When no dataframe is supplied, the Python SFD requests hourly calendar-2024
data in both normal and test mode. Supplying `data=` uses those recorded rows
directly, without fetching either source. The benchmark always supplies its
explicit fixture data; it does not depend on a changing latest-data window.

The Compiled project reads the complete recorded hourly fixture through `common_strategies.yaml`, with explicit January–September training, October validation and November–December testing windows. It runs all eight candidates through the CLI and retains native execution records. The historical factory benchmark below has its own row-count split and is not the current YAML result.

<!-- example: common-compiled-baselines source=docs/examples/compiled-research/common_strategies.yaml region=declaration scenario=search interface=compiled role=executed -->
```yaml
schema_version: "1.0"

metadata:
  name: common_strategies
  mode: development
  description: Eight fixed long-flat strategy baselines with causal indicator history

sfd:
  manifest:
    type: rule_based
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20241231.parquet
    required_columns: [datetime, open, high, low, close]
    split_dates:
      train_start: "2024-01-01"
      train_end: "2024-10-01"
      val_start: "2024-10-01"
      val_end: "2024-11-01"
      test_start: "2024-11-01"
      test_end: "2025-01-01"
    indicators:
      - func: limen.features.common_strategy_positions.common_strategy_positions
        params:
          strategy: "{strategy}"
    strategy:
      conditions:
        - id: common_strategy_long
          name: Common strategy requests long
          type: threshold
          column: common_strategy_position
          operator: ">"
          value: 0
      entry: common_strategy_long
    reference_architecture: limen.sfd.reference_architecture.rule_based.rule_based
    backtest:
      fee_bps: 10.0
      slip_bps: 5.0
  params:
    strategy:
      - buy_and_hold
      - sma_cross
      - ema_cross
      - macd
      - rsi_mean_reversion
      - bollinger_mean_reversion
      - donchian_breakout
      - time_series_momentum

uel:
  output_path: compiled-common_strategies
  record_execution: true
  checkpoint_interval: 1
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
  n_permutations: 8
  search_strategy:
    type: grid
  output_format: csv
```

`params()` has one categorical axis, `strategy`. A candidate chooses one
complete fixed configuration, avoiding combinations of irrelevant parameters
from different strategies. The result includes each split's position rate,
entry count, backtest metrics, executed-trade count and mean compounded net
PnL per executed trade. A strategy with no executed trades has a zero trade
count and an undefined mean trade PnL; it is not evidence of a zero-risk return. The project summary keeps finite activity/mean-bar fields and the complete execution tracks, while the native artifacts retain their original metric values.

## Recorded costed results: 2026-10-18

The historical offline UEL campaign completed all eight candidates on the recorded 2024
hourly BTCUSDT fixture. Its `8:1:2` row split allocated 6,388 bars to training,
798 to validation and 1,598 to testing. The test window ran from
**2024-10-26 10:00 through 2024-12-31 23:00 UTC**. These are the Python
manifest's row-count boundaries; they differ from the YAML template's date
boundaries.

The following test results include the native snapshot's **10 bps fee and
5 bps slippage per fill**, one-bar exposure lag and final liquidation. Signals
observe the completed close, and fills use that same close under the declared
idealized execution contract. Indicator history and held state restart at the
test boundary.

| Strategy | Net total return | Maximum drawdown | Executed trades | Bars held |
|---|---:|---:|---:|---:|
| `buy_and_hold` | 39.17% | -15.13% | 1 | 99.9% |
| `sma_cross` | 14.21% | -14.61% | 5 | 58.2% |
| `ema_cross` | 23.91% | -12.89% | 22 | 58.8% |
| `macd` | -17.54% | -30.89% | 65 | 47.1% |
| `rsi_mean_reversion` | -7.05% | -12.79% | 4 | 29.5% |
| `bollinger_mean_reversion` | 6.87% | -7.78% | 35 | 23.5% |
| `donchian_breakout` | 5.65% | -16.10% | 25 | 46.9% |
| `time_series_momentum` | -13.85% | -29.33% | 90 | 56.1% |

Total return compounds the persisted per-bar net returns from unit initial
equity. Maximum drawdown is the worst peak-to-trough decline of that resulting
equity path; it is distinct from the native `drawdown_bps_p5` percentile.
Bars held is the fraction of test bars with executed long exposure, including
the initial warm-up population. The
[verification report](benchmarks/common-strategies-2026-10-18/native-verification.json)
records the checks against all eight persisted execution arrays, and the
[result CSV](benchmarks/common-strategies-2026-10-18/native-test-results.csv)
retains the unrounded values.

Buy-and-hold returned the most in this rising test window; EMA was the
strongest active baseline by total return. The table also shows that less
market exposure and fewer trades do not by themselves identify the best
result. These are observations from one recorded instrument and window using
fixed settings, not parameter-selection evidence or a live-return forecast.

The run used Python 3.12.14 and Limen 5.34.0. The frozen source is reproducible
from revision `8640fd3bc378413d6fde4afa97fce188f60ad94f`. That revision
association was documented after execution; the native artifacts did not
retain an execution-time Git revision or source-code digest. The
[source provenance record](benchmarks/common-strategies-2026-10-18/native-source-provenance.json)
preserves this limitation explicitly. Cross-engine zero-cost correctness and
timing measurements are reported separately in
[Backtest engine benchmarks](Backtest-Benchmarks.md).

## Python Research and the pure signal API

The Python project calls the shared signal function directly on the same 1,598 assessment bars. It owns independent one-bar trades, entering the next open and exiting that close with a declared 2 bp round-trip cost. This differs from the native held-position snapshot, its entry/exit pricing and 10/5 bp per-fill assumptions:

<!-- example: common-python-baselines source=docs/examples/python-research/scenarios.py region=common_baselines scenario=search interface=python role=executed -->
```python
def common_baselines(data_dir, output):
    from limen.sfd.rule_based.common_strategies import STRATEGY_NAMES, strategy_signals

    raw = hourly(data_dir, None)
    train_rows, val_rows = raw.height * 8 // 11, raw.height // 11
    evaluation = raw[train_rows + val_rows:]
    output.mkdir(parents=True)
    rows = []
    for name in STRATEGY_NAMES:
        signals = strategy_signals(evaluation, name)
        # Caller-owned one-bar trades: enter next open, exit that close, 2 bp total cost.
        net = signals[:-1] * (evaluation['close'].to_numpy()[1:] / evaluation['open'].to_numpy()[1:] - 1 - .0002)
        evaluation.select('datetime').with_columns(pl.Series('desired_position', signals)).write_parquet(output / (name + '.parquet'))
        rows.append({'strategy': name, 'mean_net_return': float(net.mean()),
                     'net': net.tolist(), 'signals': signals.tolist()})
    return {'rows': rows, 'split_rows': [train_rows, val_rows, evaluation.height],
            'round_trip_bps': 2., 'execution': 'independent next-open-to-close trades'}
```

`STRATEGY_NAMES` is an immutable ordered tuple. `STRATEGY_PARAMETERS` is a
read-only mapping of read-only parameter mappings. `strategy_signals` returns
an independent NumPy array with dtype `int8`, aligned one-to-one with input
rows. Required prices must be finite and positive. Every strategy requires
`close`; Donchian additionally requires `high` and `low`, with
`low <= close <= high`. If `datetime` is present, it must be non-null and
strictly increasing. Without it, input row order supplies chronology.

Each call starts flat with no indicator history. Short inputs are valid and
remain flat until the chosen warm-up completes; an empty input returns an
empty array. The function retains all warm-up rows and its final desired
position. It applies no execution lag, fees, fill adjustment or forced exit.
The feature adapter `limen.features.common_strategy_positions` appends that
same array as `common_strategy_position` to eager or lazy Polars frames. It
preserves the original columns and prevents downstream lazy filters or slices
from being moved ahead of the stateful calculation.

The canonical project tests check causal signal prefixes, independent SMA/momentum rules and each interface's declared economics. Native cash/fee/slippage returns are checked with a separate scalar episode calculation. Those current assertions are distinct from the frozen historical table above.

The native manifest prepares each train, validation and test split separately,
as other rule-based manifests do. It therefore resets both indicator history
and position state at every split boundary. A one-call full-window benchmark
continues its state across the whole window. Those two protocols intentionally
answer different questions: compare engines using the same supplied signal
array and compare split-level research results using the same split boundaries.

When comparing engines, generate a single signal stream once and feed it to
each adapter. Declare the timing convention, costs, sizing, rounding,
final-position handling and annualization. An engine is eligible for a timing
comparison only after its equity/return path agrees under that convention.
Instrument preprocessing separately from execution if both are timed. Faster
execution and stronger trading results are different measurements; neither
establishes the other.

## Reference sources

- [vectorbt portfolio API](https://vectorbt.dev/api/portfolio/base/) describes
  its buy-and-hold constructor and signal-based portfolio interface.
- [backtesting.py's official example](https://kernc.github.io/backtesting.py/#example)
  demonstrates the moving-average crossover family.
- [Backtrader's indicator reference](https://www.backtrader.com/docu/indautoref/)
  records EMA initialization, MACD defaults and Wilder RSI calculation.
- [John Bollinger's Bollinger Band rules](https://www.bollingerbands.com/bollinger-band-rules)
  describe the 20-period, two-standard-deviation default and distinguish band
  tags from a complete trading rule.
- [Backtrader's Donchian channel recipe](https://www.backtrader.com/recipes/indicators/donchian/donchian/)
  explains excluding the current bar from a tradable channel threshold.
- [Moskowitz, Ooi and Pedersen, Time Series Momentum](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
  investigates momentum in an instrument's own past returns at a different
  horizon and portfolio design from this baseline.
