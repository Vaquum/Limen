# Common strategy baselines

`common_strategies` is a manifest-backed, rule-based Single-File Decoder (SFD)
that evaluates eight fixed long/flat baselines through Limen's existing
`RuleBasedManifest` and `RuleBasedStrategy`. It spans passive exposure, trend
following, momentum and mean reversion. These distinct holding patterns make
the catalogue useful both for strategy research and for checking backtest
engines against identical position requests.

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

Use the native template with the ordinary CLI:

```bash
limen init common-strategies.yaml --template common_strategies
limen validate common-strategies.yaml
limen run common-strategies.yaml
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

For a fully offline example from a repository checkout, run the Python SFD on
the recorded hourly fixture. The Python manifest's default split uses the
standard `8:1:2` row-count ratio; the CLI template uses its pinned date windows.
This evaluates all eight candidates and saves the ordinary UEL result artifacts:

```python
from pathlib import Path

import polars as pl
from limen import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.sfd.foundational_sfd import common_strategies

bars = pl.read_parquet("tests/fixtures/spot_1h_20240101_20241231.parquet")
uel = UniversalExperimentLoop(
    data=bars,
    sfd=common_strategies,
    search_strategy=GridStrategy(ParamDomain(common_strategies.params())),
    experiment_dir=Path("experiments/common-strategies"),
)
uel.run(
    experiment_name="common-strategies",
    n_permutations=8,
    record_execution=True,
    progress_bar=False,
)
```

`params()` has one categorical axis, `strategy`. A candidate chooses one
complete fixed configuration, avoiding combinations of irrelevant parameters
from different strategies. The result includes each split's position rate,
entry count, backtest metrics, executed-trade count and mean compounded net
PnL per executed trade. A strategy with no executed trades has a zero trade
count and an undefined mean trade PnL; it is not evidence of a zero-risk return.

## Pure signal API and engine comparisons

The same implementation is available independently of UEL:

```python
from limen.sfd.rule_based.common_strategies import (
    STRATEGY_NAMES,
    STRATEGY_PARAMETERS,
    strategy_signals,
)

positions = strategy_signals(bars, "rsi_mean_reversion")
assert len(positions) == bars.height
settings = dict(STRATEGY_PARAMETERS["rsi_mean_reversion"])
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
