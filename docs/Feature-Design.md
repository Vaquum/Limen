# Features define the information available to a model

A feature expresses a research hypothesis as information available at a decision time. Momentum describes a price change, volatility describes dispersion, and a regime label describes a configured market state. Their values, units, and availability determine what a fitted model can learn and when its prediction can be used.

The [Python project](examples/python-research/README.md) constructs features in ordinary functions. The [Compiled project](examples/compiled-research/README.md) declares indicators and features in YAML. Both projects retain recorded observations and explicit preparation boundaries.

## Prerequisites

The matching wheel and recorded-input bundle supply the complete projects. The preparation diagnostics require the `data` and `stats` extras. The research population and its bar clock are established in [historical data](Historical-Data.md) and [data bars](Data-Bars.md).

## The hypothesis determines the feature family

The package provides 117 indicator exports and 120 feature-helper exports. Those counts include related calculations and utilities. The research decision concerns the information represented by a feature family and the observations required to compute it.

| Information represented | Relevant families | Required observations |
|---|---|---|
| Direction and persistence | Momentum, moving averages, lag ranges, and cross-horizon agreement | Ordered prices and sufficient preceding history |
| Dispersion and tail behavior | OHLC volatility estimators, semivariance, jump proxies, and tail intensity | The declared price fields and rolling windows |
| Participation and flow | Volume, maker/taker shares, trade density, BVC, and VPIN | Volume and the specific recorded flow fields |
| Calendar context | Cyclical time, funding hours, and prior-only seasonal normalization | UTC observation time and a declared calendar convention |
| Structure and market state | Breakout, line, regime, and candle-structure features | Their source fields, observation windows, and causal interpretation |
| Trade-scale structure | Conserved Flux Renormalization | Ordered individual trade observations |

Bar-derived flow measures describe information recoverable from bars. An order-book hypothesis requires its own observed inputs. The [indicator](Indicators.md) and [feature](Features.md) references own the individual input and output contracts.

## Python preparation states the calculation directly

The introductory Python experiment uses one decimal-valued momentum feature. Its denominator is the close `window` observations earlier. The same function prepares research observations and later inference history:

<!-- example: design-python-momentum source=docs/examples/python-research/research.py region=momentum_feature scenario=selection interface=python role=executed -->
```python
def features(bars, window):
    return bars.with_columns(
        (pl.col("close") / pl.col("close").shift(window) - 1).alias("momentum")
    )
```

A window of four quarter-hour bars represents a different history from four activity bars. The resulting feature also differs in units from Limen's percent-valued `roc`. Feature names and units therefore remain part of the experiment definition.

## Compiled preparation declares the feature dependencies

The composable Compiled experiment calculates its ROC indicator before these lagged and fractionally differentiated features. Group names allow preparation to admit the declared momentum and price families:

<!-- example: design-compiled-features source=docs/examples/compiled-research/research.yaml region=features scenario=search interface=compiled role=executed -->
```yaml
    features:
      - func: limen.features.lag_range
        params:
          col: roc_1
          start: 1
          end: 3
          group: momentum
      - func: limen.features.fractional_diff
        params:
          d: 0.4
          cols: [close]
          threshold: 0.05
          group: price
```

The lag range consumes an existing `roc_1` column. Fractional differentiation consumes close prices and retains its declared order and truncation threshold. Changing those choices changes the model input and its warm-up population. The [fractional differentiation reference](Fractional-Differentiation.md) develops training-only order selection and the resulting columns.

Native preparation preserves feature order, excludes declared target-source columns, and retains fitted transformation state for inference. Training-fitted scaling and principal component analysis are developed in [robustness research](Robustness-Research.md).

## Stateful regimes retain a chronological policy

The seven-label market-regime classifier provides batch and streaming evaluation through the same implementation. Each series retains independent rolling history, prior-only ranks, confirmation counts, and breakout state. Its JSON snapshot preserves the bounded state required to continue a stream under the same configuration and implementation.

The Python helper campaign applies an explicit configuration version to the recorded source:

<!-- example: design-python-regimes source=docs/examples/python-research/helper_cases.py region=regime_labels scenario=search interface=python role=executed -->
```python
def regime_labels(raw):
    from limen.features.classify_market_regime import RegimeConfig
    bars = raw.with_columns(pl.lit('recorded-spot').alias('series_id'))
    return features.classify_market_regime(bars, config=RegimeConfig(version='baseline-v1'), include_audit=True)
```

The returned table joins to source bars by `series_id` and UTC `datetime`. Initialization and unavailable observations remain explicit. Trend, range, and breakout labels describe the configured rule policy. The stress field measures ranked volatility conditions. Validation of those hypotheses belongs to the research comparison.

Compiled `helper_regimes.yaml` exercises the separate categorical regime-feature helpers declared in that file. The [stateful regime contract](Features.md#deterministic-market-regimes) defines the seven-label API, its streaming state, and its output fields.

## Availability determines admission into inference

A current-bar close becomes available when that bar completes. Rolling features also require preceding completed observations. Training and inference must agree on both the numerical calculation and the history presented to it.

The line helpers distinguish completed-line information from retrospective active-span counts. `include_research_only=False` excludes those active-span columns from grouped line features. The crash-reversal helper has a separate next-row date dependency under its default structural rule. Those contracts determine the appropriate decision clock; splitting observations alone does not establish within-split causality.

The recorded helper campaign reports the calculations exercised and retains their output columns. These counts describe diagnostic coverage on the supplied observations:

<!-- result: feature-design-python-coverage interface=python scenario=search fields=helper_diagnostics.counts -->

<!-- result: feature-design-compiled-coverage interface=compiled scenario=search fields=helper_diagnostics -->

The helper tables and retained files support inspection of warm-up, missing values, and added columns. A feature comparison additionally needs a common target, assessment population, and selection rule. Continue to [target design](Target-Design.md) to define the outcome learned from those inputs.
