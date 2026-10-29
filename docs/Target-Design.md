# The target defines the outcome being learned

A target fixes the meaning of a prediction. A next-close direction, a return over a horizon, a first barrier touch, and a completed trade outcome are different observations. Model choice, evaluation, and execution follow from that definition.

The [Python project](examples/python-research/README.md) defines labels in its preparation functions. The [Compiled project](examples/compiled-research/README.md) declares a target class with training construction and later transformation arguments. Both keep the outcome separate from the information available to the predictor.

## Prerequisites

The matching wheel and recorded-input bundle supply both projects. The target diagnostics require the `data` and `stats` extras. Feature columns, their units, and their availability follow the decisions in [feature design](Feature-Design.md).

## Outcome geometry determines the experiment

The 16 target classes cover six research purposes:

| Research purpose | Target choices | Consequence for interpretation |
|---|---|---|
| Direction or threshold exceedance | Quantile, fixed threshold, next-bar up/down, and breakout targets | A binary prediction refers to the declared event and horizon. |
| Forward magnitude | Next return and forward volatility-normalized return | Forecast units follow the declared scale or volatility denominator. |
| Normalized return description | `VolNormalizedReturnTarget` | The timing contract determines whether a label describes the current or a future observation. |
| First path outcome | `TripleBarrierTarget` | Upper, lower, and vertical barriers produce three possible labels. |
| Trade opportunity or exit quality | `TradeOutcomeTarget`, `RiskRewardRatioTarget`, and `ExitQualityTarget` | The label depends on execution economics or supplied outcome columns. |
| Baseline or supplied labels | `RandomBinaryTarget` and `IdentityTarget` | The experiment declares the baseline seed or supplied label meaning. |

The [target reference](Targets.md) gives exact horizons, units, source-column exclusions, and unavailable-tail behavior. A class name alone does not establish the outcome's decision time.

## Python labels preserve the next-bar meaning

The introductory Python experiment predicts next-bar open-to-close return. A negative shift attaches that later observed outcome to the current feature row:

<!-- example: design-python-outcome source=docs/examples/python-research/research.py region=next_return_target scenario=selection interface=python role=executed -->
```python
def prepare_split(bars, round_params):
    return features(bars, round_params["window"]).with_columns(
        (pl.col("close").shift(-1) / pl.col("open").shift(-1) - 1)
        .alias("next_return")
    ).slice(12).drop_nulls(["momentum", "next_return"])
```

Preparation executes separately within each chronological partition. Each partition loses its fixed warm-up and final unavailable label. The model receives `momentum`; `next_return` supplies the fitting outcome. Inference computes the feature without requiring a future label.

The validation backtest interprets the forecast through signed next-bar exposure and a declared round-trip cost. A different holding policy requires a corresponding comparison between the predicted outcome and the traded return.

## Compiled target state fits on training observations

The introductory YAML experiment constructs its quantile target from training ROC values. The stored cutoff then transforms validation and test observations:

<!-- example: design-compiled-outcome source=docs/examples/compiled-research/selection.yaml region=fitted_target scenario=selection interface=compiled role=executed -->
```yaml
    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: roc_1
        quantile: 0.5
      transform_params:
        shift: -1
```

The constructor receives `fit_params`, and each transformation receives `transform_params`. Parameter references can vary the target across candidates. That variation changes the research question as well as the labels used to evaluate it.

The separate Triple Barrier case sets volatility multiples of `1.5` and `1.0`, a maximum horizon of three hourly observations, and an eight-observation volatility span. Its Ridge model treats the resulting ternary labels as numeric outcomes. The reported regression error measures that declared fit; interpreting a barrier class as a trade requires an explicit position mapping and policy.

<details>
<summary>Read the complete Triple Barrier experiment</summary>

<!-- example: design-compiled-barrier source=docs/examples/compiled-research/target_triplebarrier.yaml region=__file__ scenario=search interface=compiled role=executed -->
```yaml
schema_version: '1.0'
metadata:
  name: target_triplebarrier
  mode: development
sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20241231.parquet
    split_dates:
      train_start: '2024-01-01'
      train_end: '2024-02-15'
      val_start: '2024-02-15'
      val_end: '2024-03-15'
      test_start: '2024-03-15'
      test_end: '2024-04-15'
    indicators:
    - func: limen.indicators.roc
      params:
        period: 1
        group: base
    - func: limen.indicators.returns
      params:
        group: base
    - func: limen.indicators.rolling_volatility
      params:
        column: returns
        window: 12
        group: base
    - func: limen.indicators.bollinger_bands
      params:
        group: base
    target:
      name: label
      class: limen.targets.TripleBarrierTarget
      fit_params:
        upper_multiple: 1.5
        lower_multiple: 1.0
        max_horizon: 3
        span: 8
        min_periods: 8
    scaler:
      from_params: scaler_type
    reference_architecture: limen.sfd.reference_architecture.ridge_regressor
  params:
    scaler_type:
    - robust
    alpha:
    - 1.0
    solver:
    - svd
uel:
  output_path: compiled-target-triplebarrier
  search_strategy:
    type: grid
  n_permutations: 1
  prep_each_round: true
  output_format: csv
  recording:
    predictions: true
    alignment: true
```

</details>

## Execution-derived outcomes retain their economics

`TradeOutcomeTarget` evaluates paired long and short opportunities through the shared execution engine. Fees, slippage, funding, price availability, and holding limits affect the resulting outcomes. Incomplete or unfilled opportunities remain unavailable. Direction/sizing models use completed paired outcomes and retain their learning and execution identities.

`ExitQualityTarget` requires supplied exit reason and net return. `RiskRewardRatioTarget` requires supplied capturable-breakout and drawdown values. The project diagnostics first verify rejection of missing context, then exercise explicitly identified software labels derived from recorded prices. Their provenance distinguishes those labels from observed venue exits.

## The diagnostics preserve labeled and refused cases

The Python target campaign constructs each target on training observations, transforms the three chronological partitions, and retains labeled Parquet files. Its state checks require transformation to preserve fitted target state, apart from the declared random baseline. The volatility-normalized targets also enforce their training-volatility sanity gate.

<!-- result: target-design-python-outcomes interface=python scenario=search fields=target_diagnostics.TripleBarrierTarget.status,target_diagnostics.TripleBarrierTarget.partitions[].rows,target_diagnostics.TripleBarrierTarget.partitions[].null_rows,target_diagnostics.VolNormalizedReturnTarget.status,target_diagnostics.ExitQualityTarget.supplied_context.rows -->

The Compiled project executes the corresponding YAML target campaigns. Its result records both completed predictions and explicit training-gate refusals:

<!-- result: target-design-compiled-outcomes interface=compiled scenario=search fields=target_diagnostics.TripleBarrierTarget.status,target_diagnostics.TripleBarrierTarget.rows,target_diagnostics.VolNormalizedReturnTarget.status -->

A refused target leaves no successful model result for that case. The report preserves that distinction alongside the admitted populations. Continue to [model choice](Model-Selection.md) with the target meaning, label availability, and assessment population fixed.
