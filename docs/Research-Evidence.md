# Research evidence connects scores to the decisions that produced them

A candidate score describes a model, an outcome, a population, and an evaluation policy. Research analysis compares those quantities while retaining the alternatives considered and the observations consumed by selection. The best recorded value is one part of that evidence.

The [Python project](examples/python-research/README.md) supplies explicit candidate analysis through `Log`. The [Compiled project](examples/compiled-research/README.md) retains native metrics, fold return tracks, and acceptance reports. Their complete artifacts preserve the rows behind the displayed summaries.

## Prerequisites

Both projects require the matching wheel and recorded inputs. Statistical analysis uses the `stats` extra. Row-level inspection additionally requires retained predictions, alignment, prices, or ledgers appropriate to the question. [Run and recover](Run-And-Recover.md) defines those recording choices before execution.

## The question determines the evidence required

| Analysis question | Evidence consumed | Interpretation |
|---|---|---|
| Whether predictions match observed outcomes | Aligned predictions and target observations | Classification, regression, or quantile quality under the declared target |
| Whether a decision has economic value | Observed outcomes by decision class and a stated trading policy | Confusion-return analysis and costed execution answer different parts of the question. |
| Which parameters relate to score variation | Completed parameter/metric rows | Correlation describes the recorded candidate population. |
| Which parameters predict variation beyond a fitted surrogate | Held-out surrogate predictions and shuffled parameter columns | Parameter importance measures the surrogate's predictive dependence. |
| Whether a selected result survives comparison with alternatives | Synchronous retained return tracks and the supplied trial inventory | DSR/PBO reports assess selection under their stated assumptions. |
| Whether recorded experiments can be compared together | Source lineage, compatible observations, and retained trial tracks | Family reports preserve admitted groups and missing or incompatible evidence. |

[Log](Log.md), [Benchmark](Benchmark.md), and the [metrics library](Standard-Metrics-Library.md) own the exact formulas, populations, and failure conditions.

## Prediction quality and execution measure different outcomes

Binary accuracy counts correct labels. Probability quality depends on the meaning of the positive class and calibration. Continuous error measures deviations in the target's units. Quantile loss and violation rate evaluate a declared return quantile.

Confusion-return analysis examines the realized outcomes associated with true and false decisions. Snapshot and event backtests apply the selected entry, exit, exposure, and cost policy. Trade counts, missing decisions, fees, funding, and terminal exposure explain how that policy produced its return. Their denominators belong beside the reported score.

## Python analyzes an explicit candidate population

The Python analysis campaign fits 30 candidates: ten momentum windows crossed with three Ridge penalties. Each candidate records validation error and the declared mean net return. The analysis names only the parameters permitted as explanatory inputs:

<!-- example: evidence-python-analysis source=docs/examples/python-research/analysis_cases.py region=analysis_results scenario=search interface=python role=executed -->
```python
def analyze_log(loop, directory, space):
    analysis = Log(uel_object=loop)
    correlation = analysis.experiment_parameter_correlation('validation_rmse', cols_to_drop=[column for column in project.completed_log(loop).columns if column not in {*space, 'validation_rmse'}],
        heads=(1.,), n_boot=50, min_n=10, random_state=42)
    importance = analysis.experiment_parameter_importance('validation_rmse', parameters=['window', 'alpha'], n_repeats=2, random_state=42)
    correlation.write_csv(directory / 'correlation.csv')
    importance.write_csv(directory / 'importance.csv')
    return correlation, importance
```

Bootstrap correlation reports parameter relationships with validation error. Held-out random-forest importance measures the change in surrogate performance when a parameter is shuffled. A parameter's association with lower forecast error can differ from its association with higher net return.

<!-- result: evidence-python-analysis interface=python scenario=search fields=analysis.correlation,analysis.importance -->

The campaign writes `correlation.csv` and `importance.csv`. Timing and unrelated metric columns are excluded from the correlation population. The [Log contract](Log.md#parameter-importance) defines minimum candidate counts, explicit input columns, held-out fitting, and unsupported inputs. The four-candidate introductory experiment supplies too few alternatives for that importance method.

## Compiled analysis retains the alternative return tracks

The Compiled analysis case compares 20 entry thresholds over two chronological folds. It declares an advisory minimum deflated Sharpe probability of `0.95` and maximum probability of backtest overfitting of `0.20`:

<details>
<summary>Read the complete analysis experiment</summary>

<!-- example: evidence-compiled-analysis source=docs/examples/compiled-research/analysis.yaml region=declaration scenario=search interface=compiled role=executed -->
```yaml
schema_version: '1.0'
metadata:
  name: recorded_analysis_population
  mode: development
sfd:
  manifest:
    type: rule_based
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20241231.parquet
    split_walk_forward:
      n_folds: 2
      test_bars: 49
      purge_bars: 4
      embargo_bars: 3
      anchored: true
    acceptance:
      min_deflated_sharpe_probability: 0.95
      max_pbo: 0.20
    indicators:
    - func: limen.indicators.window_return
      params:
        period: 1
    strategy:
      conditions:
      - id: entry
        name: positive_return
        type: threshold
        column: ret_1
        operator: '>'
        value: '{entry_return}'
      entry: entry
    reference_architecture: limen.sfd.reference_architecture.rule_based
    backtest:
      fee_bps: 7.0
      slip_bps: 3.0
      notional_rate: 0.5
  params:
    entry_return:
    - -0.003
    - -0.0027
    - -0.0024
    - -0.0021
    - -0.0018
    - -0.0015
    - -0.0012
    - -0.0009
    - -0.0006
    - -0.0003
    - -0.0
    - 0.0003
    - 0.0006
    - 0.0009
    - 0.0012
    - 0.0015
    - 0.0018
    - 0.0021
    - 0.0024
    - 0.0027
uel:
  record_execution: true
  output_path: compiled-analysis
  recording:
    alignment: true
    fold_details: true
  n_permutations: 20
  search_strategy:
    type: grid
  prep_each_round: true
  output_format: parquet
```

</details>

The resulting report names its retained trial count, synchronous assessment bars, selection statistics, and threshold verdicts:

<!-- result: evidence-compiled-acceptance interface=compiled scenario=search fields=analysis.n_trials,analysis.n_bars,analysis.n_blocks,analysis.deflated_sharpe_probability,analysis.pbo,analysis.thresholds,analysis.verdicts -->

Deflated Sharpe probability accounts for the recorded alternatives under its moment and trial-count assumptions. Probability of backtest overfitting uses combinatorially symmetric cross-validation over the retained return blocks. The two-block setting gives a coarse PBO result. Reported threshold failures remain available for interpretation while the research run completes.

These statistics consume the campaign's assessment tracks. Their trial inventory covers the supplied retained alternatives. Earlier unrecorded searches and researcher adaptation require separate accounting. [Selection-report contracts](Benchmark.md#walk-forward-selection-report) state those limitations and the exact report fields.

## Family reports preserve comparability and missing evidence

A family report accepts explicitly supplied recorded experiments. It groups eligible tracks by compatible consumed observations and available lineage evidence. Target horizon, feature warm-up, or missing rows can produce different eligible populations. The report retains those distinctions and explains unavailable statistics.

Family analysis also retains coverage of failed, missing, and incompatible attempts where evidence exists. It does not create returns for an unsuccessful fit. The [family-report contract](Benchmark.md#comparing-a-recorded-experiment-family) defines grouping, source qualification, duplicate evidence, DSR/PBO admission, and coverage fields.

## Selection freezes an interpretation of the evidence

A selection rule names the metric, its direction, eligible candidates, and tie handling. Changing that rule after reading assessment outcomes consumes those outcomes as selection evidence. A final reserved window therefore follows a frozen decision and its retained fitted state.

Continue to [selected behavior](Selected-Behavior.md) to carry that state and output meaning into Sensors and Cohorts. [Final holdout](Final-Holdout.md) develops the separately reserved assessment and its local exposure evidence.
