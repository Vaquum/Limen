# Carry selected behavior forward

Selection freezes a research decision. Inference should use the behavior that decision evaluated, with its preparation, units, cutoff and unavailable rows preserved.

Continue with the evaluated objects returned by the [Python project](examples/python-research/README.md), or the completed result directory of the [Compiled project](examples/compiled-research/README.md). The handoff preserves different kinds of state.

## Prerequisites

Retain the completed research definition and its selection rule. Python needs the evaluated fitted objects; Compiled Research needs its original YAML and result directory. Calibration also needs the validation population it consumes.

## Python Research

The project selects the best validation return separately for each momentum window, breaking ties by alpha. It retains every fitted pipeline and wraps the selected objects in `Sensor.from_model`; `Cohort.from_python` averages their compatible next-bar return forecasts. No fit is performed after selection. The complete [Python project](examples/python-research/README.md) returns that Cohort, its loop, selected parameters and retained objects in the same process.

The `calibration` variant uses an arbitrary fitted classifier and selected threshold through `Sensor.from_python`. The preparation and output adapters remain caller-owned. A probability, a binary decision and a return forecast have different meanings; an aggregator must preserve or explicitly map them.

<!-- result: selected-python-result interface=python scenario=selection fields=selected,retained_models,inference_rows,fitting_cutoff,selection_cutoff -->

## Compiled Research

The research declaration fixes selection before `assess` reads the finished run. Run the declaration and apply its validation ranking through the CLI:

<!-- example: selected-compiled-run source=docs/examples/compiled-research/run.sh region=preparation scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/preparation.yaml
limen run --no-progress-bar compiled-research/research.yaml
limen assess compiled-research/research.yaml --results results/dev/compiled-research
```

`assessment.json` names the selected recorded rounds and bound members. `predictions.parquet` retains their Cohort decisions, probabilities where available, timestamps, reasons and actual availability. Reconstruction refits the recorded training definition and checks replayed metrics; it does not restore arbitrary saved weights.

<!-- result: selected-compiled-result interface=compiled scenario=search fields=rounds,selected_members -->

## Output meaning determines compatible composition

A Sensor carries a fitted prediction mechanism, preparation, output meaning, and relevant cutoffs. Fitting cutoff identifies observations that fitted state. Selection cutoff also includes observations used to choose that state. Actual availability then describes when each later decision can be acted upon.

| Composition path | Admitted inputs | Aggregation and identity |
|---|---|---|
| Python mean | Compatible continuous, signed, or binary outputs | Mean preserves units and signed magnitude under a common output meaning. |
| Python vote | Exact binary decisions | Majority vote resolves ties to zero and carries no calibrated probability. |
| Python custom | Explicitly mapped compatible outputs | The supplied mapper and aggregate define meaning and retain strict decision alignment. |
| Native homogeneous ordinary Cohort | Compatible Sensors from one architecture with retained artifact identity | Architecture hints select probability weighting or majority vote; ordinary regressor predictions vote up above zero. |
| Native mixed or fold-containing Cohort | Admitted Sensors declaring compatible native binary outputs | Probability weighting requires every member's probability declaration; other admitted sets vote on binary decisions. |
| Native explicit membership | Admitted rounds across experiments or qualified folds | Original experiment, round, and fold identities remain bound to each member. |

Timestamp cardinality and alignment are part of compatibility. Python composition preserves unavailable-member blocking and the latest known member availability. Native composition follows its separately documented failure and probability policy. [Cohort](Cohort.md) owns those exact behaviors.

## Reliability weighting consumes observed correctness evidence

The Bayesian Python Cohort uses a Beta correctness prior and supplied observed binary outcomes to fix each member's historical reliability weight. Evidence names the member, prediction timestamp, later outcome availability, prediction, observed outcome, and source. Generated labels do not supply observed correctness evidence.

The evidence availability extends the Cohort's selection cutoff. Weights remain fixed until a new explicit construction admits additional evidence. Correlated errors and changing regimes affect interpretation of historical correctness. The aggregate is a weighted vote and reports no calibrated probability. [Bayesian composition](Cohort.md#bayesian-reliability-voting) defines the evidence schema and posterior calculation.

## Reconstruction checks the evidence actually present

Trainer reconstructs native YAML rounds from their original definition and recorded parameters. With `results.csv`, it compares matching eligible numeric replay metrics under the architecture's tolerances. Metrics outside that intersection remain outside the comparison. Without the CSV, the component API skips metric validation and creates Sensors from the admitted reconstruction artifacts.

The complete CLI selection workflow retains the CSV and invokes the metric comparison. Qualified fold promotion additionally needs its original witnesses. Reconstruction, live caller-owned state, and protected final assessment therefore retain distinct evidence requirements. The [workflow table](Experiment-Manifest.md#supported-research-workflows) connects those requirements to each mode.

## Calibration consumes evidence

Probability calibration and threshold selection use the declared validation population. Comparing their four on/off modes explains what state was learned and retained; it does not create four untouched assessments:

### Python Research calibration

<!-- result: selected-python-calibration interface=python scenario=calibration fields=decisions,validation_reused -->

### Compiled Research calibration

<!-- result: selected-calibration-result interface=compiled scenario=calibration fields=modes -->

The [calibration reference](Calibration.md) explains fitted-state reuse and each supported method. [Sensor](Trainer.md) defines construction, output shape, guards and failure behavior. [Cohort](Cohort.md) distinguishes Python aggregation, YAML assessment and optional native component selectors. Those optional Python component APIs are not part of Compiled authoring.

Continue to [final assessment](Final-Holdout.md) with selection already frozen. The protected branch must have been reserved before its research began.
