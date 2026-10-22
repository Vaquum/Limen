# Carry selected behavior forward

Selection freezes a research decision. Inference should use the behavior that decision evaluated, with its preparation, units, cutoff and unavailable rows preserved.

The [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) use the same recorded input bundle. Their methodology, units and evaluation populations remain explicit; their scores are not a controlled comparison.

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
limen run --no-progress-bar \
  compiled-research/helper_indicators.yaml \
  compiled-research/helper_features.yaml \
  compiled-research/helper_scalers.yaml \
  compiled-research/conditional_features.yaml
```

`assessment.json` names the selected recorded rounds and bound members. `predictions.parquet` retains their Cohort decisions, probabilities where available, timestamps, reasons and actual availability. Reconstruction refits the recorded training definition and checks replayed metrics; it does not restore arbitrary saved weights.

<!-- result: selected-compiled-result interface=compiled scenario=search fields=rounds,selected_members -->

## Calibration consumes evidence

Probability calibration and threshold selection use the declared validation population. Comparing their four on/off modes explains what state was learned and retained; it does not create four untouched assessments:

### Python Research calibration

<!-- result: selected-python-calibration interface=python scenario=calibration fields=decisions,validation_reused -->

### Compiled Research calibration

<!-- result: selected-calibration-result interface=compiled scenario=calibration fields=modes -->

The [calibration reference](Calibration.md) explains fitted-state reuse and each supported method. [Sensor](Trainer.md) defines construction, output shape, guards and failure behavior. [Cohort](Cohort.md) distinguishes Python aggregation, YAML assessment and optional native component selectors. Those optional Python component APIs are not part of Compiled authoring.

Continue to [final assessment](Final-Holdout.md) with selection already frozen. The protected branch must have been reserved before its research began.
