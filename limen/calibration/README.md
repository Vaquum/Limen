# `limen.calibration`

> Calibrate model probabilities and choose decision thresholds inside manifest-driven experiments.

## Canonical docs

- [Calibration](../../docs/Calibration.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md#calibration)
- [Reference Architecture](../../docs/Reference-Architecture.md)

## In the research projects

The [Python project](../../docs/examples/python-research/README.md) calibrates an ordinary fitted classifier and chooses a threshold before Sensor construction. The [Compiled project](../../docs/examples/compiled-research/README.md) declares the same four calibration/threshold choices in its Manifest. Both `calibration` scenarios disclose reuse of validation observations.

## What this package owns

Owns probability calibration, threshold optimization, and the runtime pipeline that applies those steps to model outputs.
Does **not** own target construction, scaler fitting, model training, or downstream trade decisioning.

## Key entry points

| Entry point | Use case | Notes |
| --- | --- | --- |
| `CalibratorProtocol`, `ThresholdOptimizerProtocol` | Callable contracts for custom calibration or threshold functions | Used by manifest calibration wiring and pipeline helpers. |
| `fit_calibrator` | Fit the configured calibrator and threshold on validation data | Returns the fitted predictor, selected threshold, and validation score. |
| `apply_calibrated_predict` | Apply calibration and thresholding to test predictions | Returns `_preds`, `_probs`, `optimal_threshold`, and `val_score`. |
| `sklearn_probability_calibrator` | Scikit-learn probability calibration | Supports isotonic and sigmoid calibration. |
| `grid_threshold_optimizer` | Threshold optimizer over calibrated or raw probabilities | Produces the decision cutoff used for binary predictions. |

## Adjacent modules

- `limen.experiment.manifest_core` wires calibration into manifest execution.
- `limen.sfd.reference_architecture` consumes calibrated probabilities during evaluation.
- `limen.targets` creates the supervised labels that calibration evaluates against.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `pipeline.py` | protocol types, fit step, and calibrated prediction helper |
| `probability.py` | scikit-learn probability calibrator wrapper |
| `threshold.py` | grid threshold optimizer |

<!-- reference: package-calibration-layout source=limen/calibration/README.md region=layout:package role=layout -->
```text
calibration/
├── README.md
├── __init__.py
├── _objective_threshold.py
├── pipeline.py
├── probability.py
└── threshold.py
```

## Things to know

- Calibration is configured on a manifest and resolved per experiment round.
- Probability calibration and threshold optimization can be enabled independently.
- Validation data is the fitting surface for calibrators and threshold choice.
- `predict()` and `evaluate()` have different result-schema guarantees; see the canonical Calibration page before changing output fields.

## Read next

- [Calibration](../../docs/Calibration.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md#calibration)
