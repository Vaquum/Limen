# `limen.calibration`

> Calibrate model probabilities and choose decision thresholds inside manifest-driven experiments.

## Canonical docs

- [Calibration](../../docs/Calibration.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md#calibration)
- [Reference Architecture](../../docs/Reference-Architecture.md)

## What this package owns

Owns probability calibration, threshold optimization, and the runtime pipeline that applies those steps to model outputs.
Does **not** own target construction, scaler fitting, model training, or downstream trade decisioning.

## Key entry points

| Entry point | Use it when | Notes |
| --- | --- | --- |
| `CalibratorProtocol`, `ThresholdOptimizerProtocol` | You need the callable contracts for custom calibration or threshold functions | Used by manifest calibration wiring and pipeline helpers. |
| `fit_calibrator` | You need to fit the configured calibrator and threshold on validation data | Returns the fitted predictor, selected threshold, and validation score. |
| `apply_calibrated_predict` | You need to apply calibration and thresholding to test predictions | Returns `_preds`, `_probs`, `optimal_threshold`, and `val_score`. |
| `sklearn_probability_calibrator` | You need scikit-learn probability calibration | Supports isotonic and sigmoid calibration. |
| `grid_threshold_optimizer` | You need a threshold optimizer over calibrated or raw probabilities | Produces the decision cutoff used for binary predictions. |

## Adjacent modules

- `limen.experiment.manifest_core` wires calibration into manifest execution.
- `limen.sfd.reference_architecture` consumes calibrated probabilities during evaluation.
- `limen.targets` creates the supervised labels that calibration evaluates against.

## Quick orientation

```text
calibration/
|-- pipeline.py     # protocol types, fit step, and calibrated prediction helper
|-- probability.py  # scikit-learn probability calibrator wrapper
`-- threshold.py    # grid threshold optimizer
```

## Things to know

- Calibration is configured on a manifest and resolved per experiment round.
- Probability calibration and threshold optimization can be enabled independently.
- Validation data is the fitting surface for calibrators and threshold choice.
- `predict()` and `evaluate()` have different result-schema guarantees; see the canonical Calibration page before changing output fields.

## Read next

- [Calibration](../../docs/Calibration.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md#calibration)
