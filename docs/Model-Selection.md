# Model choice follows the prediction contract

A model family supplies an assumption about how inputs relate to an outcome. Linear coefficients, tree partitions, sequence dynamics, latent states, and linked learning stages represent different assumptions. A research comparison holds the target meaning and assessment population explicit while evaluating those alternatives.

The [Python project](examples/python-research/README.md) supplies an ordinary estimator and its fitting function. The [Compiled project](examples/compiled-research/README.md) selects a native reference architecture in YAML. Native model contracts also determine supported calibration, execution, and reconstruction.

## Prerequisites

Both projects require their matching wheel and recorded inputs. The complete model campaign additionally uses `stats`, `boosting`, `neural`, and `hmm` extras. TabPFN has its own checkpoint and environment requirements. [Target design](Target-Design.md) establishes the outcome before the learner is chosen.

## Families make different assumptions

The native package supplies 18 concrete architecture classes, including the unfitted rule strategy. Function wrappers provide the YAML-facing entry points. The [architecture reference](Reference-Architecture.md) owns the exact parameter and reconstruction contracts.

| Family | Native choices | Prediction meaning |
|---|---|---|
| Binary classifiers and baseline | Logistic, LightGBM, RBF SVM, TabPFN, Gaussian HMM, and random binary | A binary decision, with positive-class probability where the architecture supplies it |
| Neural sequence classifiers | LSTM and one-dimensional CNN | Next-close direction from declared return windows |
| Continuous forecasts | Ridge, XGBoost, and DLinear | The declared numeric target, with its recorded units |
| Tail forecast | Entropy/VaR regressor | A signed forward-return quantile |
| Linked learning | Residual, feature-linked, pseudo-label, weight-linked, and direction/sizing | The output contract of the composed learning stages |
| Explicit strategy | Rule-based strategy | A position requested by the declared predicate policy |

DLinear separates trend and residual windows before fitting linear heads. Gaussian HMM estimates direction from fitted return-state dynamics; latent state numbers have no inherent directional meaning. Entropy/VaR predicts a return quantile and reports quantile metrics. Each interpretation remains distinct when a score is read.

## Python fitting retains the chosen estimator

The introductory experiment fits a standard scaler and Ridge model in one sklearn pipeline:

<!-- example: choice-python-estimator source=docs/examples/python-research/research.py region=fitted_scaler scenario=selection interface=python role=executed -->
```python
def fit(train, round_params):
    return make_pipeline(StandardScaler(), Ridge(alpha=round_params["alpha"])).fit(
        train.select("momentum"), train["next_return"].to_numpy()
    )
```

The fitting function consumes training observations only. The retained pipeline supplies the exact scaler and estimator used by the selected Sensor. Replacing Ridge with a caller-defined estimator also requires compatible prediction behavior, scoring, and inference preparation. UEL executes the supplied functions and records their returned metrics.

## Compiled campaigns select native architectures

The comprehensive shell command executes the actual YAML model-family cases. Each file declares its own features, target, preparation, and model parameters:

<!-- example: choice-compiled-models source=docs/examples/compiled-research/run.sh region=models scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar \
  compiled-research/common_strategies.yaml \
  compiled-research/family_ridge_regressor.yaml \
  compiled-research/family_dlinear_regressor.yaml \
  compiled-research/family_xgboost_regressor.yaml \
  compiled-research/family_random_binary.yaml \
  compiled-research/family_direction_sizing.yaml \
  compiled-research/family_rbf_svm_binary.yaml \
  compiled-research/family_gaussian_hmm_binary.yaml \
  compiled-research/family_lstm_binary.yaml \
  compiled-research/family_residual_regressor.yaml \
  compiled-research/family_cnn1d_binary.yaml \
  compiled-research/family_entropy_var_regressor.yaml \
  compiled-research/family_feature_link_binary.yaml \
  compiled-research/family_pseudo_label_binary.yaml \
  compiled-research/family_weight_link_binary.yaml
```

These cases establish execution under each declared contract. Their labels, feature windows, and populations differ, so model comparison starts by establishing which cases answer the same question. The [foundational SFD catalog](Built-In-SFDs.md) describes packaged starting definitions; the architecture reference describes the reusable learners beneath them.

The regression diagnostics reconstruct labels from recorded prices and compare retained predictions with reported root mean squared error. The quantile case also checks pinball loss and violation rate:

<!-- result: model-choice-regression-results interface=compiled scenario=search fields=families -->

These values report the supplied cases. Ranking unlike target units or horizons by raw error would change the meaning of the comparison.

## Execution and promotion follow the model output

A binary decision supports a different execution path from a signed exposure or continuous quantile. Direction/sizing emits exposure in `[-1, 1]` under its retained trade contract. Entropy/VaR retains its forecast metrics without attaching trading metrics. Explicit conversion belongs in the research definition when a forecast is interpreted as a position.

Calibration, fitted preprocessing, parallel admission, and deterministic replay vary by architecture. Optional dependencies also have separate device and environment constraints. The [supported-workflow table](Experiment-Manifest.md#supported-research-workflows) connects those decisions to recovery and inference. A completed research case and eligibility for protected final assessment have separate requirements.

## Reproducibility includes fitting and selection

A search seed controls candidate generation. Model seeds, optional backend versions, thread counts, and input identities control other parts of the fit. Trainer uses the architecture's declared reconstruction behavior and numeric tolerances when replaying selected recorded rounds.

TabPFN's optional campaign retains a real CPU run with a pinned checkpoint and environment. Its retained proof has a different execution lifecycle from the ordinary model campaign. The [optional-model instructions](examples/compiled-research/README.md#optional-model-campaign) identify the actual observation and reproduction requirements.

Continue to [rule research](Rule-Research.md) for explicit predicate strategies, then [define the experiment](Compiled-Research.md) with a chosen target, learner, and evaluation policy. [Linked models](Linked-Models.md) develops the bounded multi-stage learners after execution and timing have been established.
