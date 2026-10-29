# Linked learning stages change the information used by a fit

A linked model uses one learning stage to change another stage's features, labels, weights, or predictions. The order matters because each intermediate output has a fitting history and an availability time. Chronological out-of-sample outputs provide a different training input from predictions fitted on the same labels.

The [Python project](examples/python-research/README.md) supplies a caller-owned residual model. The [Compiled project](examples/compiled-research/README.md) supplies bounded native linked architectures in YAML. Each case records its own data, target, and learning mechanism.

## Prerequisites

Both projects require the matching wheel, recorded-input bundle, and the `data` and `boosting` extras. [Model choice](Model-Selection.md) establishes output meaning, and [trading behavior](Trading-Behavior.md) establishes the economics used when outcomes or exposure become learning inputs.

## Each link changes a specific learning input

| Learning mechanism | Native implementation | Information passed between stages |
|---|---|---|
| Residual correction | `ResidualRegressor` | Ridge error becomes the XGBoost fitting target; predictions add both components. |
| Feature linkage | `FeatureLinkedBinary` | Chronological upstream probabilities become a downstream feature. |
| Weight linkage | `WeightLinkedBinary` | Chronological upstream prediction errors determine downstream sample weights. |
| Pseudo-label learning | `PseudoLabelBinary` | A fitted teacher supplies generated training labels for a declared real observation pool. |
| Conditional sizing | `DirectionSizingModel` | Chronological direction predictions can become inputs to the sizing component. |

Each native architecture has its own admitted configuration. The [architecture reference](Reference-Architecture.md) owns preprocessing, target, source, chronology, and reconstruction requirements. Those fixed learning mechanisms determine the available connections.

## Python defines residual correction in ordinary functions

The Python case first fits a scaler and Ridge estimator. An XGBoost regressor then learns the remaining training error. Prediction adds the two fitted outputs:

<!-- example: linked-python-residual source=docs/examples/python-research/linked_cases.py region=residual_fit scenario=search interface=python role=executed -->
```python
def fit_residual(train, depth):
    from xgboost import XGBRegressor

    x = train.select('momentum').to_numpy()
    y = train['next_return'].to_numpy()
    baseline = make_pipeline(StandardScaler(), Ridge(alpha=1., solver='svd')).fit(x, y)
    residuals = y - baseline.predict(x)
    correction = XGBRegressor(n_estimators=8, max_depth=depth, learning_rate=.1,
                              random_state=42, n_jobs=1).fit(x, residuals)
    return baseline, correction


def residual_predict(fitted, frame):
    baseline, correction = fitted
    x = frame.select('momentum').to_numpy()
    return baseline.predict(x) + correction.predict(x)
```

Both learners fit only the training partition. The correction stage uses in-sample Ridge residuals, as declared by this residual-learning design. It does not create out-of-sample stacking features. The candidate comparison varies tree depth between one and two while retaining the first-stage fit and all other correction parameters.

The complete `residual_research` function partitions the recorded hourly observations before feature and label construction. Validation selects depth by mean net next-bar return with the project's 2-basis-point round-trip cost. The selected pair then predicts the test partition. Parquet artifacts retain each component's output and their sum.

<!-- result: linked-python-selection interface=python scenario=search fields=linked.candidates,linked.selected_depth,linked.partition_rows,linked.test_rmse,linked.test_net_return -->

Independent checks reconstruct features, observed labels, and the Ridge solution from recorded prices. They also verify residual fitting, component addition, validation ranking, and the selected test result. The result describes this caller-owned model and declared population.

## Native feature linkage fits chronological downstream inputs

The Compiled feature-link case uses raw return windows, a next-close-up target, and logistic upstream and downstream components. Three chronological splits produce the upstream training predictions used by the downstream classifier. The declared observation interval is 900 seconds.

<details>
<summary>Read the complete feature-link experiment</summary>

<!-- example: linked-compiled-feature source=docs/examples/compiled-research/family_feature_link_binary.yaml region=declaration scenario=search interface=compiled role=executed -->
```yaml
schema_version: '1.0'
metadata:
  name: book_feature_link_binary
  mode: development
sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_15m_20250101_20250531.parquet
    split_dates:
      train_start: '2025-01-01'
      train_end: '2025-01-15'
      val_start: '2025-01-15'
      val_end: '2025-01-18'
      test_start: '2025-01-18'
      test_end: '2025-01-22'
    indicators:
    - func: limen.indicators.window_return
      params:
        period: 1
    target:
      name: next_close_up
      class: limen.targets.NextBarUpTarget
    backtest:
      fee_bps: 7.0
      slip_bps: 3.0
      notional_rate: 0.5
    reference_architecture: limen.sfd.reference_architecture.feature_link_binary
    features:
    - func: limen.features.lag_range
      params:
        col: ret_1
        start: 0
        end: 23
    data_dict_extension:
      func: limen.sfd.reference_architecture.feature_link_binary.recorded_rows
  params:
    upstream_params:
    - C: 1.0
    downstream_params:
    - C: 1.0
    n_splits:
    - 3
    observation_interval_seconds:
    - 900
uel:
  output_path: compiled-family_feature_link_binary
  n_permutations: 1
  search_strategy:
    type: grid
  output_format: csv
  checkpoint_interval: 1
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
```

</details>

The `recorded_rows` extension binds the raw observations needed to establish label maturity and row identity. This native path admits its declared return-window features and target. Outer fitted scalers and PCA are excluded by the linked architecture's preparation contract.

Weight linkage uses the corresponding chronological upstream errors as actual sample weights. Pseudo-label learning instead requires an explicit pinned pool of real observations, with chronology and membership checks. Generated labels train the learner; evaluation retains observed outcomes. The native pseudo-label pool has its own restrictions, including its protected-research boundary.

## Direction, sizing, and allocation retain execution identity

Direction/sizing learns from completed paired long and short outcomes. Conditional sizing uses chronological direction predictions whose fitting data and outcome maturity precede each downstream block. The selected Sensor preserves the signed exposure and frozen execution contract.

Learned meta-allocation supplies a separate research path around a primary strategy. Its dynamic execution and learning behavior is restricted to fresh serial research. Resume, multiple native workers, and Trainer/Sensor promotion are excluded by that path's admission rules. [Allocation contracts](Backtest.md#learned-allocation-around-a-primary) explain its recorded opportunities, costs, funding, and fitted policy.

## The campaign preserves model-specific results

The Compiled model campaign executes the linked-family YAML files through the same CLI commands used in [model choice](Model-Selection.md). Its regression diagnostics retain the residual model's recorded test error and population:

<!-- result: linked-compiled-residual interface=compiled scenario=search fields=families.residual_regressor -->

Binary link checks independently reconstruct observed next-close labels and reconcile predictions with recorded accuracy. Learning bindings and package tests additionally verify the chronology and identity contracts specific to each native link. The [supported-workflow table](Experiment-Manifest.md#supported-research-workflows) identifies which results can continue into reconstruction or protected assessment.

Continue to [research evidence](Research-Evidence.md) to interpret these candidate results alongside their component choices, selection history, and assessment populations.
