# Controlled perturbations test dependence on research choices

A perturbation changes a declared research choice and measures the resulting behavior. Population, feature set, representation, target, and learner parameters provide five distinct axes. The comparison remains interpretable when the changed axis and assessment population are recorded together.

The [Python project](examples/python-research/README.md) chooses columns and fitted transforms directly. The [Compiled project](examples/compiled-research/README.md) declares preparation choices as candidate parameters. Their worked cases retain the prepared observations needed to inspect those changes.

## Prerequisites

Both projects require the matching wheel, recorded-input bundle, and `data` and `stats` extras. The source population, target, model, and execution policy follow the [experiment definition](Compiled-Research.md). A comparison needs an explicit baseline before perturbations are selected.

## The varied axis determines the comparison

| Axis varied | Supported choices | Evidence to retain |
|---|---|---|
| Observations | Trailing windows, contiguous regime blocks, and bar thresholds | Admitted source rows, sampling policy, and resulting split geometry |
| Feature columns | Groups, conditional inclusion, and seeded Drop-N ablation | Ordered retained and dropped columns |
| Feature representation | Scaling, relative/log/difference/fractional transformations, and PCA | Training-fitted state and inference feature order |
| Target | Horizon, threshold, barrier geometry, candidate jitter, and training-label smoothing | Label meaning, candidate identity, and unavailable outcomes |
| Learner | Architecture-specific fitting and complexity parameters | Consumed parameters, fitting population, and model identity |

Population changes can also change warm-up and eligible labels. Their resulting scores require that changed denominator. A representation change can preserve rows while changing every model input. The [perturbation reference](Perturbation-Strategies.md) owns the exact primitives and parameter syntax.

## Python retains the fitted representation

The Python preparation case divides 1,440 recorded hourly observations into raw partitions of 864, 288, and 288 rows. It builds ROC and fractional features independently within each partition, applies Triple Barrier labels, and fits robust scaling on the training features.

The baseline retains four features. The compressed case removes `roc_12` and fits two PCA components. Both cases retain transformed partitions, fitted scaler statistics, PCA state where used, predictions, and regression error:

<details>
<summary>Read the complete Python preparation comparison</summary>

<!-- example: robustness-python-preparation source=docs/examples/python-research/scenarios.py region=preparation scenario=search interface=python role=executed -->
```python
def preparation(data_dir, output):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from limen.features import fractional_diff, find_min_d
    from limen.indicators import roc
    from limen.scalers import RobustScaler
    from limen.targets import TripleBarrierTarget
    from limen.utils import adf_test

    raw = hourly(data_dir, 1440)
    partitions = [raw[:864], raw[864:1152], raw[1152:1440]]
    d = find_min_d(partitions[0], 'close', d_end=.4, step=.2, threshold=.05)
    stationarity = adf_test(fractional_diff(partitions[0], d=d, cols=['close'], threshold=.05)['close_fracdiff'])
    cases = []
    for compressed in (False, True):
        name = 'compressed' if compressed else 'baseline'
        directory = output / name
        directory.mkdir(parents=True)
        target = TripleBarrierTarget(partitions[0], 'label', upper_multiple=1.5,
                                     lower_multiple=1., max_horizon=3, span=8, min_periods=8)
        columns = ['close', 'roc_3', 'roc_12', 'close_fracdiff']
        dropped = ['roc_12'] if compressed else []
        features = [column for column in columns if column not in dropped]
        prepared = []
        for partition in partitions:
            frame = fractional_diff(roc(roc(partition, period=3), period=12),
                                    d=.4, cols=['close'], threshold=.05)
            prepared.append(target.transform(frame).select('datetime', *features, 'label').drop_nulls())
        scaler = RobustScaler(prepared[0].select(features))
        scaled = [scaler.transform(frame.select(features)) for frame in prepared]
        pca = PCA(n_components=2, svd_solver='full') if compressed else None
        matrices = ([pca.fit_transform(scaled[0].to_numpy()),
                     *[pca.transform(frame.to_numpy()) for frame in scaled[1:]]]
                    if pca is not None else [frame.to_numpy() for frame in scaled])
        names = ['pc_0', 'pc_1'] if compressed else features
        model = Ridge(alpha=1., solver='svd').fit(matrices[0], prepared[0]['label'].to_numpy())
        predictions = model.predict(matrices[2])
        for split, frame, matrix in zip(('train', 'val', 'test'), prepared, matrices, strict=True):
            pl.DataFrame(matrix, schema=names, orient='row').with_columns(
                frame['datetime'], frame['label']).write_parquet(directory / (split + '.parquet'))
        cases.append({'name': name, 'features': features, 'dropped': dropped,
                      'rows': [frame.height for frame in prepared],
                      'medians': scaler.medians, 'iqrs': scaler.iqrs,
                      'pca_mean': None if pca is None else pca.mean_.tolist(),
                      'components': None if pca is None else pca.components_.tolist(),
                      'predictions': predictions.tolist(),
                      'rmse': float(np.sqrt(np.mean((predictions-prepared[2]['label'].to_numpy())**2)))})
    return {'cases': cases, 'partition_rows': [864, 288, 288], 'carried_context': False,
            'fixed_d': .4, 'weight_threshold': .05, 'adf_grid_result': d,
            'adf_stationary': bool(stationarity.stationary), 'adf_p_value': float(stationarity.p_value)}
```

</details>

This comparison changes feature ablation and PCA together. Its result measures that combined change. A separate comparison of each axis requires candidates that vary one choice at a time while preserving the others.

The training-only stationarity diagnostic tests a bounded fractional-order grid. Feature construction explicitly uses order `0.4`; the diagnostic's selected order is recorded separately. Its result describes the supplied training population and statistical test.

<!-- result: robustness-python-preparation interface=python scenario=search fields=preparation.partition_rows,preparation.cases[].name,preparation.cases[].features,preparation.cases[].dropped,preparation.cases[].rows,preparation.cases[].rmse -->

## Compiled preparation declares the candidate axes

The composable YAML experiment crosses two regularization values, two ablation counts, and two PCA states. Group selection and the ablation seed remain fixed:

<!-- example: robustness-compiled-domain source=docs/examples/compiled-research/research.yaml region=parameters scenario=search interface=compiled role=executed -->
```yaml
  params:
    C: [0.5, 1.0]
    random_state: [42]
    feature_groups: [momentum|price]
    feature_drop_count: [0, 1]
    feature_drop_seed: [7]
    auto_pca: [false, true]
    pca_k: [2]
    use_calibration: [true]
    use_threshold: [true]
```

The eight combinations include the uncompressed and compressed alternatives at each regularization value. Recorded round parameters identify which representation produced each score. Training fits the target, scaler, and PCA state before their application to later splits.

<!-- example: robustness-compiled-run source=docs/examples/compiled-research/run.sh region=preparation scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/preparation.yaml
limen run --no-progress-bar compiled-research/research.yaml
limen assess compiled-research/research.yaml --results results/dev/compiled-research
```

<!-- result: robustness-compiled-candidates interface=compiled scenario=search fields=rounds,selected_members -->

The complete result directory retains candidate-level parameters and preparation witnesses. The two selected members reflect the declared validation ranking. Their selection does not isolate the causal contribution of each preparation choice.

## Fitted scalers and per-frame transforms have different state

The five registered scaler types provide linear column policies, logistic-workflow policies, robust training statistics, rank-Gaussian mapping, and causal rolling robust scaling. Their [reference](Scalers.md) defines training state and context requirements. Causal rolling scaling also requires sufficient preceding history at inference.

Stateless Z-score, MAD, winsorization, trimming, and shifting functions operate on the supplied frame. Their per-call statistics do not provide a training-fit/apply-later contract. The [transform reference](Transforms.md) states those boundaries and the consequences of removing rows.

## Target perturbations preserve outcome provenance

Candidate target jitter uses deterministic candidate identity and its declared seed. Label smoothing applies to training labels. Observed evaluation outcomes retain their own meaning and availability. A target perturbation therefore belongs in the experiment definition and reconstruction evidence.

The label and feature choices used to select a configuration consume selection evidence. Stability across declared perturbations is evidence about those comparisons. Assessment on [successive windows](Walk-Forward.md) and [reserved final observations](Final-Holdout.md) answers additional questions about later populations.

Continue to [search](Advanced-Search.md) to choose how the declared alternatives are traversed. [Research evidence](Research-Evidence.md) develops correlation, importance, and family reports after completed candidates are available.
