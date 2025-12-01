# Regime Diversified Opinion Pools

The Regime Diversified Opinion Pools (RDOP) system provides an intelligent approach to dynamic model selection and prediction aggregation using regime-based clustering. RDOP extends the Loop experiment framework by adding a model diversification layer that adapts to different market regimes, providing more robust predictions by combining multiple models intelligently.

## Overview

RDOP operates through a two-phase pipeline:

1. **Offline Phase**: Analyzes historical confusion metrics from model experiment results, clusters models into regimes using performance similarity, and performs intra-regime model diversification through PCA-based selection.

2. **Online Phase**: For new predictions, dynamically selects the appropriate regime pool based on current market conditions and aggregates predictions from multiple models within the pool.

The system integrates seamlessly with existing Loop infrastructure, consuming experiment results and producing aggregated predictions for improved stability and robustness across different market conditions.

## Offline Pipeline

### `RegimeDiversifiedOpinionPools.offline_pipeline`

Trains the RDOP system by analyzing historical model performance and creating regime-based model pools.

### Args

| Parameter             | Type              | Description                                                                 |
|-----------------------|-------------------|-----------------------------------------------------------------------------|
| `confusion_metrics`   | `pd.DataFrame`    | Pandas dataframe with experiment confusion metrics (output from `uel.experiment_confusion_metrics`) |
| `perf_cols`          | `list[str]`       | Performance column names to use for filtering and clustering (default: standard metric columns) |
| `iqr_multiplier`      | `float`           | IQR multiplier for outlier filtering (default: 3.0) |
| `target_count`        | `int`             | Target number of models to select per regime (default: 100) |
| `n_pca_components`    | `int`             | Number of PCA components for diversification (default: None, automatic) |
| `n_pca_clusters`      | `int`             | Number of clusters for PCA-based selection within regimes (default: 8) |
| `k_regimes`           | `int`             | Number of regimes to detect via clustering (default: 6) |

### Returns

Returns `pl.DataFrame` containing all selected models across regimes with added 'regime' column indicating regime membership.

### Processing Steps

1. **Sanity Filtering**: Removes models with null/invalid performance metrics
2. **Outlier Filtering**: Removes statistically extreme outliers using IQR method
3. **Regime Clustering**: Groups models by performance similarity using K-means clustering
4. **Intra-Regime Diversification**: Uses PCA-space clustering to select diverse representative models within each regime

### Error Handling

- If all models fail sanity checks: Returns original metrics with regime 0 (logs warning)
- If all models removed by outlier filtering: Falls back to sanity-filtered metrics (logs warning)

## Online Pipeline

### `RegimeDiversifiedOpinionPools.online_pipeline`

Runs predictions through the trained regime pools and aggregates results.

### Args

| Parameter                | Type             | Description                                                                |
|--------------------------|------------------|----------------------------------------------------------------------------|
| `data`                   | `pl.DataFrame`   | Prediction dataset with standard klines columns |
| `aggregation_method`     | `str`            | Aggregation method: 'mean', 'median', or 'majority_vote' (default: 'mean') |
| `aggregation_threshold`  | `float`          | Threshold for binary classification (default: 0.5) |

### Returns

Returns `pl.DataFrame` with original data plus aggregated prediction columns:

- `regime_k_prediction`: Aggregated prediction for regime k using specified aggregation method
- Individual regime predictions are available for analysis and risk management

### Processing Steps

1. **Model Pool Iteration**: Runs predictions through all models in each trained regime pool
2. **Intra-Regime Aggregation**: Combines predictions within each regime using specified method
3. **Regime Identification**: Assigns predictions to market regimes (currently uses regime 0 as placeholder)

### Aggregation Methods

- **`mean`**: Average prediction across all models in the regime pool
- **`median`**: Median prediction for robustness against outliers
- **`majority_vote`**: Majority voting for categorical predictions

## Initialization

### `RegimeDiversifiedOpinionPools.__init__`

Creates the RDOP system instance.

### Args

| Parameter      | Type               | Description                                                                |
|----------------|--------------------|----------------------------------------------------------------------------|
| `sfm`          | `SingleFileModel`  | SFM to use for individual model experiments (must match offline training) |
| `random_state` | `int`              | Random state for reproducible clustering and model selection (default: 42) |

### Notes

- Manifest is automatically extracted from SFM during initialization
- Supports SFMs with or without custom manifest functions
- Thread-safe for parallel operation after training

## Integration Points

### With Universal Experiment Loop

RDOP extends UEL workflows by providing model-level diversification:

```python
# Standard UEL produces individual model results
uel = UniversalExperimentLoop(data=train_data, single_file_model=your_sfm)
uel.run('training_experiment', n_permutations=500)

# RDOP consumes UEL results for regime-aware aggregation
rdop = RegimeDiversifiedOpinionPools(your_sfm)
offline_result = rdop.offline_pipeline(uel.experiment_confusion_metrics, k_regimes=3)

# RDOP provides more stable predictions than individual models
online_result = rdop.online_pipeline(production_data)
```

### With Experiment Manifest

Fully supports manifest-based SFMs with Universal Split-First architecture:

```python
# Manifest-based SFM works seamlessly with RDOP
if hasattr(your_sfm, 'manifest'):
    manifest = your_sfm.manifest()
    uel.run('experiment', manifest=manifest)  # RDOP will use same manifest
```

## Architecture Benefits

### Improved Stability
- Reduces overfitting by combining multiple models
- Adapts to different market regimes automatically
- Provides more robust predictions than individual models

### Enhanced Interpretability
- Clear separation of models by performance similarity
- Regime identification helps understanding prediction contexts
- Aggregated confidence scores available for risk management

### Production Readiness
- Seamless integration with existing Loop infrastructure
- Efficient caching prevents repeated expensive operations
- Graceful error handling prevents prediction pipeline failures

## Example: Complete Workflow

```python
import loop
from loop import sfm
from loop.regime_diversified_opinion_pools import RegimeDiversifiedOpinionPools

# Step 1: Train individual models (existing Loop workflow)
uel = loop.UniversalExperimentLoop(data=train_data, single_file_model=sfm.lightgbm.tradeable_regressor)
uel.run('lgboost_training', n_permutations=1000, prep_each_round=True)
confusion_df = uel.experiment_confusion_metrics

# Step 2: Train RDOP diversification system
rdop = RegimeDiversifiedOpinionPools(sfm.lightgbm.tradeable_regressor)
regime_pools = rdop.offline_pipeline(
    confusion_df,
    k_regimes=5,                  # 5 different market regimes
    target_count=25,              # 25 models per regime
    iqr_multiplier=2.5           # Moderate outlier filtering
)

# Step 3: Production predictions with regime-aware aggregation
production_predictions = rdop.online_pipeline(
    data=live_kline_data,
    aggregation_method='median'   # Robust against outliers
)

# RDOP provides more stable, regime-adapted predictions
# Access aggregated predictions by regime
regime_0_predictions = production_predictions['regime_0_prediction']
regime_1_predictions = production_predictions['regime_1_prediction']
# etc.
```

The RDOP system enhances Loop's predictive capabilities by adding intelligent model diversification and regime awareness to the standard experimental workflow.
