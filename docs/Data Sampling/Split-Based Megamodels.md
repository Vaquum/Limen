# Split-Based Megamodels

UEL-compatible megamodel creation using multiple train/validation splits on the same dataset.

## Overview

Split-based megamodels create megamodel predictions by training multiple models on different train/validation splits of the same dataset. This approach is complementary to data sampling strategies and provides model stability through split diversity rather than data diversity.

## Core Concept

While data sampling strategies create different datasets to train on, split-based megamodels:
- Keep the same overall dataset
- Create different train/validation boundaries  
- Train separate models on each split configuration
- Average predictions for megamodel results

## uel_split_megamodel

The main utility for creating UEL-compatible split-based megamodels.

### Basic Usage

```python
from loop.utils import uel_split_megamodel
from loop.sfm.lightgbm import tradeable_regressor

results = uel_split_megamodel(
    original_data=data,
    sfm_module=tradeable_regressor,
    n_models=5,
    seed=42
)

# Access megamodel predictions
megamodel_predictions = results['megamodel_predictions']
mae_improvement = results['mae_improvement_pct']
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `original_data` | `pl.DataFrame` | Required | Klines dataset with datetime and numeric columns |
| `sfm_module` | `module` | Required | Single File Model module (e.g., tradeable_regressor) |
| `n_models` | `int` | 5 | Number of models to create with different splits |
| `split_ratios` | `tuple` | (0.7, 0.15, 0.15) | Train, validation, test split ratios |
| `n_permutations` | `int` | 2 | Number of parameter permutations per UEL run |
| `experiment_base_name` | `str` | 'split_megamodel' | Base name for UEL experiment names |
| `seed` | `int \| None` | None | Random seed for reproducible splits |

### Return Values

The function returns a dictionary containing:

| Key | Type | Description |
|-----|------|-------------|
| `megamodel_predictions` | `np.ndarray` | Averaged predictions across all models |
| `individual_predictions` | `list[np.ndarray]` | Predictions from each individual model |
| `uel_results` | `list[dict]` | Complete UEL results from each run |
| `best_models` | `list` | Best model from each UEL run |
| `megamodel_mae` | `float` | Mean Absolute Error of megamodel predictions |
| `megamodel_r2` | `float` | R² score of megamodel predictions |
| `megamodel_rmse` | `float` | Root Mean Square Error of megamodel predictions |
| `mae_improvement_pct` | `float` | Percentage improvement over best individual model |
| `individual_metrics` | `list[dict]` | Performance metrics for each model |

## UEL Integration

### Full UEL Compatibility

Split-based megamodels are fully UEL-compatible because they:

1. **Use complete UEL workflow**: Each model trains through `UniversalExperimentLoop`
2. **Preserve parameter sweeping**: Each split gets full parameter optimization
3. **Maintain all artifacts**: UEL logging, models, experiment logs preserved
4. **Work with any SFM**: Compatible with all Single File Model modules

### Multiple UEL Runs

The function performs multiple independent UEL runs:

```python
# Conceptually, this happens n_models times:
for i in range(n_models):
    shuffled_data = original_data.sample(seed=seed+i, shuffle=True)
    uel = loop.UniversalExperimentLoop(shuffled_data, sfm_module)
    uel.run(experiment_name=f'split_megamodel_{i}')
    # Extract best model and collect predictions
```

## Comparison with Data Sampling

| Aspect | Split-Based Megamodels | Data Sampling |
|--------|----------------------|---------------|
| **Data Strategy** | Same dataset, different splits | Different datasets, same splits |
| **Diversity Source** | Train/val boundaries | Data subsets/sampling |
| **UEL Integration** | Multiple full UEL runs | UEL on each dataset |
| **Computational Cost** | Higher (multiple UEL runs) | Lower (single UEL per dataset) |
| **Best For** | Model stability testing | Data representation diversity |

## Advanced Usage

### Custom Split Ratios

```python
# Custom train/val/test split
results = uel_split_megamodel(
    data,
    tradeable_regressor,
    split_ratios=(0.8, 0.1, 0.1),  # 80% train, 10% val, 10% test
    n_models=3
)
```

### Increased Parameter Exploration

```python
# More thorough parameter sweeping per model
results = uel_split_megamodel(
    data,
    tradeable_regressor,
    n_permutations=10,  # More parameter combinations tested
    n_models=3
)
```

### Combining with Data Sampling

```python
from loop.utils import random_subsets_sampling, uel_split_megamodel

# Double megamodel: data diversity + split diversity
datasets = random_subsets_sampling(data, n_samples=3, seed=42)

all_results = []
for i, dataset in enumerate(datasets):
    split_megamodel = uel_split_megamodel(
        dataset,
        tradeable_regressor,
        n_models=3,
        experiment_base_name=f'combo_{i}',
        seed=42
    )
    all_results.append(split_megamodel)

# Super megamodel: 3 datasets × 3 split models = 9 total models
```

## Performance Considerations

### Computational Cost

Split-based megamodels are computationally expensive because:
- Each model requires a full UEL run
- Parameter sweeping happens for each split
- Total UEL runs = `n_models × n_permutations`

### Memory Usage

- Stores all UEL instances and models in memory
- Individual predictions arrays for megamodel calculation
- Consider memory constraints for large datasets or many models

### Optimization Tips

1. **Start small**: Begin with `n_models=3` and `n_permutations=2`
2. **Use seeding**: Always set `seed` for reproducible results
3. **Monitor resources**: Watch memory usage with large datasets
4. **Combine strategically**: Use with data sampling for maximum diversity

## Error Handling

The function handles common issues gracefully:

- **Failed UEL runs**: Continues with successful runs, reports failures
- **Prediction errors**: Skips failed models, uses available predictions
- **Missing metrics**: Uses first available metric (MAE, RMSE, MSE)
- **Empty results**: Raises informative error if all runs fail

## Best Practices

### When to Use Split-Based Megamodels

- **Model stability testing**: Verify performance across different splits
- **Megamodel improvement**: When single models show high variance
- **SFM evaluation**: Test how well SFM generalizes to different boundaries
- **Production megamodels**: When computational cost is acceptable

### When to Prefer Data Sampling

- **Large datasets**: When computational cost is prohibitive
- **Data diversity**: When different data characteristics matter more
- **Quick experimentation**: Faster iteration cycles needed
- **Resource constraints**: Limited computational resources

### Recommended Configurations

**Development/Testing:**
```python
uel_split_megamodel(data, sfm, n_models=3, n_permutations=2)
```

**Production/Research:**
```python
uel_split_megamodel(data, sfm, n_models=5, n_permutations=5)
```

**High-Performance:**
```python
uel_split_megamodel(data, sfm, n_models=10, n_permutations=10)
```

## Implementation Notes

### SFM Compatibility

Works with any SFM that follows Loop patterns:
- Simple SFMs (linear regression, logistic regression)
- Complex SFMs (tradeable_regressor with regime models)
- Custom SFMs with proper `prep()` and `model()` functions

### Randomization Strategy

Uses different random seeds for data shuffling to create diverse train/val boundaries while maintaining the same overall dataset composition.

### Metric Selection

Automatically selects the first available standard metric (MAE, RMSE, MSE) for model ranking. Lower values indicate better models for all supported metrics.