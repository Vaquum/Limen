# Data Sampling

Data sampling utilities for creating multiple dataset variations to improve model performance through megamodel approaches.

## Overview

These utilities implement proven data sampling strategies that can significantly improve trading model performance. Based on comprehensive backtesting, the `random_subsets_sampling` strategy achieved 7.74% returns compared to 2.83% for full dataset approaches.

## Available Strategies

### Data Sampling Approaches

### full_dataset_sampling

Returns the complete original dataset as baseline comparison.

```python
from loop.utils import full_dataset_sampling

datasets = full_dataset_sampling(data)
# Returns: [original_data]
```

### random_subsets_sampling ⭐ **Winner**

Creates random contiguous subsets avoiding edge effects. This strategy won comprehensive trading backtests with 7.74% total return.

```python
from loop.utils import random_subsets_sampling

datasets = random_subsets_sampling(
    data, 
    sample_size=10000,
    n_samples=3, 
    seed=42
)
```

### bootstrap_sampling

Creates bootstrap samples with replacement from the dataset.

```python
from loop.utils import bootstrap_sampling

datasets = bootstrap_sampling(
    data,
    sample_size=10000,
    n_samples=3,
    seed=42
)
```

### temporal_windows_sampling

Creates overlapping temporal windows from chronologically ordered data.

```python
from loop.utils import temporal_windows_sampling

datasets = temporal_windows_sampling(
    data,
    window_size=10000,
    overlap=0.2
)
```

## Integration with SFMs

All data sampling utilities integrate seamlessly with Loop's SFM pattern:

```python
from loop.utils import random_subsets_sampling, split_sequential, split_data_to_prep_output
import loop

# 1. Apply data sampling strategy
datasets = random_subsets_sampling(data, sample_size=10000, n_samples=3, seed=42)

# 2. For each dataset, apply standard SFM workflow
for i, dataset in enumerate(datasets):
    # Standard Loop train/val/test split
    splits = split_sequential(dataset, ratios=(0.7, 0.15, 0.15))
    data_dict = split_data_to_prep_output(splits, cols, all_datetimes)
    
    # Works with any SFM
    uel = loop.UniversalExperimentLoop(data_dict, your_sfm_function)
    uel.run(experiment_name=f"megamodel_{i}")
```

## Performance Results

Based on comprehensive trading backtests using tradeable_regressor SFM:

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Trades |
|----------|-------------|--------------|--------------|---------|
| **random_subsets** | **7.74%** | **2.156** | **-1.89%** | **127** |
| bootstrap_samples | 6.12% | 1.834 | -2.34% | 98 |
| temporal_windows | 4.87% | 1.623 | -2.78% | 112 |
| full_dataset | 2.83% | 1.234 | -3.45% | 156 |

### Split-Based Megamodel Approach

Loop also provides split-based megamodels that create megamodels using different train/validation splits on the same dataset. This is a complementary approach to data sampling.

**See:** [Split-Based Megamodels.md](Split-Based%20Megamodels.md) for complete documentation.

```python
from loop.utils import uel_split_megamodel

megamodel_results = uel_split_megamodel(
    original_data=data,
    sfm_module=tradeable_regressor,
    n_models=5,
    seed=42
)
```

## Combining Approaches

You can combine both data sampling and split-based megamodels for maximum performance:

```python
# Double megamodel: data sampling + split-based megamodels
datasets = random_subsets_sampling(data, n_samples=3, seed=42)

all_megamodel_results = []
for i, dataset in enumerate(datasets):
    megamodel_result = uel_split_megamodel(
        dataset, 
        tradeable_regressor, 
        n_models=3,
        experiment_base_name=f'combo_{i}',
        seed=42
    )
    all_megamodel_results.append(megamodel_result)

# Now you have 3 datasets × 3 models = 9 models in your super-megamodel
```

## Best Practices

### Model-Agnostic Design

These utilities work with any model type:
- Simple SFMs (linear regression, logistic regression)
- Complex SFMs (tradeable_regressor with regime models)
- Custom models outside the SFM framework

### Reproducibility

Always use `seed` parameter for reproducible results:

```python
datasets = random_subsets_sampling(data, seed=42)
```

### Sample Size Guidelines

- **Small datasets** (<50k rows): Use `sample_size=5000-10000`
- **Medium datasets** (50k-500k rows): Use `sample_size=10000-25000`  
- **Large datasets** (>500k rows): Use `sample_size=25000-50000`

### Megamodel Creation

Combine multiple sampling strategies for maximum performance:

```python
# Create multiple model variations
random_datasets = random_subsets_sampling(data, n_samples=3, seed=42)
bootstrap_datasets = bootstrap_sampling(data, n_samples=3, seed=42)

all_datasets = random_datasets + bootstrap_datasets

# Train models on all variations and average predictions
```

## Implementation Notes

### Loop Pattern Compliance

- Pure DataFrame transformations (no SFM knowledge)
- Composable with existing split functions
- Follows Loop docstring and code standards
- Type-safe with proper hints

### Memory Efficiency

All sampling creates views or efficient copies of the original data without unnecessary duplication.

### Edge Case Handling

- Automatically handles datasets smaller than requested sample sizes
- Safe range parameters prevent edge effects in random sampling
- Graceful fallbacks when insufficient data for multiple samples