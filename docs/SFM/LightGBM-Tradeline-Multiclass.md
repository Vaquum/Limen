# LightGBM Tradeline Multiclass: Complete Implementation Documentation

## Table of Contents
1. [Overview](#overview)
2. [Performance Summary](#performance-summary)
3. [Architecture](#architecture)
4. [Implementation Guide](#implementation-guide)
5. [Key Components](#key-components)
6. [Configuration Parameters](#configuration-parameters)
7. [Data Flow](#data-flow)
8. [Backtesting Results](#backtesting-results)
9. [Deployment Considerations](#deployment-considerations)

## Overview

LightGBM Tradeline Multiclass is a line-detection based trading system that identifies linear price movements ("lines") in historical data to predict future price direction. It's a multiclass classifier that outputs three signals: No Trade (0), Long (1), or Short (2).

### Key Innovation
The system detects significant linear price movements (Q75 - top 25% by height) and uses their patterns to predict future profitable moves. Unlike traditional technical indicators, it focuses on the geometry of price action.

## Performance Summary

### Backtesting Results (Apr-Sep 2024)
| Metric | Value |
|--------|--------|
| **Total Return** | **297.9%** |
| **Test Period** | April 1 - September 30, 2024 (6 months) |
| **Number of Trades** | 341 |
| **Win Rate** | 63.3% |
| **Model Accuracy** | 83.8% |

### Key Finding
The parameter sweep revealed that **model accuracy does NOT correlate with trading returns**. The P75/P75 configuration with 83.8% accuracy achieved 297.9% returns, while P85/P85 with 88.9% accuracy only achieved 172.6% returns.

## Architecture

### 1. Line Detection System
The system identifies linear price movements by:
- Scanning all possible price segments up to 48 hours
- Filtering movements > 0.3% (min_height_pct)
- Applying quantile filtering (default Q75 - top 25% of movements)

### 2. Threshold Calibration
- Dynamically calibrates profit thresholds using percentiles of ALL detected lines
- P75 = 75th percentile of line heights ≈ 3.05%
- Separate calibration for long and short thresholds

### 3. Model Architecture
- **Base Framework**: LightGBM gradient boosting
- **Calibration**: Isotonic calibration for probability estimates
- **Class Balancing**: Automatic class weights for imbalanced data

## Key Features

- **Line-based Features**: Detects and analyzes linear price movements over time
- **Quantile Filtering**: Focuses on significant moves using configurable quantile thresholds (default Q75)
- **Dynamic Thresholds**: Calibrates profit targets based on percentiles of historical price movements
- **LightGBM Classifier**: Uses gradient boosting with isotonic calibration for probability estimates
- **Class Balancing**: Handles imbalanced data with automatic class weighting

## Parameters

### Line Detection Parameters
- `quantile_threshold`: [0.60, 0.70, 0.75, 0.80, 0.85] - Filters lines by height percentile
- `min_height_pct`: [0.003] - Minimum 0.3% price movement to qualify as a line
- `max_duration_hours`: [48] - Maximum line duration in hours

### Threshold Calibration
- `long_threshold_percentile`: [65, 75, 85] - Percentile for long profit targets
- `short_threshold_percentile`: [65, 75, 85] - Percentile for short profit targets

### LightGBM Parameters
- `num_leaves`: [31, 63]
- `learning_rate`: [0.05, 0.1]
- `min_child_samples`: [20, 40]
- `lambda_l1`: [0, 0.1]
- `lambda_l2`: [0, 0.1]
- `n_estimators`: [500]

### Other Parameters
- `use_calibration`: [True] - Isotonic calibration for probability estimates
- `lookahead_hours`: [48] - Future window for labeling

## Feature Engineering

### Temporal Features
- Hour of day
- Day of week

### Price Features
- Returns: 1h, 6h, 12h, 24h, 48h
- Acceleration: 6h, 24h
- Distance from 24h high/low
- Position in 24h range
- Volatility: 6h, 24h
- Volume expansion ratio

### Line-based Features
- Active lines count
- Hours since last significant move
- Line momentum (6h)
- Trending score
- Reversal potential

## Label Generation

The model creates three classes based on future price movements:
- **Class 0 (No Trade)**: Future moves don't meet thresholds
- **Class 1 (Long)**: Future upward move exceeds long threshold
- **Class 2 (Short)**: Future downward move exceeds short threshold

Thresholds are dynamically calibrated using percentiles of all detected lines.

## Performance Notes

### Computational Complexity
- Line detection: O(n × m) where n = data points, m = max_duration
- Feature computation: O(n × k) where k = number of lines
- The model includes progress logging for long-running computations

### Key Findings from Backtesting
- P75/P75 configuration achieved 297.9% returns over 6 months
- Higher model accuracy doesn't guarantee better trading returns
- Optimal performance balances signal quality with trading frequency

## Usage Example

```python
from loop import sfm
from loop import UniversalExperimentLoop

# Load your data
data = get_btc_hourly_data()  # Returns Polars DataFrame

# Run parameter sweep
uel = UniversalExperimentLoop(
    data=data,
    single_file_model=sfm.lightgbm.tradeline_multiclass
)

results = uel.run(
    experiment_name="tradeline_sweep",
    n_permutations=100,
    prep_each_round=True
)
```

## Implementation Details

### Data Requirements
- Requires OHLCV data with datetime index
- Minimum 48 hours of historical data for feature computation
- Best with several months of data for training

### Memory Optimization
- Supports caching of computed line features when using UEL
- Progress logging helps monitor long computations
- Efficient Polars-based data processing

## Configuration Parameters

```python
CONFIG = {
    # Line detection parameters
    'quantile_threshold': [0.60, 0.70, 0.75, 0.80, 0.85],
    'min_height_pct': [0.003],
    'max_duration_hours': [48],
    
    # Threshold calibration
    'long_threshold_percentile': [65, 75, 85],
    'short_threshold_percentile': [65, 75, 85],
    
    # LightGBM parameters
    'num_leaves': [31, 63],
    'learning_rate': [0.05, 0.1],
    'feature_fraction': [0.9],
    'bagging_fraction': [0.8],
    'min_child_samples': [20, 40],
    'lambda_l1': [0, 0.1],
    'lambda_l2': [0, 0.1],
    'n_estimators': [500],
    
    # Other parameters
    'use_calibration': [True],
    'calibration_method': ['isotonic'],
    'lookahead_hours': [48]
}
```

## Data Flow

1. **Raw Data** → Hourly OHLCV bars with datetime
2. **Line Detection** → Find all price lines up to 48h duration
3. **Quantile Filtering** → Keep top 25% of lines by height (Q75)
4. **Threshold Calibration** → Calculate percentiles of ALL lines
5. **Feature Engineering** → Compute 32 technical features
6. **Label Creation** → Assign multiclass labels based on future moves
7. **Model Training** → LightGBM with isotonic calibration
8. **Prediction** → Output class probabilities for No Trade/Long/Short

## Backtesting Results

### Parameter Sweep Results (Top 5 by Returns)

| Model | Accuracy | Returns | Trades | Win Rate | Thresholds |
|-------|----------|---------|--------|----------|------------|
| P75/P75 | 0.838 | 297.9% | 341 | 63.3% | Long: 3.05%, Short: 3.05% |
| P70/P70 | 0.797 | 246.2% | 389 | 60.8% | Long: 2.42%, Short: 2.42% |
| P65/P65 | 0.754 | 226.1% | 433 | 59.6% | Long: 1.96%, Short: 1.96% |
| P80/P85 | 0.875 | 180.3% | 285 | 55.5% | Long: 3.83%, Short: 4.75% |
| P85/P85 | 0.889 | 172.6% | 232 | 54.3% | Long: 4.75%, Short: 4.75% |

### Analysis
- **Correlation between accuracy and returns**: -0.182 (weak negative)
- Lower threshold models (P65-P75) generate more trading opportunities
- More frequent wins at lower thresholds compound faster
- The "sweet spot" P75 balances signal quality with frequency

## Deployment Considerations

### 1. Infrastructure Requirements
- **Compute**: High during line detection phase (O(n×m) complexity)
- **Memory**: ~1GB for line storage and features
- **Latency**: Line computation can take minutes for large datasets
- **Data**: Minimum 48 hours history, ideally several months

### 2. Optimization Strategies
- Cache computed line features when running parameter sweeps
- Use smaller data windows for real-time inference
- Pre-compute lines during off-hours for production

### 3. Risk Controls
- Monitor threshold calibration stability
- Track line detection counts over time
- Alert on significant changes in line patterns

### 4. Model Management
- Retrain monthly to capture evolving patterns
- Monitor accuracy vs returns relationship
- Consider separate models for different market regimes

## Key Success Factors

1. **Geometric Pattern Recognition**: Lines capture price action geometry better than point indicators
2. **Dynamic Calibration**: Thresholds adapt to market conditions
3. **Optimal Balance**: P75 configuration balances quality and frequency
4. **Compounding Effect**: More frequent smaller wins compound to larger returns

## Future Enhancements

1. **Asymmetric Thresholds**: Test different percentiles for long vs short
2. **Regime Awareness**: Adapt thresholds based on market volatility
3. **Multi-Timeframe Lines**: Combine hourly with daily line patterns
4. **Exit Optimization**: Use lines for dynamic exit targets

## Model Uniqueness

### Why Lines Work
Traditional indicators focus on point-in-time calculations. Lines capture the full geometry of price movements:
- **Duration**: How long moves take
- **Height**: Magnitude of moves
- **Frequency**: How often significant moves occur
- **Patterns**: Relationships between consecutive lines

This geometric approach provides richer information about market structure than traditional momentum or mean-reversion indicators.

## Conclusion

The Tradeline Multiclass SFM demonstrates that sophisticated pattern recognition combined with proper threshold calibration can achieve exceptional returns. The key insight that model accuracy ≠ trading performance guides us to optimize for the right metrics: total returns through balanced signal generation.

## Model Files

- Main SFM: `/loop/sfm/lightgbm/tradeline_multiclass.py`
- Utilities: `/loop/sfm/lightgbm/utils/tradeline_multiclass.py`
- Test: Integrated in `/loop/tests/test_sfm.py`
- Documentation: `/docs/SFM/LightGBM-Tradeline-Multiclass.md`
