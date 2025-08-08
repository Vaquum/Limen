# Momentum-Volatility Strategy SFM

## Overview
A Single File Model implementation of the momentum-volatility trading strategy. This SFM creates a binary classifier that predicts whether to be in position (long) or out of position based on momentum and volatility indicators.

## Strategy Logic

### Core Concept
- **Entry**: When momentum is strong and volatility is controlled
- **Exit**: When momentum weakens or volatility spikes
- Uses dynamic percentile thresholds calculated from rolling historical windows
- Pure technical analysis approach without external dependencies (no VIX)

### Key Features
- 48-hour momentum period optimal (tested 24, 48, 72 hours)
- Dynamic threshold adaptation to market conditions
- Binary classification: 1 = be in position, 0 = stay out
- Validated 30-40% returns over 6-month periods

## Implementation

### Location
`/Users/beyondsyntax/Loop/loop/sfm/rules_based/momentum_volatility.py`

### Parameters
```python
{
    'momentum_period': [24, 48, 72],           # Hours for momentum calculation
    'volatility_period': [24, 48, 72],         # Hours for volatility calculation
    'momentum_buy_pct': [55, 60, 65, 70, 75, 80],   # Entry momentum percentile
    'momentum_sell_pct': [30, 35, 40, 45],          # Exit momentum percentile
    'volatility_buy_pct': [75, 80, 85, 90],         # Entry volatility percentile
    'volatility_sell_pct': [85, 90, 95],            # Exit volatility percentile
    'lookback_window': [300, 400, 500],             # Historical window for percentiles
}
```

### Data Preparation (`prep`)
1. **Calculate Returns**: Using Loop's `returns` indicator
2. **Compute Momentum**: Simple moving average of returns using `sma` indicator
3. **Compute Volatility**: Rolling standard deviation using `rolling_volatility` indicator
4. **Dynamic Thresholds**: Rolling percentiles for entry/exit signals
5. **Generate Signals**: Buy and sell signals based on thresholds
6. **Create Target**: Binary target `should_be_long` (1=in position, 0=out)
7. **Normalize Features**: Z-score normalization using training statistics
8. **Split Data**: 8:1:2 train/val/test split

### Model Training (`model`)
- Uses Logistic Regression classifier
- L2 regularization (C=1.0)
- LBFGS solver for optimization
- Returns standard binary metrics plus strategy-specific metrics

### Output Metrics
- Standard: `recall`, `precision`, `fpr`, `auc`, `accuracy`
- Strategy-specific: `strategy_return`, `num_trades`
- Private: `_preds` (predictions), `_clf` (trained classifier)

## Best Configurations

### Config #1: Balanced (48-hour, moderate thresholds)
- Momentum period: 48 hours
- Buy when momentum > 70th percentile
- Sell when momentum < 35th percentile
- Volatility thresholds: 80/90 percentiles
- **Performance**: ~42% mean ROI

### Config #2: Conservative (higher entry barrier)
- Momentum period: 48 hours
- Buy when momentum > 80th percentile
- Sell when momentum < 40th percentile
- Fewer trades, higher win rate
- **Performance**: ~32% mean ROI

## Usage with Universal Experiment Loop

### Run Single Configuration
```python
from loop.uel import UniversalExperimentLoop
from loop.sfm.rules_based import momentum_volatility

uel = UniversalExperimentLoop(
    model=momentum_volatility,
    data_source='your_data_source',
    n_rounds=100
)
results = uel.run()
```

### Parameter Optimization
```python
# UEL will automatically sweep through the parameter grid
uel = UniversalExperimentLoop(
    model=momentum_volatility,
    data_source='your_data_source',
    n_rounds=500,  # Test 500 random parameter combinations
    optimization_mode=True
)
best_params = uel.run()
```

## Integration Notes

### Data Requirements
- Hourly OHLC data with at least 500 hours of history
- Standard klines format with 'open', 'high', 'low', 'close', 'volume' columns
- No external data dependencies (VIX not needed)

### Compatibility
- Fully compatible with Loop's Universal Experiment Loop
- Uses standard Loop indicators and features
- Returns standard binary classification metrics
- Can be used with Loop's backtesting framework

### Key Differences from Standalone Version
- Integrated with Loop's indicator library
- Proper train/val/test splitting using Loop utilities
- Normalized features for better classifier performance
- Returns Loop-standard metrics dictionary

## Performance Characteristics
- **Expected Returns**: 30-40% over 6-month periods
- **Win Rate**: 40-48% typical
- **Number of Trades**: 150-300 per 6 months
- **Max Drawdown**: 15-20% typical
- **Sharpe Ratio**: 1.5-1.7 range
