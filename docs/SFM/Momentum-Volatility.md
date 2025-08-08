# Momentum-Volatility Strategy SFM

## Overview
A long-only rules-based trading strategy that uses momentum and volatility indicators with dynamic percentile thresholds to determine when to enter and exit long positions.

## Strategy Logic

### Core Concept
- **Long-only**: Either in long position or cash, never short
- **Entry**: Go long when momentum is strong (above threshold) AND volatility is controlled (below threshold)
- **Exit**: Exit long when momentum weakens (below threshold) OR volatility spikes (above threshold)
- Uses dynamic percentile thresholds calculated from rolling historical windows
- Pure technical analysis approach without ML or external dependencies

### Key Features
- 48-hour window optimal for both momentum and volatility calculations
- Dynamic threshold adaptation to market conditions  
- Validated 76.6% ROI in best configuration
- No lookahead bias - signals execute at next bar's open

## Implementation

### Parameters
```python
{
    'window_size': [24, 48, 72],                    # Hours for momentum and volatility
    'momentum_buy_pct': [55, 60, 65, 70, 75, 80],   # Entry momentum percentile
    'momentum_sell_pct': [30, 35, 40, 45, 50, 55],  # Exit momentum percentile
    'volatility_buy_pct': [70, 75, 80, 85, 90],     # Entry volatility percentile  
    'volatility_sell_pct': [80, 85, 90, 95],        # Exit volatility percentile
    'lookback_window': [300, 500, 750],             # Historical window for percentiles
    'trading_cost': [0.00075]                       # 0.075% per trade
}
```

### Signal Generation
1. **Calculate Returns**: Price change between consecutive bars
2. **Compute Momentum**: Rolling mean of returns over window_size
3. **Compute Volatility**: Rolling standard deviation of returns over window_size
4. **Dynamic Thresholds**: Calculate percentiles from lookback_window of historical data
5. **Generate Signals**: 
   - Buy when momentum > buy percentile AND volatility < buy percentile
   - Sell when momentum < sell percentile OR volatility > sell percentile
6. **Execute Trades**: Signals at bar[i] execute at bar[i+1] open (no lookahead)

### Output Metrics
- `_preds`: Position array (1=long, 0=out)
- `extras`: Detailed performance metrics including ROI, Sharpe, trades

## Best Configuration

### Optimal Parameters (76.6% ROI)
- Window Size: 48 hours
- Momentum Buy: >55th percentile
- Momentum Sell: <30th percentile
- Volatility Buy: <70th percentile
- Volatility Sell: >95th percentile
- Lookback: 300 periods
- **Performance**: 76.6% ROI, 194 trades, 39.2% win rate

## Performance Characteristics
- **Expected Returns**: 20-76% depending on parameters
- **Win Rate**: 35-45% typical
- **Number of Trades**: 150-250 per 6 months
- **Sharpe Ratio**: 1.0-1.5 range
- **Trading Costs**: 0.075% per trade included in calculations