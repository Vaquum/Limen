# Momentum-Volatility Strategy SFM

## Overview
Rules-based trading strategies using momentum and volatility indicators with dynamic percentile thresholds. Available in two variants:
- **Long-only** (`momentum_volatility_longonly.py`): Traditional long/cash positions
- **Bidirectional** (`momentum_volatility.py`): Long/short/cash positions

## Strategy Logic

### Core Concept
Both strategies use dynamic percentile thresholds calculated from rolling historical windows with pure technical analysis (no ML).

#### Long-Only Strategy
- **Positions**: Long (1) or Cash (0)
- **Entry**: Go long when momentum > buy threshold AND volatility < entry threshold
- **Exit**: Exit when momentum < sell threshold OR volatility > exit threshold

#### Bidirectional Strategy  
- **Positions**: Long (1), Short (-1), or Cash (0)
- **Long Entry**: Momentum > buy threshold AND volatility < entry threshold
- **Long Exit**: Momentum < sell threshold OR volatility > exit threshold
- **Short Entry**: Momentum < short threshold AND volatility < entry threshold
- **Short Exit**: Momentum > cover threshold OR volatility > exit threshold

### Key Features
- 48-hour window optimal for both momentum and volatility calculations
- Dynamic threshold adaptation to market conditions  
- No lookahead bias - signals execute at next bar's open
- Trading costs included (0.075% per trade)

## Implementation

### Parameters

#### Long-Only Version
```python
{
    'window_size': [24, 48, 72],
    'momentum_buy_pct': [55, 60, 65, 70, 75, 80],
    'momentum_sell_pct': [30, 35, 40, 45, 50, 55],
    'volatility_buy_pct': [70, 75, 80, 85, 90],
    'volatility_sell_pct': [80, 85, 90, 95],
    'lookback_window': [300, 500, 750],
    'trading_cost': [0.00075]
}
```

#### Bidirectional Version
```python
{
    'window_size': [24, 48, 72],
    # Long thresholds
    'momentum_buy_pct': [55, 60, 65, 70, 75, 80],
    'momentum_sell_pct': [30, 35, 40, 45, 50, 55],
    # Short thresholds  
    'momentum_short_pct': [20, 25, 30, 35, 40, 45],
    'momentum_cover_pct': [50, 55, 60, 65, 70],
    # Volatility thresholds
    'volatility_entry_pct': [70, 75, 80, 85, 90],
    'volatility_exit_pct': [80, 85, 90, 95],
    'lookback_window': [300, 500, 750],
    'trading_cost': [0.00075]
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
- `_preds`: Position array (long-only: 0/1, bidirectional: -1/0/1)
- `extras`: Detailed performance metrics including ROI, Sharpe, trades, separate long/short statistics

## Best Configurations

### Long-Only Strategy (76.6% ROI)
- Window Size: 48 hours
- Momentum Buy: >55th percentile
- Momentum Sell: <30th percentile
- Volatility Buy: <70th percentile
- Volatility Sell: >95th percentile
- Lookback: 300 periods
- **Performance**: 76.6% ROI, 194 trades, 39.2% win rate

### Bidirectional Strategy (229.3% ROI)
- Window Size: 48 hours  
- Long Entry/Exit: >55th / <30th percentile
- Short Entry/Exit: <20th / >50th percentile
- Volatility Entry/Exit: <85th / >95th percentile
- Lookback: TBD (sweep in progress)
- **Performance**: 229.3% ROI, 103 long trades, 86 short trades

## Performance Characteristics

### Long-Only Strategy
- **Expected Returns**: 20-76% ROI
- **Win Rate**: 35-45% typical
- **Number of Trades**: 150-250 per 6 months
- **Sharpe Ratio**: 1.0-1.5 range

### Bidirectional Strategy
- **Expected Returns**: 100-230% ROI
- **Win Rate**: 40-50% typical
- **Long/Short Mix**: ~55% long, ~45% short trades
- **Number of Trades**: 180-300 per 6 months
- **Sharpe Ratio**: 1.5-2.0+ range

Both strategies include 0.075% trading costs per trade in all calculations.