import polars as pl
import numpy as np
from typing import Dict

def params():
    return {
        'window_size': [24, 48, 72],
        'momentum_buy_pct': [55, 60, 65, 70, 75, 80],
        'momentum_sell_pct': [30, 35, 40, 45, 50, 55],
        'volatility_buy_pct': [70, 75, 80, 85, 90],
        'volatility_sell_pct': [80, 85, 90, 95],
        'lookback_window': [300, 500, 750],
        'trading_cost': [0.00075]  # 0.075% per trade
    }

def prep(data, round_params):
    return data

def model(data: pl.DataFrame, round_params: Dict) -> Dict:
    """
    Momentum-volatility strategy using dynamic percentile thresholds.
    """
    
    # Convert to numpy for faster processing
    closes = data['close'].to_numpy()
    
    # Build returns history
    returns_history = []
    for i in range(len(closes)):
        if i > 1:
            # Use close[i-1] and close[i-2] for returns at position i
            ret = (closes[i-1] - closes[i-2]) / (closes[i-2] + 1e-10)
            returns_history.append(ret)
    
    # Track positions and orders
    positions = []
    position_open = False
    pending_order = None
    
    for i in range(len(closes)):
        # Execute pending orders at bar open
        if pending_order is not None and i > 0:
            if pending_order['type'] == 'buy' and not position_open:
                position_open = True
            elif pending_order['type'] == 'sell' and position_open:
                position_open = False
            pending_order = None
        
        # Need enough history for calculations
        if i < 2 or len(returns_history) < round_params['window_size']:
            positions.append(1 if position_open else 0)
            continue
        
        # Get the current index in returns_history
        # At bar i, we have returns up to index i-2 (since returns_history is built with i>1)
        current_returns_idx = min(i - 2, len(returns_history) - 1)
        
        if current_returns_idx < round_params['window_size'] - 1:
            positions.append(1 if position_open else 0)
            continue
        
        # Calculate momentum and volatility using available returns
        recent_returns = returns_history[max(0, current_returns_idx - round_params['window_size'] + 1):current_returns_idx + 1]
        
        if len(recent_returns) < round_params['window_size']:
            positions.append(1 if position_open else 0)
            continue
            
        current_momentum = np.mean(recent_returns)
        current_volatility = np.std(recent_returns)
        
        # Calculate thresholds from historical data
        if current_returns_idx + 1 >= min(round_params['lookback_window'], 100):
            lookback = min(round_params['lookback_window'], current_returns_idx + 1)
            historical_returns = returns_history[max(0, current_returns_idx - lookback + 1):current_returns_idx + 1]
            
            momentums = []
            volatilities = []
            
            for j in range(round_params['window_size'], len(historical_returns) + 1):
                window = historical_returns[j-round_params['window_size']:j]
                momentums.append(np.mean(window))
                volatilities.append(np.std(window))
            
            if momentums and volatilities:
                mom_buy_thresh = np.percentile(momentums, round_params['momentum_buy_pct'])
                mom_sell_thresh = np.percentile(momentums, round_params['momentum_sell_pct'])
                vol_buy_thresh = np.percentile(volatilities, round_params['volatility_buy_pct'])
                vol_sell_thresh = np.percentile(volatilities, round_params['volatility_sell_pct'])
            else:
                positions.append(1 if position_open else 0)
                continue
        else:
            positions.append(1 if position_open else 0)
            continue
        
        # Generate signals for NEXT bar execution
        if position_open and pending_order is None:
            if current_momentum < mom_sell_thresh or current_volatility > vol_sell_thresh:
                if i < len(closes) - 1:
                    pending_order = {'type': 'sell', 'bar': i}
        
        if not position_open and pending_order is None:
            if current_momentum > mom_buy_thresh and current_volatility < vol_buy_thresh:
                if i < len(closes) - 1:
                    pending_order = {'type': 'buy', 'bar': i}
        
        positions.append(1 if position_open else 0)
    
    # Calculate performance metrics
    positions_array = np.array(positions)
    
    # Count trades first
    num_trades = 0
    for i in range(1, len(positions)):
        if positions[i] != positions[i-1]:
            num_trades += 1
    
    # Calculate actual returns for performance
    actual_returns = []
    for i in range(len(closes) - 1):
        actual_returns.append((closes[i+1] - closes[i]) / (closes[i] + 1e-10))
    actual_returns.append(0)
    
    actual_returns_array = np.array(actual_returns)
    
    # Strategy returns WITHOUT costs first
    strategy_returns = positions_array[:-1] * actual_returns_array[:-1]
    
    # Apply trading costs
    cost_adjustment = num_trades * round_params.get('trading_cost', 0.00075)
    
    total_positions = np.sum(positions_array)
    total_return = np.sum(strategy_returns) - cost_adjustment
    
    if total_positions > 0:
        positions_sum = np.sum(positions_array[:-1])
        avg_return_when_long = total_return / (positions_sum if positions_sum > 0 else 1)
        winning_periods = np.sum(strategy_returns > 0)
        periods_in_position = np.sum(positions_array[:-1] > 0)
        win_rate = winning_periods / periods_in_position if periods_in_position > 0 else 0
        
        annualization_factor = np.sqrt(252 * 12)  # Assuming hourly bars, 252 trading days
        if len(strategy_returns) > 1:
            returns_std = np.std(strategy_returns)
            if returns_std > 1e-10:
                sharpe = np.mean(strategy_returns) / returns_std * annualization_factor
            else:
                sharpe = 0
        else:
            sharpe = 0
    else:
        avg_return_when_long = 0
        win_rate = 0
        sharpe = 0
    
    # Create ground truth labels based on actual returns
    # Binary classification: should be long (1) or out (0)
    return_threshold = 0.001  # 0.1% return threshold
    
    y_true = np.zeros(len(actual_returns_array) - 1)  # Exclude last bar
    y_true[actual_returns_array[:-1] > return_threshold] = 1  # Should be long
    
    y_pred = positions_array[:-1]
    
    # Calculate binary classification metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # AUC would need probability scores, using 0.5 as baseline
    auc = 0.5
    
    return {
        'accuracy': round(accuracy, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'auc': auc,
        '_preds': positions,
        'extras': {
            'total_return': round(total_return * 100, 2),
            'avg_return_when_long': round(avg_return_when_long * 100, 3),
            'num_long_positions': int(total_positions),
            'position_rate': round(total_positions / len(positions), 3) if len(positions) > 0 else 0,
            'sharpe_ratio': round(sharpe, 2),
            'num_trades': num_trades,
            'win_rate': round(win_rate * 100, 2),
            'momentum_buy_pct': round_params['momentum_buy_pct'],
            'momentum_sell_pct': round_params['momentum_sell_pct'],
            'volatility_buy_pct': round_params['volatility_buy_pct'],
            'volatility_sell_pct': round_params['volatility_sell_pct'],
            'lookback_window': round_params['lookback_window']
        }
    }