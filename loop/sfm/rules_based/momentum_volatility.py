import polars as pl
import numpy as np
from typing import Dict
from loop.metrics.multiclass_metrics import multiclass_metrics
from loop.utils.splits import split_sequential, split_data_to_prep_output

# Constants
EPSILON = 1e-10  # Prevent division by zero
TRADING_DAYS_PER_YEAR = 252
HOURS_PER_DAY = 12  # Trading hours per day
RETURN_THRESHOLD = 0.001  # 0.1% threshold for classifying returns

def params():
    return {
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
        'trading_cost': [0.001]  # 0.1% per trade
    }

def prep(data, round_params):
    
    all_datetimes = data['datetime'].to_list()

    split_data = split_sequential(data, (8, 1, 2))
    data_dict = split_data_to_prep_output(split_data, data.columns, all_datetimes)

    return data_dict

def model(data: pl.DataFrame, round_params: Dict) -> Dict:
    '''
    Bidirectional momentum-volatility strategy using dynamic percentile thresholds.
    Supports both long and short positions.
    '''
    
    # Convert to numpy for faster processing
    closes = data['x_test']['close'].to_numpy()
    
    # Build returns history
    returns_history = []
    for i in range(len(closes)):
        if i > 1:
            # Use close[i-1] and close[i-2] for returns at position i
            ret = (closes[i-1] - closes[i-2]) / (closes[i-2] + EPSILON)
            returns_history.append(ret)
    
    # Track positions and orders
    positions = []  # -1 = short, 0 = out, 1 = long
    position = 0  # Current position state
    pending_order = None
    
    # Track trades for performance metrics
    long_trades = 0
    short_trades = 0
    long_wins = 0
    short_wins = 0
    entry_price = 0
    
    for i in range(len(closes)):
        # Execute pending orders at bar open
        if pending_order is not None and i > 0:
            if pending_order['type'] == 'buy' and position == 0:
                position = 1
                entry_price = closes[i]
                long_trades += 1
            elif pending_order['type'] == 'sell' and position == 1:
                if closes[i] > entry_price:
                    long_wins += 1
                position = 0
            elif pending_order['type'] == 'short' and position == 0:
                position = -1
                entry_price = closes[i]
                short_trades += 1
            elif pending_order['type'] == 'cover' and position == -1:
                if closes[i] < entry_price:
                    short_wins += 1
                position = 0
            pending_order = None
        
        # Need enough history for calculations
        if i < 2 or len(returns_history) < round_params['window_size']:
            positions.append(position)
            continue
        
        # Get the current index in returns_history
        current_returns_idx = min(i - 2, len(returns_history) - 1)
        
        if current_returns_idx < round_params['window_size'] - 1:
            positions.append(position)
            continue
        
        # Calculate momentum and volatility using available returns
        recent_returns = returns_history[max(0, current_returns_idx - round_params['window_size'] + 1):current_returns_idx + 1]
        
        if len(recent_returns) < round_params['window_size']:
            positions.append(position)
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
                mom_short_thresh = np.percentile(momentums, round_params['momentum_short_pct'])
                mom_cover_thresh = np.percentile(momentums, round_params['momentum_cover_pct'])
                vol_entry_thresh = np.percentile(volatilities, round_params['volatility_entry_pct'])
                vol_exit_thresh = np.percentile(volatilities, round_params['volatility_exit_pct'])
            else:
                positions.append(position)
                continue
        else:
            positions.append(position)
            continue
        
        # Generate signals for NEXT bar execution
        if position == 1 and pending_order is None:
            # Exit long position
            if current_momentum < mom_sell_thresh or current_volatility > vol_exit_thresh:
                if i < len(closes) - 1:
                    pending_order = {'type': 'sell', 'bar': i}
        
        elif position == -1 and pending_order is None:
            # Exit short position
            if current_momentum > mom_cover_thresh or current_volatility > vol_exit_thresh:
                if i < len(closes) - 1:
                    pending_order = {'type': 'cover', 'bar': i}
        
        elif position == 0 and pending_order is None:
            # Enter new position
            if current_momentum > mom_buy_thresh and current_volatility < vol_entry_thresh:
                # Go long
                if i < len(closes) - 1:
                    pending_order = {'type': 'buy', 'bar': i}
            elif current_momentum < mom_short_thresh and current_volatility < vol_entry_thresh:
                # Go short
                if i < len(closes) - 1:
                    pending_order = {'type': 'short', 'bar': i}
        
        positions.append(position)
    
    # Calculate performance metrics
    positions_array = np.array(positions)
    
    # Count total trades
    num_trades = 0
    for i in range(1, len(positions)):
        if positions[i] != positions[i-1] and positions[i-1] != 0:
            num_trades += 1
    
    # Calculate actual returns for performance
    actual_returns = []
    for i in range(len(closes) - 1):
        actual_returns.append((closes[i+1] - closes[i]) / (closes[i] + EPSILON))
    actual_returns.append(0)
    
    actual_returns_array = np.array(actual_returns)
    
    # Strategy returns (multiply by position: 1 for long, -1 for short, 0 for out)
    strategy_returns = positions_array[:-1] * actual_returns_array[:-1]
    
    # Apply trading costs
    cost_adjustment = num_trades * round_params.get('trading_cost', 0.001)
    
    total_return = np.sum(strategy_returns) - cost_adjustment
    
    # Calculate metrics
    long_positions = np.sum(positions_array == 1)
    short_positions = np.sum(positions_array == -1)
    total_positions = long_positions + short_positions
    
    if total_positions > 0:
        winning_periods = np.sum(strategy_returns > 0)
        periods_in_position = np.sum(positions_array[:-1] != 0)
        win_rate = winning_periods / periods_in_position if periods_in_position > 0 else 0
        
        annualization_factor = np.sqrt(TRADING_DAYS_PER_YEAR * HOURS_PER_DAY)
        if len(strategy_returns) > 1:
            returns_std = np.std(strategy_returns)
            if returns_std > EPSILON:
                sharpe = np.mean(strategy_returns) / returns_std * annualization_factor
            else:
                sharpe = 0
        else:
            sharpe = 0
    else:
        win_rate = 0
        sharpe = 0
    
    # Calculate long/short specific win rates
    long_win_rate = (long_wins / long_trades * 100) if long_trades > 0 else 0
    short_win_rate = (short_wins / short_trades * 100) if short_trades > 0 else 0
    
    # Create ground truth labels based on actual returns
    
    y_true = np.zeros(len(actual_returns_array) - 1)  # Exclude last bar
    y_true[actual_returns_array[:-1] > RETURN_THRESHOLD] = 1  # Should be long
    y_true[actual_returns_array[:-1] < -RETURN_THRESHOLD] = 2  # Should be short (class 2)
    
    # Map predictions to sklearn format (short=-1 becomes class 2)
    y_pred = positions_array[:-1].copy()
    y_pred[y_pred == -1] = 2
    
    # Create data dict for multiclass_metrics
    data_dict = {'y_test': y_true}
    
    # Create pseudo-probabilities based on position confidence
    # For rules-based, we use 0.8 confidence for chosen class, 0.1 for others
    y_proba = np.zeros((len(y_pred), 3))
    for i, pred in enumerate(y_pred):
        y_proba[i, int(pred)] = 0.8
        for j in range(3):
            if j != int(pred):
                y_proba[i, j] = 0.1
    
    # Calculate metrics using Loop's multiclass_metrics
    metrics = multiclass_metrics(data_dict, y_pred, y_proba)
    
    return {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'auc': metrics['auc'],
        '_preds': positions,
        'extras': {
            'total_return': round(total_return * 100, 2),
            'num_long_positions': int(long_positions),
            'num_short_positions': int(short_positions),
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_win_rate': round(long_win_rate, 1),
            'short_win_rate': round(short_win_rate, 1),
            'position_rate': round(total_positions / len(positions), 3) if len(positions) > 0 else 0,
            'sharpe_ratio': round(sharpe, 2),
            'num_trades': num_trades,
            'win_rate': round(win_rate * 100, 2),
            'momentum_buy_pct': round_params['momentum_buy_pct'],
            'momentum_sell_pct': round_params['momentum_sell_pct'],
            'momentum_short_pct': round_params['momentum_short_pct'],
            'momentum_cover_pct': round_params['momentum_cover_pct'],
            'volatility_entry_pct': round_params['volatility_entry_pct'],
            'volatility_exit_pct': round_params['volatility_exit_pct'],
            'lookback_window': round_params['lookback_window']
        }
    }