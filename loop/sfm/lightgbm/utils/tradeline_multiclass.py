#!/usr/bin/env python3
'''
Optimized Utility functions for Tradeline Multiclass SFM
Implements line detection, feature engineering, and labeling for multiclass trading predictions
Uses sliding window event-driven algorithm for O(m log m + n) complexity
'''

import numpy as np
import polars as pl
from typing import List, Dict, Tuple, Any, Optional
from sklearn.utils.class_weight import compute_class_weight

# Import Loop indicators
from loop.indicators.price_change_pct import price_change_pct
from loop.indicators.rolling_volatility import rolling_volatility
from loop.indicators.atr import atr
from loop.features.active_lines import active_lines
from loop.features.hours_since_big_move import hours_since_big_move
from loop.features.hours_since_quantile_line import hours_since_quantile_line
from loop.features.active_quantile_count import active_quantile_count
from loop.features.quantile_line_density import quantile_line_density
from loop.features.price_range_position import price_range_position
from loop.features.distance_from_high import distance_from_high
from loop.features.distance_from_low import distance_from_low

MIN_VOLATILITY_THRESHOLD = 1e-6  # Minimum volatility to avoid division issues


def find_price_lines(df: pl.DataFrame, 
                     max_duration_hours: int = 48, 
                     min_height_pct: float = 0.003) -> Tuple[List[Dict], List[Dict]]:
    '''
    Find linear price movements (lines) in the data.
    UNCHANGED - this function is already reasonably efficient
    
    Args:
        df: DataFrame with 'close' price column
        max_duration_hours: Maximum duration of a line in hours
        min_height_pct: Minimum height as percentage of start price
        
    Returns:
        Tuple of (long_lines, short_lines) where each is a list of dicts
    '''
    long_lines = []
    short_lines = []
    
    prices = df.get_column('close').to_numpy()
    n_prices = len(prices)
    
    for start_idx in range(n_prices):
            
        start_price = prices[start_idx]
        max_end_idx = min(start_idx + max_duration_hours, n_prices)
        
        for end_idx in range(start_idx + 1, max_end_idx):
            end_price = prices[end_idx]
            height_pct = (end_price - start_price) / start_price
            
            if abs(height_pct) >= min_height_pct:
                duration_hours = end_idx - start_idx
                line_data = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_price': start_price,
                    'end_price': end_price,
                    'height_pct': height_pct,
                    'duration_hours': duration_hours,
                    'start_time': df.item(start_idx, 'datetime') if 'datetime' in df.columns else start_idx,
                    'end_time': df.item(end_idx, 'datetime') if 'datetime' in df.columns else end_idx
                }
                
                if height_pct > 0:
                    long_lines.append(line_data)
                else:
                    short_lines.append(line_data)
    
    return long_lines, short_lines


def filter_lines_by_quantile(lines: List[Dict], quantile: float) -> List[Dict]:
    '''
    Filter lines to keep only those above the specified quantile by height.
    UNCHANGED - already efficient
    
    Args:
        lines: List of line dictionaries
        quantile: Quantile threshold (e.g., 0.75 for Q75)
        
    Returns:
        Filtered list of lines
    '''
    if not lines:
        return []
    
    heights = [abs(line['height_pct']) for line in lines]
    threshold = np.quantile(heights, quantile)
    
    filtered = [line for line in lines if abs(line['height_pct']) >= threshold]
    
    return filtered


def compute_line_features(
    df: pl.DataFrame,
    long_lines: List[Dict],
    short_lines: List[Dict],
    *,
    density_lookback_hours: int = 48,
    big_move_lookback_hours: int = 168,
) -> pl.DataFrame:
    '''
    OPTIMIZED: Compute line-based features using sliding window event-driven algorithm.
    Complexity: O(m log m + n) instead of O(n * m)
    
    Args:
        df (pl.DataFrame): Klines dataset with 'close', 'datetime' columns
        long_lines (List[Dict]): List of long line dictionaries
        short_lines (List[Dict]): List of short line dictionaries
    
    Returns:
        pl.DataFrame: DataFrame with line-based features added
    '''
    df = active_lines(df, long_lines, short_lines)
    df = hours_since_big_move(df, long_lines, short_lines, big_move_lookback_hours)
    return df


def compute_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Compute temporal features from datetime column including hour and day of week.
    UNCHANGED - already efficient
    
    Args:
        df (pl.DataFrame): Klines dataset with 'datetime' column
    
    Returns:
        pl.DataFrame: DataFrame with temporal features added
    '''
    df = df.with_columns([
        pl.col('datetime').dt.hour().alias('hour_of_day'),
        pl.col('datetime').dt.weekday().alias('day_of_week')
    ])
    
    return df


def compute_price_features(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Compute comprehensive price-based features including returns, acceleration, and volatility.
    UNCHANGED - already efficient
    
    Args:
        df (pl.DataFrame): Klines dataset with 'open', 'high', 'low', 'close', 'volume' columns
    
    Returns:
        pl.DataFrame: DataFrame with comprehensive price-based features
    '''
    for period in [1, 6, 12, 24, 48]:
        df = price_change_pct(df, period)
        df = df.rename({f'price_change_pct_{period}': f'ret_{period}h'})
    
    df = df.with_columns([
        (pl.col('ret_6h') - pl.col('ret_6h').shift(6)).alias('accel_6h'),
        (pl.col('ret_24h') - pl.col('ret_24h').shift(24)).alias('accel_24h')
    ])
    
    df = distance_from_high(df, period=24)
    df = distance_from_low(df, period=24)
    df = price_range_position(df, period=24)
    
    df = df.rename({
        'distance_from_high': 'dist_from_high',
        'distance_from_low': 'dist_from_low',
        'price_range_position': 'position_in_range'
    })
    
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns_temp')
    ])
    
    for period in [6, 24]:
        df = rolling_volatility(df, 'returns_temp', period)
        df = df.rename({f'returns_temp_volatility_{period}': f'vol_{period}h'})
    
    df = df.with_columns(
        pl.when(pl.col('vol_6h') < MIN_VOLATILITY_THRESHOLD)
        .then(1.0)  # When 6h volatility is near zero, assume no expansion
        .otherwise(pl.col('vol_24h') / pl.col('vol_6h'))
        .alias('vol_expansion')
    )
    
    df = df.drop('returns_temp')
    
    df = df.with_columns([
        ((pl.col('ret_6h') + pl.col('ret_12h') + pl.col('ret_24h')) / 3).alias('ret_mean'),
        pl.concat_list([pl.col('ret_6h'), pl.col('ret_12h'), pl.col('ret_24h')]).list.std().alias('ret_std'),
        ((pl.col('accel_6h') + pl.col('accel_24h')) / 2).alias('accel_mean')
    ])
    
    return df


def create_multiclass_labels(df: pl.DataFrame,
                           long_threshold: float,
                           short_threshold: float,
                           lookahead_hours: int = 48) -> pl.DataFrame:
    '''
    Compute 3-class labels based on future price movements.
    UNCHANGED - already efficient
    
    Args:
        df (pl.DataFrame): Klines dataset with 'close' column
        long_threshold (float): Threshold for long trades (positive)
        short_threshold (float): Threshold for short trades (positive, will be negated)
        lookahead_hours (int): Hours to look ahead for price movements
        
    Returns:
        pl.DataFrame: The input data with new column 'label'
    '''
    close_prices = df.get_column('close')
    n_rows = len(df)
    labels = np.zeros(n_rows, dtype=int)
    
    long_count = 0
    short_count = 0
    
    for idx in range(n_rows - lookahead_hours):
        current_price = close_prices[idx]
        
        future_24h_idx = min(idx + 24, n_rows - 1)
        future_48h_idx = min(idx + lookahead_hours, n_rows - 1)
        
        future_prices = close_prices[idx:future_48h_idx+1]
        max_future_price = future_prices.max()
        min_future_price = future_prices.min()
        
        max_future_ret = (max_future_price - current_price) / current_price
        min_future_ret = (min_future_price - current_price) / current_price
        
        ret_24h = (close_prices[future_24h_idx] - current_price) / current_price
        ret_48h = (close_prices[future_48h_idx] - current_price) / current_price
        
        if max_future_ret >= long_threshold:
            if ret_48h > long_threshold or ret_24h > long_threshold:
                labels[idx] = 1  # LONG
                long_count += 1
        elif abs(min_future_ret) >= short_threshold:
            if ret_48h < -short_threshold or ret_24h < -short_threshold:
                labels[idx] = 2  # SHORT
                short_count += 1
    
    df = df.with_columns(pl.Series('label', labels))
    
    return df


    


def apply_class_weights(y_train: np.ndarray) -> np.ndarray:
    '''
    Compute balanced sample weights to handle class imbalance in multiclass classification.
    UNCHANGED - already efficient
    
    Args:
        y_train (np.ndarray): Array of training labels (0, 1, or 2)
        
    Returns:
        np.ndarray: Array of sample weights, same length as y_train
    '''
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    sample_weights = np.zeros(len(y_train))
    for i, cls in enumerate(classes):
        mask = y_train == cls
        sample_weights[mask] = class_weights[i]
    
    return sample_weights


def compute_quantile_line_features(
    df: pl.DataFrame,
    long_lines_filtered: List[Dict[str, Any]],
    short_lines_filtered: List[Dict[str, Any]],
    quantile_threshold: float = 0.75,
    *,
    density_lookback_hours: int = 48,
) -> pl.DataFrame:
    
    '''
    Compute quantile-filtered line pattern features for trading predictions.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'datetime', 'open', 'high', 'low', 'close', 'volume' columns
        long_lines_filtered (list): Filtered long line patterns from quantile filtering
        short_lines_filtered (list): Filtered short line patterns from quantile filtering
        quantile_threshold (float): Quantile threshold used for filtering (e.g., 0.75 for Q75)
        
    Returns:
        pl.DataFrame: The input data with six new columns: 'hours_since_quantile_line', 'quantile_momentum_6h', 'active_quantile_count', 'quantile_line_density_48h', 'avg_quantile_height_24h', 'quantile_direction_bias'
    '''
    df = hours_since_quantile_line(df, long_lines_filtered, short_lines_filtered, density_lookback_hours)
    df = active_quantile_count(df, long_lines_filtered, short_lines_filtered)
    df = quantile_line_density(df, long_lines_filtered, short_lines_filtered, density_lookback_hours)
    return df


def calculate_atr(df: pl.DataFrame, period: int = 24) -> pl.DataFrame:
    
    '''
    Compute Average True Range (ATR) indicator over specified period.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        period (int): Number of periods for ATR calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'atr_pct'
    '''
    df = atr(df, period=period)
    col = f"atr_{period}"
    if col in df.columns:
        df = df.with_columns([(pl.col(col) / pl.col('close')).alias('atr_pct')])
        df = df.drop([col])
    return df


def apply_complete_exit_strategy(df: pl.DataFrame, predictions: np.ndarray, probabilities: np.ndarray, long_threshold: float = 0.034, short_threshold: float = 0.034, config: Optional[Dict[str, Any]] = None) -> Tuple[pl.DataFrame, Dict[str, float]]:
    
    '''
    Compute complete exit strategy with ATR-based risk management and position sizing.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'close', 'atr_pct' columns
        predictions (np.ndarray): Model predictions array with class labels
        probabilities (np.ndarray): Model probability predictions for confidence filtering
        long_threshold (float): Profit threshold for long positions
        short_threshold (float): Profit threshold for short positions
        config (Dict[str, Any]): Configuration dictionary with trading parameters
        
    Returns:
        tuple: Original dataframe and trading results dictionary with performance metrics
    '''
    if config is None:
        config = {}
    
    confidence_threshold = config.get('confidence_threshold', 0.60)
    position_size = config.get('position_size', 0.199)
    max_positions = config.get('max_positions', 3)
    min_stop_loss = config.get('min_stop_loss', 0.01)
    max_stop_loss = config.get('max_stop_loss', 0.04)
    atr_stop_multiplier = config.get('atr_stop_multiplier', 1.5)
    trailing_activation = config.get('trailing_activation', 0.02)
    trailing_distance = config.get('trailing_distance', 0.5)
    loser_timeout_hours = config.get('loser_timeout_hours', 24)
    max_hold_hours = config.get('max_hold_hours', 48)
    initial_capital = config.get('initial_capital', 100000.0)
    default_atr_pct = config.get('default_atr_pct', 0.015)
    n_rows = len(df)
    if n_rows == 0 or len(predictions) == 0:
        return df, {
            'total_return_net_pct': 0.0,
            'trade_win_rate_pct': 0.0,
            'trades_count': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
    prices = df.select(pl.col('close')).to_numpy().flatten()
    atr_values = df.select(pl.col('atr_pct')).to_numpy().flatten() if 'atr_pct' in df.columns else np.full(n_rows, default_atr_pct)
    max_probs = np.max(probabilities, axis=1) if len(probabilities.shape) > 1 else probabilities
    high_conf_mask = (max_probs >= confidence_threshold) & (predictions != 0)
    entry_signals = np.where(high_conf_mask)[0]
    if len(entry_signals) == 0:
        return df, {
            'total_return_net_pct': 0.0,
            'trade_win_rate_pct': 0.0,
            'trades_count': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
    capital = initial_capital
    trades = []
    positions: List[Tuple[int, float, int, float, float]] = []  
    check_indices = set(entry_signals)
    for i in range(0, n_rows, 24):
        check_indices.add(i)
    check_indices_list = sorted(list(check_indices))
    for i in check_indices_list:
        if i >= n_rows:
            break
        current_price = prices[i]
        current_atr = atr_values[i]
        positions_to_close = []
        for pos_idx, (entry_idx, entry_price, direction, size, peak_profit) in enumerate(positions):
            hours_held = i - entry_idx
            if direction == 1:  
                pnl_pct = (current_price - entry_price) / entry_price
            else:  
                pnl_pct = (entry_price - current_price) / entry_price
            peak_profit = max(peak_profit, pnl_pct)
            positions[pos_idx] = (entry_idx, entry_price, direction, size, peak_profit)
            should_exit = False
            exit_reason = ""
            take_profit = long_threshold if direction == 1 else short_threshold
            if pnl_pct >= take_profit:
                should_exit = True
                exit_reason = "take_profit"
            elif pnl_pct <= -max(min_stop_loss, min(max_stop_loss, atr_stop_multiplier * current_atr)):
                should_exit = True
                exit_reason = "stop_loss"
            elif peak_profit >= trailing_activation:
                trailing_stop = peak_profit - (trailing_distance * current_atr)
                if pnl_pct <= trailing_stop:
                    should_exit = True
                    exit_reason = "trailing_stop"
            elif (pnl_pct < 0 and hours_held >= loser_timeout_hours) or hours_held >= max_hold_hours:
                should_exit = True
                exit_reason = "time_exit"
            if should_exit:
                positions_to_close.append((pos_idx, pnl_pct, exit_reason))
        for pos_idx, pnl_pct, exit_reason in reversed(positions_to_close):
            entry_idx, entry_price, direction, size, _ = positions.pop(pos_idx)
            net_pnl = pnl_pct * size
            capital += net_pnl
            trades.append({
                'entry_time': entry_idx,
                'exit_time': i,
                'pnl_pct': pnl_pct,
                'net_pnl': net_pnl,
                'exit_reason': exit_reason
            })
        if i in entry_signals and len(positions) < max_positions:
            pred = predictions[i]
            direction = 1 if pred == 1 else -1
            pos_size = capital * position_size
            positions.append((i, current_price, direction, pos_size, 0.0))
    if trades:
        total_return = (capital - initial_capital) / initial_capital * 100
        winning_trades = [t for t in trades if t['net_pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in trades if t['net_pnl'] < 0]
        avg_loss = np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
        trading_results = {
            'total_return_net_pct': total_return,
            'trade_win_rate_pct': win_rate,
            'trades_count': len(trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    else:
        trading_results = {
            'total_return_net_pct': 0.0,
            'trade_win_rate_pct': 0.0,
            'trades_count': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
    return df, trading_results