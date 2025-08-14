#!/usr/bin/env python3
'''
Utility functions for Tradeline Multiclass SFM
Implements line detection, feature engineering, and labeling for multiclass trading predictions
'''

import numpy as np
import polars as pl
import logging
from typing import List, Dict, Tuple, Union
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

# Import Loop indicators
from loop.indicators.price_change_pct import price_change_pct
from loop.indicators.rolling_volatility import rolling_volatility
from loop.features.price_range_position import price_range_position
from loop.features.distance_from_high import distance_from_high
from loop.features.distance_from_low import distance_from_low

MIN_VOLATILITY_THRESHOLD = 1e-6  # Minimum volatility to avoid division issues


def find_price_lines(df: pl.DataFrame, 
                     max_duration_hours: int = 48, 
                     min_height_pct: float = 0.003) -> Tuple[List[Dict], List[Dict]]:
    '''
    Find linear price movements (lines) in the data.
    
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
    
    logging.debug(f"Filtering lines by Q{int(quantile*100)}:")
    logging.debug(f"  Total lines: {len(lines)}")
    logging.debug(f"  Height percentiles - P25: {np.percentile(heights, 25):.3%}, P50: {np.percentile(heights, 50):.3%}, P75: {np.percentile(heights, 75):.3%}")
    logging.debug(f"  Q{int(quantile*100)} threshold: {threshold:.3%}")
    
    filtered = [line for line in lines if abs(line['height_pct']) >= threshold]
    
    return filtered


def compute_line_features(df: pl.DataFrame, 
                         long_lines: List[Dict], 
                         short_lines: List[Dict]) -> pl.DataFrame:
    '''
    Compute line-based features for each row including active lines, momentum, and reversal signals.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'close', 'datetime' columns
        long_lines (List[Dict]): List of long line dictionaries
        short_lines (List[Dict]): List of short line dictionaries
    
    Returns:
        pl.DataFrame: DataFrame with line-based features added
    '''
    n_rows = len(df)
    active_lines = np.zeros(n_rows)
    hours_since_big_move = np.full(n_rows, 168.0)  # Default to 1 week
    line_momentum_6h = np.zeros(n_rows)
    trending_score = np.zeros(n_rows)
    reversal_potential = np.zeros(n_rows)
    
    all_lines = [(line, 1) for line in long_lines] + [(line, -1) for line in short_lines]
    
    for idx in range(n_rows):
        active_count = 0
        recent_long_count = 0
        recent_short_count = 0
        
        for line, direction in all_lines:
            if line['start_idx'] <= idx <= line['end_idx']:
                active_count += 1
            
            if line['end_idx'] < idx and (idx - line['end_idx']) < 168:
                hours_since = idx - line['end_idx']
                hours_since_big_move[idx] = min(hours_since_big_move[idx], hours_since)
            
            if line['end_idx'] >= idx - 6 and line['end_idx'] < idx:
                if direction > 0:
                    recent_long_count += 1
                else:
                    recent_short_count += 1
        
        active_lines[idx] = active_count
        line_momentum_6h[idx] = recent_long_count - recent_short_count
        
        total_recent = recent_long_count + recent_short_count
        if total_recent > 0:
            trending_score[idx] = (recent_long_count - recent_short_count) / total_recent
            reversal_potential[idx] = min(recent_long_count, recent_short_count) / max(recent_long_count, recent_short_count)
        else:
            trending_score[idx] = 0.0  # Neutral trend (0 = balanced between -1 and +1)
            reversal_potential[idx] = 0.0  # No reversal setup (0 = no opposing lines to create reversal)
    
    df = df.with_columns([
        pl.Series('active_lines', active_lines),
        pl.Series('hours_since_big_move', hours_since_big_move),
        pl.Series('line_momentum_6h', line_momentum_6h),
        pl.Series('trending_score', trending_score),
        pl.Series('reversal_potential', reversal_potential)
    ])
    
    return df


def compute_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Compute temporal features from datetime column including hour and day of week.
    
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


class LGBWrapper(BaseEstimator, ClassifierMixin):
    '''
    Wrapper for LightGBM to work with scikit-learn calibration and provide standardized interface.
    
    Args:
        lgb_model: Trained LightGBM model
    '''
    def __init__(self, lgb_model):
        self.lgb_model = lgb_model
        self.scaler = StandardScaler()
        self.classes_ = np.array([0, 1, 2])
        self._is_fitted = False
        self._estimator_type = "classifier"  # Explicitly mark as classifier
        
    def fit(self, X, y):
        self.scaler.fit(X)
        self._is_fitted = True
        return self
        
    def predict(self, X):
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return np.argmax(self.lgb_model.predict(X_scaled, num_iteration=self.lgb_model.best_iteration), axis=1)
        
    def predict_proba(self, X):
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.lgb_model.predict(X_scaled, num_iteration=self.lgb_model.best_iteration)




def apply_class_weights(y_train: np.ndarray) -> np.ndarray:
    '''
    Compute balanced sample weights to handle class imbalance in multiclass classification.
    
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