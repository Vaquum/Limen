#!/usr/bin/env python3
"""
Utility functions for Tradeline Multiclass SFM
Implements line detection, feature engineering, and labeling for multiclass trading predictions
"""

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
from loop.indicators.price_range_position import price_range_position
from loop.indicators.distance_from_high import distance_from_high
from loop.indicators.distance_from_low import distance_from_low


def find_price_lines(df: pl.DataFrame, 
                     max_duration_hours: int = 48, 
                     min_height_pct: float = 0.003) -> Tuple[List[Dict], List[Dict]]:
    """
    Find linear price movements (lines) in the data.
    
    Args:
        df: DataFrame with 'close' price column
        max_duration_hours: Maximum duration of a line in hours
        min_height_pct: Minimum height as percentage of start price
        
    Returns:
        Tuple of (long_lines, short_lines) where each is a list of dicts
    """
    long_lines = []
    short_lines = []
    
    prices = df.get_column('close').to_numpy()
    n_prices = len(prices)
    
    logging.debug(f"Finding price lines in {n_prices} data points...")
    logging.debug(f"This will check up to {n_prices * max_duration_hours // 2} potential lines")
    
    # Progress tracking
    last_progress = 0
    
    for start_idx in range(n_prices):
        # Progress update every 5%
        progress = int(start_idx / n_prices * 100)
        if progress >= last_progress + 5:
            logging.debug(f"  Line computation progress: {progress}% ({start_idx}/{n_prices} start points processed)")
            logging.debug(f"  Found {len(long_lines)} long and {len(short_lines)} short lines so far")
            last_progress = progress
            
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
                    'start_time': df[start_idx, 'datetime'] if 'datetime' in df.columns else start_idx,
                    'end_time': df[end_idx, 'datetime'] if 'datetime' in df.columns else end_idx
                }
                
                if height_pct > 0:
                    long_lines.append(line_data)
                else:
                    short_lines.append(line_data)
    
    logging.debug(f"Line computation complete!")
    return long_lines, short_lines


def filter_lines_by_quantile(lines: List[Dict], quantile: float) -> List[Dict]:
    """
    Filter lines to keep only those above the specified quantile by height.
    
    Args:
        lines: List of line dictionaries
        quantile: Quantile threshold (e.g., 0.75 for Q75)
        
    Returns:
        Filtered list of lines
    """
    if not lines:
        return []
    
    heights = [abs(line['height_pct']) for line in lines]
    threshold = np.quantile(heights, quantile)
    
    # Debug logging
    logging.debug(f"Filtering lines by Q{int(quantile*100)}:")
    logging.debug(f"  Total lines: {len(lines)}")
    logging.debug(f"  Height percentiles - P25: {np.percentile(heights, 25):.3%}, P50: {np.percentile(heights, 50):.3%}, P75: {np.percentile(heights, 75):.3%}")
    logging.debug(f"  Q{int(quantile*100)} threshold: {threshold:.3%}")
    
    filtered = [line for line in lines if abs(line['height_pct']) >= threshold]
    logging.debug(f"  Lines after filtering: {len(filtered)}")
    
    return filtered


def compute_line_features(df: pl.DataFrame, 
                         long_lines: List[Dict], 
                         short_lines: List[Dict]) -> pl.DataFrame:
    """
    Compute line-based features for each row.
    
    Features include:
    - active_lines: Number of currently active lines
    - hours_since_big_move: Hours since last line ended
    - line_momentum_6h: Net line activity in past 6 hours
    - trending_score: Directional bias of recent lines
    - reversal_potential: Opposite direction line activity
    """
    # Initialize feature columns
    n_rows = len(df)
    logging.debug(f"Computing line features for {n_rows} rows...")
    active_lines = np.zeros(n_rows)
    hours_since_big_move = np.full(n_rows, 168.0)  # Default to 1 week
    line_momentum_6h = np.zeros(n_rows)
    trending_score = np.zeros(n_rows)
    reversal_potential = np.zeros(n_rows)
    
    # Process each line
    all_lines = [(line, 1) for line in long_lines] + [(line, -1) for line in short_lines]
    logging.debug(f"Processing {len(all_lines)} total lines for feature extraction")
    
    last_progress = 0
    for idx in range(n_rows):
        # Progress update every 10%
        progress = int(idx / n_rows * 100)
        if progress >= last_progress + 10:
            logging.debug(f"  Line feature computation progress: {progress}% ({idx}/{n_rows} rows processed)")
            last_progress = progress
        # Count active lines
        active_count = 0
        recent_long_count = 0
        recent_short_count = 0
        
        for line, direction in all_lines:
            # Check if line is active at this index
            if line['start_idx'] <= idx <= line['end_idx']:
                active_count += 1
            
            # Check if line ended recently (within 168 hours)
            if line['end_idx'] < idx and (idx - line['end_idx']) < 168:
                hours_since = idx - line['end_idx']
                hours_since_big_move[idx] = min(hours_since_big_move[idx], hours_since)
            
            # Count lines in past 6 hours
            if line['end_idx'] >= idx - 6 and line['end_idx'] < idx:
                if direction > 0:
                    recent_long_count += 1
                else:
                    recent_short_count += 1
        
        active_lines[idx] = active_count
        line_momentum_6h[idx] = recent_long_count - recent_short_count
        
        # Calculate trending score and reversal potential
        if recent_long_count + recent_short_count > 0:
            trending_score[idx] = (recent_long_count - recent_short_count) / (recent_long_count + recent_short_count)
            reversal_potential[idx] = min(recent_long_count, recent_short_count) / max(recent_long_count, recent_short_count, 1)
    
    # Add features to DataFrame
    logging.debug("Adding line features to DataFrame...")
    df = df.with_columns([
        pl.Series('active_lines', active_lines),
        pl.Series('hours_since_big_move', hours_since_big_move),
        pl.Series('line_momentum_6h', line_momentum_6h),
        pl.Series('trending_score', trending_score),
        pl.Series('reversal_potential', reversal_potential)
    ])
    
    return df


def compute_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute temporal features from datetime column.
    """
    logging.debug("Extracting temporal features (hour, day of week)...")
    # Extract hour and day of week
    df = df.with_columns([
        pl.col('datetime').dt.hour().alias('hour_of_day'),
        pl.col('datetime').dt.weekday().alias('day_of_week')
    ])
    
    return df


def compute_price_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute price-based features including returns, acceleration, and volatility.
    """
    logging.debug("Computing price-based features...")
    
    # Returns over different periods
    logging.debug("  Computing returns over multiple periods...")
    for period in [1, 6, 12, 24, 48]:
        df = price_change_pct(df, period)
        # Rename to expected format
        df = df.rename({f'price_change_pct_{period}': f'ret_{period}h'})
    
    # Acceleration (change in returns)
    logging.debug("  Computing acceleration features...")
    df = df.with_columns([
        (pl.col('ret_6h') - pl.col('ret_6h').shift(6)).alias('accel_6h'),
        (pl.col('ret_24h') - pl.col('ret_24h').shift(24)).alias('accel_24h')
    ])
    
    # Distance from recent high/low
    logging.debug("  Computing distance from high/low features...")
    df = distance_from_high(df, period=24)
    df = distance_from_low(df, period=24)
    df = price_range_position(df, period=24)
    
    # Rename columns to match expected names
    df = df.rename({
        'distance_from_high': 'dist_from_high',
        'distance_from_low': 'dist_from_low',
        'price_range_position': 'position_in_range'
    })
    
    # Volatility features
    logging.debug("  Computing volatility features...")
    # First add returns column for volatility calculation
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns_temp')
    ])
    
    # Use rolling_volatility indicator
    for period in [6, 24]:
        df = rolling_volatility(df, 'returns_temp', period)
        df = df.rename({f'returns_temp_volatility_{period}': f'vol_{period}h'})
    
    # Volume expansion
    df = df.with_columns(
        (pl.col('vol_24h') / (pl.col('vol_6h') + 1e-8)).alias('vol_expansion')
    )
    
    # Clean up temp column
    df = df.drop('returns_temp')
    
    # Feature stats
    logging.debug("  Computing feature statistics...")
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
    """
    Create 3-class labels based on future price movements.
    
    Classes:
    - 0: No trade (neither long nor short criteria met)
    - 1: Long (positive return exceeds long threshold)
    - 2: Short (negative return exceeds short threshold)
    
    Args:
        df: DataFrame with price data
        long_threshold: Threshold for long trades (positive)
        short_threshold: Threshold for short trades (positive, will be negated)
        lookahead_hours: Hours to look ahead for price movements
    """
    close_prices = df.get_column('close')
    n_rows = len(df)
    labels = np.zeros(n_rows, dtype=int)
    logging.debug(f"Creating labels for {n_rows} rows with thresholds - Long: {long_threshold:.3%}, Short: {short_threshold:.3%}")
    
    last_progress = 0
    long_count = 0
    short_count = 0
    
    for idx in range(n_rows - lookahead_hours):
        # Progress update every 10%
        progress = int(idx / (n_rows - lookahead_hours) * 100)
        if progress >= last_progress + 10:
            logging.debug(f"  Label creation progress: {progress}% ({idx}/{n_rows - lookahead_hours} rows processed)")
            logging.debug(f"    Current counts - Long: {long_count}, Short: {short_count}, No trade: {idx - long_count - short_count}")
            last_progress = progress
        current_price = close_prices[idx]
        
        # Look at future prices
        future_24h_idx = min(idx + 24, n_rows - 1)
        future_48h_idx = min(idx + lookahead_hours, n_rows - 1)
        
        # Calculate max/min future prices
        future_prices = close_prices[idx:future_48h_idx+1]
        max_future_price = future_prices.max()
        min_future_price = future_prices.min()
        
        # Calculate returns
        max_future_ret = (max_future_price - current_price) / current_price
        min_future_ret = (min_future_price - current_price) / current_price
        
        # Also check specific horizons
        ret_24h = (close_prices[future_24h_idx] - current_price) / current_price
        ret_48h = (close_prices[future_48h_idx] - current_price) / current_price
        
        # Label logic with separate thresholds
        if max_future_ret >= long_threshold:
            if ret_48h > long_threshold or ret_24h > long_threshold:
                labels[idx] = 1  # LONG
                long_count += 1
        elif abs(min_future_ret) >= short_threshold:
            if ret_48h < -short_threshold or ret_24h < -short_threshold:
                labels[idx] = 2  # SHORT
                short_count += 1
        # else remains 0 (NO_TRADE)
    
    logging.debug(f"Label creation complete - Long: {long_count}, Short: {short_count}, No trade: {n_rows - lookahead_hours - long_count - short_count}")
    
    df = df.with_columns(pl.Series('label', labels))
    
    return df


class LGBWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for LightGBM to work with scikit-learn calibration.
    """
    def __init__(self, lgb_model):
        self.lgb_model = lgb_model
        self.scaler = StandardScaler()
        self.classes_ = np.array([0, 1, 2])
        self._is_fitted = False
        self._estimator_type = "classifier"  # Explicitly mark as classifier
        
    def fit(self, X, y):
        # Fit the scaler
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


def create_lgb_wrapper(lgb_model) -> LGBWrapper:
    """Create a wrapper for LightGBM model for calibration."""
    return LGBWrapper(lgb_model)


def apply_class_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Calculate sample weights to handle class imbalance.
    """
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    # Create sample weights
    sample_weights = np.zeros(len(y_train))
    for i, cls in enumerate(classes):
        mask = y_train == cls
        sample_weights[mask] = class_weights[i]
    
    return sample_weights