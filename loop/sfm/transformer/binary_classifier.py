'''
Transformer Binary Classifier - UEL Single File Model format
Bitcoin long-only trading strategy with transformer-based sequence modeling

This module implements a complete transformer-based binary classifier for cryptocurrency
regime prediction using 1-minute OHLC data with technical indicators and liquidity features.
The architecture follows UEL (Universal Experiment Loop) conventions for reproducible
machine learning experiments.
'''
"Given the current market state, is there a high probability that the next price move will be at least +X%?"

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix


from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.metrics.binary_metrics import binary_metrics
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import keras
from keras import backend as K
import gc
from keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import AdamW
from keras_hub.layers import RotaryEmbedding  
from loop.metrics.binary_metrics import binary_metrics
import polars as pl
import numpy as np
from loop.manifest import Manifest
from loop.manifest import _apply_fitted_transform
import loop.manifest
import datetime
from loop.manifest import Manifest
from loop.indicators import returns, wilder_rsi, macd, atr
from loop.features import vwap, volume_spike, price_range_position, ma_slope_regime, price_vs_band_regime, volume_ratio


def add_cyclical_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add cyclical time-based features for temporal pattern recognition.
    
    Creates sine/cosine encodings for hour, minute, and day components to capture
    periodic market behavior patterns. These cyclical features help the transformer
    model understand temporal relationships in financial data.
    
    Args:
        df: Polars DataFrame containing 'datetime' column
        
    Returns:
        DataFrame with added cyclical features: sin/cos_hour, sin/cos_minute, sin/cos_day
    """
    df = df.with_columns([
        pl.col('datetime').dt.hour().alias('hour'),
        pl.col('datetime').dt.minute().alias('minute'),
        pl.col('datetime').dt.day().alias('day'),
        pl.col('datetime').dt.weekday().alias('weekday'),  # Monday=0, Sunday=6
        # pl.col('datetime').dt.days_in_month().alias('days_in_month'),
    ])
    df = df.with_columns([
        np.sin(2 * np.pi * pl.col('hour') / 24).alias('sin_hour'),
        np.cos(2 * np.pi * pl.col('hour') / 24).alias('cos_hour'),
        np.sin(2 * np.pi * pl.col('minute') / 60).alias('sin_minute'),
        np.cos(2 * np.pi * pl.col('minute') / 60).alias('cos_minute'),
        # Day-of-month encoding using actual days in month
        # np.sin(2 * np.pi * (pl.col('day') - 1) / pl.col('days_in_month')).alias('sin_day'),
        # np.cos(2 * np.pi * (pl.col('day') - 1) / pl.col('days_in_month')).alias('cos_day'),
        # Day-of-week encoding (Monday=0, Sunday=6)
        np.sin(2 * np.pi * pl.col('weekday') / 7).alias('sin_weekday'),
        np.cos(2 * np.pi * pl.col('weekday') / 7).alias('cos_weekday'),
    ])
    return df.drop(['hour', 'minute', 'day', 'weekday'])


def regime_target(df: pl.DataFrame, prediction_window: int, pct_move_threshold: float) -> pl.DataFrame:
    """
    Label each row as a 'long regime' if forward windowed return exceeds pct_move_threshold.
    
    Creates binary labels based on future price movement over a specified window.
    This target engineering approach identifies periods favorable for long positions
    by comparing forward returns against historical thresholds.
    
    Args:
        df: Input DataFrame with 'close' prices
        prediction_window: forecast horizon in minutes  
        pct_move_threshold: return threshold to classify as regime (e.g., 0.005 for 0.5%)
        
    Returns:
        DataFrame with added 'target_regime' binary column
    """
    closes = df['close'].to_numpy()
    windowed_returns = np.zeros_like(closes)
    # Compute forward window return for every row
    for i in range(len(closes)):
        end_idx = min(i + prediction_window, len(closes) - 1)
        windowed_returns[i] = (closes[end_idx] - closes[i]) / closes[i]
    # Determine threshold on training split only (in manifest, handled via fitted param)
    label = (windowed_returns > pct_move_threshold).astype(int)
    df = df.with_columns([pl.Series('target_regime', label)])

    # Print description of label distribution for sanity check
    num_total = len(label)
    num_positive = int(label.sum())
    num_negative = num_total - num_positive
    ratio_positive = num_positive / num_total
    ratio_negative = num_negative / num_total

    print(f"[LABEL SANITY CHECK] pct_move_threshold={pct_move_threshold:.5f}")
    print(f"  Total samples: {num_total}")
    print(f"  Positive regime labels (1): {num_positive} ({ratio_positive*100:.2f}%)")
    print(f"  Negative regime labels (0): {num_negative} ({ratio_negative*100:.2f}%)")
    print("-" * 40)

    return df



def my_make_fitted_scaler(param_name: str, transform_class):
    """
    Custom scaler factory for Polars DataFrame compatibility with sklearn scalers.
    
    Creates a fitted scaler that works with Polars DataFrames while preserving
    the manifest's fitting paradigm. This ensures proper data leakage prevention
    by fitting scalers only on training data.
    """
    class PolarsColumnScaler:
        """
        Polars-compatible wrapper for sklearn preprocessing scalers.
        
        This wrapper class adapts sklearn scalers to work with Polars DataFrames
        while maintaining column order and data types. It automatically identifies
        numeric feature columns and excludes datetime and target columns.
        """
        def __init__(self, data: pl.DataFrame):
            """
            Initialize and fit scaler on training data.
            
            Args:
                data: Training DataFrame used to fit the scaler parameters
            """
            # Exclude non-feature columns; adjust target name if needed
            drop_cols = ['datetime', 'target_regime']
            numeric_dtypes = (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
            self.feature_cols = [
                c for c in data.columns
                if c not in drop_cols and data[c].dtype in numeric_dtypes
            ]
            # Fit sklearn scaler on numpy array (no feature names)
            X = data.select(self.feature_cols).to_numpy()
            self.scaler = transform_class()
            self.scaler.fit(X)


        def transform(self, data: pl.DataFrame) -> pl.DataFrame:
            """
            Apply fitted scaling transformation to new data.
            
            Args:
                data: DataFrame to transform using fitted scaler parameters
                
            Returns:
                DataFrame with scaled feature columns
            """
            # Select the exact same columns in the same order
            X = data.select(self.feature_cols).to_numpy()
            Xs = self.scaler.transform(X)
            # Replace scaled columns back into a Polars DataFrame
            cols = [pl.Series(name, Xs[:, j]) for j, name in enumerate(self.feature_cols)]
            return data.with_columns(cols)


    # Return a "fitted transform entry" that Manifest expects
    return (
        [(param_name, lambda df: PolarsColumnScaler(df), {})],
        _apply_fitted_transform,
        {'fitted_transform': param_name}
    )

loop.manifest.make_fitted_scaler = my_make_fitted_scaler


def manifest():
    
    """
    Configure the data processing pipeline and experiment manifest.
    
    This function defines the complete data preprocessing pipeline including:
    - Data splitting strategy (8:1:2 ratio for train:val:test)
    - Required input columns for OHLC and liquidity data
    - Feature engineering steps (cyclical time features)
    - Target variable creation with regime labeling
    - Data scaling using StandardScaler
    
    The manifest ensures reproducible experiments and proper data leakage prevention
    by fitting all transformations only on training data.
    
    Returns:
        Configured Manifest object ready for data preparation
    """
    def base_bar_formation(data: pl.DataFrame, **params) -> pl.DataFrame:
        """Identity function for basic bar data - no additional processing needed."""
        return data


    # Define required columns for cryptocurrency market data with liquidity features
    required_cols = [
        'datetime', 'open', 'high', 'low', 'close', 'mean', 'std', 'median', 'iqr',
        'volume', 'maker_ratio', 'no_of_trades', 'open_liquidity', 'high_liquidity',
        'low_liquidity', 'close_liquidity', 'liquidity_sum', 'maker_volume', 'maker_liquidity'
    ]
    return(
        Manifest()
        .set_split_config(7, 2, 1)  
        .set_bar_formation(base_bar_formation, bar_type='base')
        .set_required_bar_columns(required_cols)
        .add_feature(add_cyclical_features)
        # Indicators
        .add_indicator(wilder_rsi, period='rsi_period')
        .add_indicator(macd, fast_period='macd_fast', slow_period='macd_slow')
        .add_indicator(atr, period='atr_period')
        # Features
        .add_feature(vwap, price_col='close', volume_col='volume')
        .add_feature(volume_ratio, period='vr_period')
        .add_feature(volume_spike, period='vol_spike_period', use_zscore=True)
        .add_feature(price_range_position, period='range_pos_period')
        .add_feature(ma_slope_regime, period='ma_regime_period')
        .with_target('target_regime')
        .add_fitted_transform(regime_target)  # Create binary regime labels
        .with_params(prediction_window='prediction_window', pct_move_threshold='pct_move_threshold')
        .done()
        .set_scaler(RobustScaler)
    )



def params():
    """
    SFM-Compliant Hyperparameter Definition for Transformer Binary Classifier

    Returns an optimized hyperparameter sweep space for transformer-based Bitcoin trading models.

    - Parameter space spans architectural, optimization, and sequence modeling dimensions.
    - Only includes valid (d_model, num_heads) pairs to prevent attention layer config errors.
    - All selection ranges use closer gaps for more granular sweeps, as per convergence best practices.

    Returns:
        dict: Parameter grid for sweep. Each key is a parameter name; each value is a list of values to scan.
    """

    sweep_space = {

        # =========================
        # Model Architecture Parameters
        # =========================

        'd_model': [32, 40, 48, 56, 64, 72, 80, 96, 128],  # Model width: must be divisible by num_heads
        'num_heads': [2, 4, 8],                            # Number of attention heads
        'num_layers': [2, 3, 4, 5],                        # Number of transformer blocks
        'dropout': [0.15, 0.17, 0.18, 0.22, 0.35, 0.4, 0.42, 0.5],  # Dropout rates for regularization
        'positional_encoding_type': ['rotary'],            # Type of positional encoding

        # =========================
        # Optimization & Training Parameters
        # =========================

        'learning_rate': [1e-4, 1.5e-4, 2e-4, 2.5e-4, 4e-4, 6e-4, 8e-4, 1e-3],  # Learning rate
        'batch_size': [32, 48, 64, 80, 96, 112, 128],                            # Batch size
        'weight_decay': [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1],    # Increased weight decay (L2 regularization)
        'epochs': [50, 65, 75, 100],                                             # Number of training epochs
        'seed': [42, 77, 2025],                                                  # Random seeds for reproducibility

        # =========================
        # Data & Sequence Modeling Parameters
        # =========================

        'seq_length': [45, 60, 90, 105, 120],            # Input sequence length
        'prediction_window': [5, 10, 30, 60, 90],        # Prediction horizon

        # =========================
        # Target Engineering Parameters
        # =========================

        'pct_move_threshold': [0.005, 0.0010, 0.0012, 0.0015, 0.0018, 0.0020, 0.0024],  # Thresholds for classifying significant price moves

        # Indicator Parameters
        'rsi_period': [10, 14, 21],               # Short, classic, and slow
        'macd_fast': [8, 10, 12],                 # Faster for minute bars
        'macd_slow': [17, 21, 26],                # Smoothed bassline for microtrends
        'macd_signal': [5, 7, 9],                 # Smoothing signal
        'atr_period': [5, 7, 10],                 # Short-range volatility

        # Feature Parameters
        'vr_period': [10, 14, 21],                # Volume Ratio periods
        'vol_spike_period': [14, 20, 25],         # Range of z-score windows for spike detection
        'range_pos_period': [10, 14, 20],         # Local context; tighter for faster, broader for slower environments
        'ma_regime_period': [14, 20, 30],         # Trend context—test different regime perceptions
    }
    return sweep_space



def is_valid_datetime(dt):
    # Accepts datetime.datetime, numpy.datetime64, rejects None and others
    if dt is None:
        return False
    if isinstance(dt, (datetime.datetime, np.datetime64)):
        return True
    return False


def prep(data, round_params, manifest):
    """
    Leakproof data preparation with proper sequence alignment for transformer models.
    ...
    """
    # Prepare model-ready splits via manifest
    data_dict = manifest.prepare_data(data, round_params)
    
    print("\n" + "="*60)
    print("NaN SANITIZATION CHECK")
    print("="*60)
    
    # Check for NaNs in each split (numeric columns only)
    for split_name in ['x_train', 'x_val', 'x_test']:
        if split_name in data_dict:
            X = data_dict[split_name]
            if isinstance(X, pl.DataFrame):
                # Get numeric column names only
                numeric_cols = [
                    c for c in X.columns
                    if X[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
                ]

                # Print NaN counts per numeric column
                nan_counts = X.select([
                    pl.col(col).is_nan().sum().alias(col) 
                    for col in numeric_cols
                ])
                print(f"\n{split_name} NaN counts per numeric column:")
                print(nan_counts)
                
                print(f"{split_name} shape before NaN removal: {X.shape}")
                
                # Create boolean mask as a Series (not Expr)
                mask = X.select(
                    ~pl.any_horizontal([pl.col(col).is_nan() for col in numeric_cols])
                ).to_series()
                
                # Remove rows with ANY NaN in numeric columns
                X_clean = X.filter(mask)
                print(f"{split_name} shape after NaN removal: {X_clean.shape}")
                print(f"Rows dropped: {len(X) - len(X_clean)}")
                
                # Update data_dict with cleaned data
                data_dict[split_name] = X_clean
                
                # Align corresponding y labels using the SAME mask
                y_key = split_name.replace('x_', 'y_')
                if y_key in data_dict:
                    y = data_dict[y_key]
                    if isinstance(y, pl.Series):
                        data_dict[y_key] = y.filter(mask)
                    elif isinstance(y, pl.DataFrame):
                        data_dict[y_key] = y.filter(mask)
    
    # Final verification: assert NO NaNs remain in numeric columns
    for split_name in ['x_train', 'x_val', 'x_test']:
        if split_name in data_dict:
            X = data_dict[split_name]
            if isinstance(X, pl.DataFrame):
                numeric_cols = [
                    c for c in X.columns
                    if X[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
                ]
                has_nan = X.select([
                    pl.any_horizontal([pl.col(col).is_nan() for col in numeric_cols])
                ]).to_series().any()
                assert not has_nan, f"ERROR: {split_name} still contains NaN in numeric columns after cleaning!"
    
    print("\n✓ All splits verified NaN-free in numeric columns")
    print("="*60 + "\n")
    
    # Compute effective window length (for safest windowing across splits)
    seq_len_eff = min(
        round_params['seq_length'],
        data_dict['x_train'].shape[0],
        data_dict['x_val'].shape[0],
        data_dict['x_test'].shape[0],
    )


# Alignment edit: Only add valid (non-null) missing datetimes
    if '_alignment' in data_dict:
        align   = dict(data_dict['_alignment'])
        missing = list(align.get('missing_datetimes', []))
        test_dt = align.get('test_datetimes', None)

        valid_dt = []
        if test_dt is not None and hasattr(test_dt, '__getitem__') and hasattr(test_dt, '__len__') and seq_len_eff > 1:
            dt_slice = test_dt[:seq_len_eff - 1]
            valid_dt = [dt for dt in dt_slice if is_valid_datetime(dt)]
        # Also filter any existing 'missing' from historical sources
        missing = [dt for dt in missing if is_valid_datetime(dt)]
        missing += valid_dt
        align['missing_datetimes'] = missing
        data_dict['_alignment'] = align

        # FINAL SANITIZATION before returning
        missing_filtered = [x for x in align['missing_datetimes'] if is_valid_datetime(x)]    
        data_dict['_alignment']['missing_datetimes'] = missing_filtered
       

    # NumPy conversions
    exclude_cols = ['datetime', 'target_regime']
    numeric_dtypes = (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    
    # Convert feature DataFrames to NumPy (numeric columns only)
    for k in ['x_train', 'x_val', 'x_test']:
        if isinstance(data_dict[k], pl.DataFrame):
            df = data_dict[k]
            # Select only numeric feature columns
            feature_cols = [
                c for c in df.columns 
                if c not in exclude_cols and df[c].dtype in numeric_dtypes
            ]
            # Convert to NumPy array, then cast dtype if needed
            arr = df.select(feature_cols).to_numpy()
            data_dict[k] = arr.astype(np.float64)
    
    # Convert target Series/DataFrames to NumPy (already numeric)
    for k in ['y_train', 'y_val', 'y_test']:
        if isinstance(data_dict[k], pl.Series):
            data_dict[k] = data_dict[k].to_numpy().ravel().astype(np.int32)
        elif isinstance(data_dict[k], pl.DataFrame):
            data_dict[k] = data_dict[k].to_numpy().ravel().astype(np.int32)

    # Verify dtypes before returning
    print("\nDtype verification after conversion:")
    for k in ['x_train', 'x_val', 'x_test']:
        print(f"{k} dtype: {data_dict[k].dtype}, shape: {data_dict[k].shape}")
    for k in ['y_train', 'y_val', 'y_test']:
        print(f"{k} dtype: {data_dict[k].dtype}, shape: {data_dict[k].shape}")
    
    # Window y_test as before
    raw_y_test = data_dict['y_test']
    
    return data_dict


def transformer_encoder_block(d_model, num_heads, dropout, use_rotary):
    """
    Build a single transformer encoder block with optional rotary positional encoding.
    
    This function creates the core building block of the transformer architecture:
    - Multi-head self-attention mechanism for capturing sequence relationships
    - Feed-forward network for non-linear transformations  
    - Residual connections and layer normalization for training stability
    - Optional rotary positional encoding for improved position awareness
    
    The encoder block maintains the d_model dimensionality throughout all operations,
    ensuring compatible residual connections and stable gradient flow.
    
    Args:
        d_model: Model embedding/hidden dimension size
        num_heads: Number of attention heads for parallel attention computation
        dropout: Dropout probability for regularization
        use_rotary: Whether to apply rotary positional encoding
        
    Returns:
        Function that applies transformer encoder block to input tensors
        Input/output shape: (batch, timesteps, d_model)
    """
    def block(x):
        # Instead of adding or learning position vectors, 
        # rotary encoding rotates the feature space in a way that preserves relative positions, 
        # which can help with generalization in time series.
        if use_rotary:
            x = RotaryEmbedding(sequence_axis=1, feature_axis=2)(x)
            
        # Multi-head self-attention with residual connection
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )(x, x)
        attn_output = Dropout(dropout)(attn_output)
        out1 = Add()([x, attn_output])
        out1 = LayerNormalization()(out1)

        # Feed-forward network with residual connection
        ff = Dense(4 * d_model, activation='relu')(out1)  # Standard 4x expansion
        ff = Dropout(dropout)(ff)
        ff = Dense(d_model)(ff)  # Project back to d_model
        out2 = Add()([out1, ff])
        out2 = LayerNormalization()(out2)
        return out2
    return block



def make_windows(X2d: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Build sliding windows over row-major time series data for sequence modeling.
    
    Creates overlapping sequences from tabular time series data, where each window
    contains seq_len consecutive timesteps. This transformation is essential for
    feeding time series data into sequence models like transformers.
    
    Args:
        X2d: Feature matrix of shape (n_rows, n_features)
        y: Target vector of shape (n_rows,)
        seq_len: Length of each sequence window
        
    Returns:
        X3: Windowed features of shape (n_samples, seq_len, n_features)
        y2: Aligned targets of shape (n_samples,) 
        
    Raises:
        ValueError: If insufficient data rows for the specified sequence length
    """
    if len(X2d) < seq_len:
        raise ValueError(f"Not enough rows ({len(X2d)}) for seq_length={seq_len}")
    X3 = np.stack([X2d[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(X2d))], axis=0)
    y2 = y[seq_len - 1:]
    return X3, y2



def model(data, round_params):
    """
    Build, train, and evaluate transformer-based binary regime classifier.
    
    This function implements the complete modeling pipeline:
    1. Model Architecture: Transformer encoder with input projection and pooling
    2. Training Setup: AdamW optimizer with early stopping and label smoothing  
    3. Sequence Processing: Sliding window conversion for time series data
    4. Evaluation: Binary classification metrics on windowed test predictions
    
    The model uses a projected embedding approach where input features are first
    projected to d_model dimensions, then processed through transformer layers,
    and finally pooled and classified. This design allows flexible input feature
    dimensions while maintaining efficient attention computations.
    
    Key Features:
    - Rotary positional encoding for improved sequence understanding
    - Early stopping to prevent overfitting on noisy financial data
    - UEL-compatible prediction alignment (no padding/invalid predictions)
    
    Args:
        data: Preprocessed data dictionary with train/val/test splits
        round_params: Hyperparameter configuration for current experiment round
        
    Returns:
        Dictionary containing metrics, predictions, and model artifacts for UEL logging
    """
    # --- Unpack hyperparameters ---
    d_model = round_params['d_model']                      # Transformer embedding dimension
    num_heads = round_params['num_heads']                  # Number of attention heads
    num_layers = round_params['num_layers']                # Number of transformer encoder blocks
    dropout = round_params['dropout']                      # Dropout rate for regularization
    learning_rate = round_params['learning_rate']          # Learning rate for optimizer
    weight_decay = round_params.get('weight_decay', 0.0)   # L2 regularization (default 0.0 if not set)
    batch_size = round_params['batch_size']                # Batch size for training
    epochs = round_params['epochs']                        # Number of training epochs
    seed = round_params.get('seed', 42)                    # Random seed for reproducibility (default 42)
    seq_length = round_params['seq_length']                # Sequence length for windowing
    use_rotary = round_params.get('positional_encoding_type', 'rotary') == 'rotary'  # Use rotary positional encoding if specified
    pct_move_threshold = round_params['pct_move_threshold'] # Threshold for regime target labeling

    np.random.seed(seed)                                   # Set numpy random seed for reproducibility

    # if d_model % num_heads != 0:
    #     raise ValueError(f"Invalid transformer configuration: d_model ({d_model}) not divisible by num_heads ({num_heads})")
    # --- Extract data arrays ---
    X_train, X_val, X_test = data['x_train'], data['x_val'], data['x_test'] # Inputs
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test'] # Binary Labels

    print("Before windowing -- x_train:", X_train.shape, "y_train:", y_train.shape)
    print("x_val:", X_val.shape, "y_val:", y_val.shape)
    print("x_test:", X_test.shape, "y_test:", y_test.shape)


    # --- Sequence Windowing for Transformer Input ---
    # Compute effective sequence length (limited by smallest split size)
    # This ensures the window size fits in all splits, avoiding index errors.
    seq_len_eff = min(seq_length, X_train.shape[0], X_val.shape[0], X_test.shape[0])

    # Convert raw 2D arrays into 3D sliding windows for sequence modeling.
    # Each sample is a sequence of 'seq_len_eff' consecutive timesteps.
    X_train, y_train = make_windows(X_train, y_train, seq_len_eff)
    X_val,   y_val   = make_windows(X_val,   y_val,   seq_len_eff)
    X_test,  y_test  = make_windows(X_test,  y_test,  seq_len_eff)

    # Store windowed arrays back into the data dictionary for reference.
    data['X_train'] = X_train
    data['y_train'] = y_train
    data['X_val'] = X_val
    data['y_val'] = y_val
    data['X_test'] = X_test
    data['y_test'] = y_test

    # Print shapes after windowing for debugging and verification.
    print("After windowing -- X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # Number of features per timestep (needed for model input shape).
    n_features = X_train.shape[2]
    print(f"Number of features: {n_features}")
    # Update sequence length to match effective window size for model input.
    seq_length = seq_len_eff  # ensure consistency for Input shape


    # --- Build transformer architecture ---
    # Input projection: map features to transformer embedding space
    input_layer = Input(shape=(seq_length, n_features), name='input')
    x = Dense(d_model, name='input_projection')(input_layer) # Dense layer maps raw features to the transformer embedding dimension (d_model).
    
    # Stack transformer encoder blocks
    for _ in range(num_layers):
        x = transformer_encoder_block(d_model, num_heads, dropout, use_rotary)(x)
        
    # Global pooling and classification head
    x = GlobalAveragePooling1D()(x)  # Aggregate sequence information
    x = Dropout(dropout)(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    model_tf = Model(inputs=input_layer, outputs=output)

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model_tf.compile(
        optimizer=optimizer,  # type: ignore
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Add early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,  # Stop if val_loss doesn't improve for 10 epochs
        restore_best_weights=True,  # Restore weights from best epoch
        start_from_epoch= 10,  # Start checking after 15 epochs
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR by half
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    model_tf.fit(
        X_train, y_train,                      # Training features and labels
        validation_data=(X_val, y_val),        # Validation data for monitoring
        batch_size=batch_size,                 # Number of samples per gradient update
        epochs=epochs,                         # Number of training epochs
        verbose=1,                             # Show training progress       # type: ignore   
        callbacks=[early_stop, reduce_lr]  
    )

    

    test_probs = model_tf.predict(X_test, batch_size=batch_size, verbose=1).flatten()  # Predicted probabilities for test set #type: ignore
    test_preds = (test_probs > 0.5).astype(int)                                       # Convert probabilities to binary predictions if prob > 0.5
    print("test_preds shape:", test_preds.shape, "test_probs shape:", test_probs.shape, "y_test shape:", y_test.shape)
    assert len(test_preds) == len(y_test), "Predictions and test labels are not aligned!"


    # --- Compute evaluation metrics ---
    round_results = binary_metrics(data={'y_test': data['y_test']}, preds=test_preds, probs=test_probs)
    round_results['_preds'] = test_preds          # UEL will collect for logging
    round_results['models'] = [model_tf]          # UEL will collect trained model

    # --- Store validation results and metadata ---
    # Store validation arrays inside extras so not logged as columns
    val_probs = model_tf.predict(X_val, batch_size=batch_size, verbose=1).flatten() # Validation probabilities #type: ignore
    val_preds = (val_probs > 0.5).astype(int)                                       # Validation binary predictions
    round_results['extras'] = {
        'seq_len_eff': seq_len_eff,        # Actual sequence length used
        'val_preds': val_preds,            # Validation set predicted labels
        'val_probs': val_probs,            # Validation set predicted probabilities 
        'val_targets': y_val               # Validation targets
    }

    # --- Clean up Keras session to prevent memory leaks in UEL loops ---
    print("Final round_results keys:", round_results.keys())
    #Final round_results keys: dict_keys(['recall', 'precision', 'fpr', 'auc', 'accuracy', '_preds', 'models', 'extras'])
    print("First 10 preds:", test_preds[:10])
    print("First 10 y_test:", y_test[:10])

    K.clear_session()  # Clear Keras backend session
    del model_tf
    gc.collect()
    return round_results  # Return results dictionary for UEL logging
