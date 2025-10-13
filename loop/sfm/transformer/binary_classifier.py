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
        .set_split_config(8, 1, 2)  # 80% train, 10% validation, 20% test
        .set_bar_formation(base_bar_formation, bar_type='base')
        .set_required_bar_columns(required_cols)
        .add_feature(add_cyclical_features)  # Add temporal cyclical features
        .with_target('target_regime')
        .add_fitted_transform(regime_target)  # Create binary regime labels
        .with_params(prediction_window='prediction_window', pct_move_threshold='pct_move_threshold')
        .done()
        .set_scaler(RobustScaler)  # Apply standard scaling to features
    )



def params():
    """
    Define hyperparameter search space for transformer model optimization.
    
    This function specifies the parameter ranges for automated hyperparameter tuning.
    Parameters are organized into categories:
    - Model architecture: embedding dimensions, attention heads, layers
    - Training dynamics: learning rate, batch size, regularization  
    - Sequence modeling: context length and prediction windows
    - Target engineering: regime detection thresholds
    
    Returns:
        Dictionary of parameter names mapped to lists of candidate values
    """
    return {
        # Model architecture and training params
        'd_model':        [32, 48, 64],            # Model width
        'num_heads':      [2, 4, 8],                   # More heads: try up to 8
        'num_layers':     [1, 2, 3, 4],                # Up to 4 transformer blocks
        'dropout':        [0.002, 0.05, 0.1, 0.15, 0.2, 0.25],         # Dropout for regularization
        # Optimization/training
        'learning_rate':  [1e-4, 5e-4, 1e-3, 2e-3],    # Extended, with finer gradation
        'batch_size':     [16, 32, 64, 128],           # Try small and large batches
        'weight_decay':   [0.0, 1e-5, 1e-4, 1e-3],     # Add smaller decay values
        'epochs':         [50, 75, 100, 150],           # Try longer training
        'seed':           [42, 77, 2025],              # Different initialization seeds

        # Sequence modeling
        'seq_length':     [45, 60, 90, 120],       # Expand context window
        'prediction_window': [30, 60, 90],             # Multiple regime horizons

        # Model tricks
        'positional_encoding_type': ['rotary'], # Try classic, rotary, etc.

        # Target engineering
        'pct_move_threshold': [0.003, 0.0024, 0.0015, 0.0012],
        'target_shift':      [0, 1],   # Regularization for noisy regime signals
    }


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
    
    This function performs critical data preparation steps:
    - Applies manifest transformations (scaling, feature engineering, target creation)
    - Handles sequence windowing alignment to prevent data leakage
    - Converts Polars DataFrames to NumPy arrays for Keras compatibility
    - Manages test set alignment for proper prediction logging
    
    The prep function ensures that the first (seq_length-1) test predictions are marked
    as missing since they don't have sufficient historical context, maintaining
    integrity for evaluation metrics and logging.
    
    Args:
        data: Raw input DataFrame
        round_params: Current hyperparameter configuration
        manifest: Configured manifest with all transformations
        
    Returns:
        Dictionary containing train/val/test splits as NumPy arrays with alignment info
    """
    # Prepare model-ready splits via manifest
    data_dict = manifest.prepare_data(data, round_params)


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
    for k in ['x_train', 'x_val', 'x_test']:
        if isinstance(data_dict[k], pl.DataFrame):
            data_dict[k] = data_dict[k].to_numpy()
    for k in ['y_train', 'y_val', 'y_test']:
        if isinstance(data_dict[k], pl.Series):
            data_dict[k] = data_dict[k].to_numpy().ravel()
        elif isinstance(data_dict[k], pl.DataFrame):
            data_dict[k] = data_dict[k].to_numpy().ravel()

    # Window y_test as before
    raw_y_test = data_dict['y_test']
    if len(raw_y_test) > seq_len_eff - 1:
        data_dict['y_test'] = raw_y_test[seq_len_eff - 1:]
        data_dict['_raw_y_test'] = raw_y_test


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
    - Label smoothing for calibrated probability outputs
    - UEL-compatible prediction alignment (no padding/invalid predictions)
    
    Args:
        data: Preprocessed data dictionary with train/val/test splits
        round_params: Hyperparameter configuration for current experiment round
        
    Returns:
        Dictionary containing metrics, predictions, and model artifacts for UEL logging
    """
    # --- Unpack hyperparameters ---
    d_model = round_params['d_model']
    num_heads = round_params['num_heads']
    num_layers = round_params['num_layers']
    dropout = round_params['dropout']
    learning_rate = round_params['learning_rate']
    weight_decay = round_params.get('weight_decay', 0.0)
    batch_size = round_params['batch_size']
    epochs = round_params['epochs']
    seed = round_params.get('seed', 42)
    seq_length = round_params['seq_length']
    use_rotary = round_params.get('positional_encoding_type', 'rotary') == 'rotary'
    pct_move_threshold = round_params['pct_move_threshold']

    np.random.seed(seed)


    # --- Extract data arrays ---
    X_train, X_val, X_test = data['x_train'], data['x_val'], data['x_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    print("Before windowing -- x_train:", X_train.shape, "y_train:", y_train.shape)
    print("x_val:", X_val.shape, "y_val:", y_val.shape)
    print("x_test:", X_test.shape, "y_test:", y_test.shape)


    # Compute effective sequence length (limited by smallest split size)
    seq_len_eff = min(seq_length, X_train.shape[0], X_val.shape[0], X_test.shape[0]) #This line ensures that the window size used for all splits is the largest possible value that fits in every split, so the model can be trained and evaluated without issues, regardless of how the data is split or how much data is available in each set.
    X_train, y_train = make_windows(X_train, y_train, seq_len_eff)
    X_val,   y_val   = make_windows(X_val,   y_val,   seq_len_eff)
    X_test,  y_test  = make_windows(X_test,  y_test,  seq_len_eff)

    print("After windowing -- X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)


    n_features = X_train.shape[2]
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


    # --- Compile model with advanced training configuration ---
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model_tf.compile(
        optimizer=optimizer,  # type: ignore
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )


    model_tf.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=0  # type: ignore
    )


    # --- Generate test predictions ---
    # WARNING: Do not use a different/raw test label vector!
    # Only use data['y_test'] for predictions and Log/metrics.
    test_probs = model_tf.predict(X_test, batch_size=batch_size, verbose=0).flatten() # type: ignore
    test_preds = (test_probs > 0.5).astype(int)

    print("Test pred min/max:", test_preds.min(), test_preds.max(),
      "y_test min/max:", y_test.min(), y_test.max())
    print("test_preds shape:", test_preds.shape, "test_probs shape:", test_probs.shape, "y_test shape:", y_test.shape)
    # assert len(test_preds) == len(y_test), "Predictions and test labels are not aligned!"

    # --- Compute evaluation metrics ---
    # Pass windowed y_test to metrics for proper alignment
    round_results = binary_metrics(data={'y_test': data['y_test']}, preds=test_preds, probs=test_probs)


    # --- Prepare UEL artifacts ---
    # UEL artifacts (UEL pops _preds and models, extras are kept out of log columns)
    round_results['_preds'] = test_preds          # UEL will collect for logging
    round_results['models'] = [model_tf]          # UEL will collect trained model


    # --- Store validation results and metadata ---
    # Store validation arrays inside extras so not logged as columns
    val_probs = model_tf.predict(X_val, batch_size=batch_size, verbose=0).flatten() # type: ignore
    val_preds = (val_probs > 0.5).astype(int)
    round_results['extras'] = {
        'seq_len_eff': seq_len_eff,        # Actual sequence length used
        'val_preds': val_preds,            # Validation predictions
        'val_probs': val_probs,   
        'val_targets': y_val        # Validation probabilities
    }

    # Clean up Keras session to prevent memory leaks in UEL loops

    print("Final round_results keys:", round_results.keys())
    print("First 10 preds:", test_preds[:10])
    print("First 10 y_test:", y_test[:10])

    K.clear_session()
    gc.collect()
    return round_results