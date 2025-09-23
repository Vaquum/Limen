'''
Transformer Binary Classifier - UEL Single File Model format
Bitcoin long-only trading strategy with transformer-based sequence modeling
'''

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
from keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from keras.models import Model
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping
from keras_hub.layers import RotaryEmbedding  
from loop.metrics.binary_metrics import binary_metrics
import polars as pl
import numpy as np
from loop.manifest import Manifest
from loop.manifest import _apply_fitted_transform
import loop.manifest

def add_cyclical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add cyclical features for hour, minute, and day to df (Polars native)"""
    df = df.with_columns([
        (pl.col('datetime').dt.hour().alias('hour')),
        (pl.col('datetime').dt.minute().alias('minute')),
        (pl.col('datetime').dt.day().alias('day')),
    ])
    df = df.with_columns([
        (np.sin(2 * np.pi * pl.col('hour') / 24).alias('sin_hour')),
        (np.cos(2 * np.pi * pl.col('hour') / 24).alias('cos_hour')),
        (np.sin(2 * np.pi * pl.col('minute') / 60).alias('sin_minute')),
        (np.cos(2 * np.pi * pl.col('minute') / 60).alias('cos_minute')),
        (np.sin(2 * np.pi * pl.col('day') / 31).alias('sin_day')),
        (np.cos(2 * np.pi * pl.col('day') / 31).alias('cos_day')),
    ])
    return df.drop(['hour', 'minute', 'day'])

def regime_target(df: pl.DataFrame, prediction_window: int, target_quantile: float) -> pl.DataFrame:
    """
    Label each row as a 'long regime' if forward windowed return exceeds quantile threshold.
    - prediction_window: forecast horizon in minutes
    - target_quantile: quantile to use as threshold (e.g., 0.55)
    """
    closes = df['close'].to_numpy()
    windowed_returns = np.zeros_like(closes)
    # Compute forward window return for every row
    for i in range(len(closes)):
        end_idx = min(i + prediction_window, len(closes) - 1)
        windowed_returns[i] = (closes[end_idx] - closes[i]) / closes[i]
    # Determine quantile threshold on training split only (in manifest, handled via fitted param)
    quantile_cutoff = np.quantile(windowed_returns, target_quantile)
    label = (windowed_returns > quantile_cutoff).astype(int)
    df = df.with_columns([pl.Series('target_regime', label)])
    return df


# Keep these imports in your SFM file
import polars as pl
import numpy as np
from loop.manifest import _apply_fitted_transform
import loop.manifest

def my_make_fitted_scaler(param_name: str, transform_class):
    class PolarsColumnScaler:
        def __init__(self, data: pl.DataFrame):
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

# Register the override so Manifest.set_scaler(StandardScaler) uses it
loop.manifest.make_fitted_scaler = my_make_fitted_scaler

print(loop.manifest.make_fitted_scaler)


def manifest():
    def base_bar_formation(data: pl.DataFrame, **params) -> pl.DataFrame:
        return data

    required_cols = [
        'datetime', 'open', 'high', 'low', 'close', 'mean', 'std', 'median', 'iqr',
        'volume', 'maker_ratio', 'no_of_trades', 'open_liquidity', 'high_liquidity',
        'low_liquidity', 'close_liquidity', 'liquidity_sum', 'maker_volume', 'maker_liquidity'
    ]

    from sklearn.preprocessing import StandardScaler

    return(
        Manifest()
        .set_split_config(8, 1, 2)
        .set_bar_formation(base_bar_formation, bar_type='base')
        .set_required_bar_columns(required_cols)
        .add_feature(add_cyclical_features)
        .with_target('target_regime')
            .add_fitted_transform(regime_target)
                .fit_param(
                    '_regime_cutoff',
                    lambda df, prediction_window, target_quantile:
                        np.quantile(
                            (df['close'].shift(-prediction_window) - df['close']) / df['close'],
                            target_quantile
                        ),
                    prediction_window='prediction_window', target_quantile='target_quantile'
                )
                .with_params(prediction_window='prediction_window', target_quantile='target_quantile')
            .done()
        .set_scaler(StandardScaler)
    )


def params():
    return {
        # Model architecture and training params
        'd_model': [32, 64],             # Embedding/hidden size
        'num_heads': [2, 4],             # Multi-head attention count
        'num_layers': [1, 2],            # Transformer encoder layers
        'dropout': [0.1, 0.2],           # Dropout for regularization
        'learning_rate': [1e-3, 5e-4],   # Adam optimizer learning rate
        'batch_size': [32, 64],          # Batch size
        'weight_decay': [0.0, 1e-4],
        'epochs': [5, 10],              # Number of epochs (low for speed)
        'seed': [42],                   # Random seed for reproducibility
        'early_stopping_patience': [3],  # Early stopping patience

        # Sequence and regime context params
        'seq_length': [30, 60],          # Number of context bars (1-min bars = 30-60min)
        'prediction_window': [60],       # Window to classify regime ahead (in minutes)
        'positional_encoding_type': ['rotary'], # Positional encoding

        # Target engineering
        'target_quantile': [0.45, 0.55],
        'target_shift': [0],             # [0] means start window immediately after context

        # Output regularization/calibration
        'label_smoothing': [0.0, 0.1],   # Regularization for noisy regime signals
    }


def prep(data, round_params, manifest):
    """
    Delegate to manifest's prepare_data for split-first leakproof prep.
    data: Polars DataFrame
    round_params: dict with single values for each param
    manifest: manifest() instance from this file
    Returns: data_dict ready for model()
    """
    data_dict = manifest.prepare_data(data, round_params)
    # --- Ensure model inputs are NumPy arrays ---
    for k in ['x_train', 'x_val', 'x_test']:
        if isinstance(data_dict[k], pl.DataFrame):
            data_dict[k] = data_dict[k].to_numpy()
    for k in ['y_train', 'y_val', 'y_test']:
        if isinstance(data_dict[k], pl.Series):
            data_dict[k] = data_dict[k].to_numpy().ravel()
        elif isinstance(data_dict[k], pl.DataFrame):
            data_dict[k] = data_dict[k].to_numpy().ravel()
  
    print("Prep output shapes/types:")
    for k in ['x_train', 'x_val', 'x_test']:
        print(f"{k}: {type(data_dict[k])}, shape: {data_dict[k].shape}")
    for k in ['y_train', 'y_val', 'y_test']:
        print(f"{k}: {type(data_dict[k])}, shape: {data_dict[k].shape}")
    
    return data_dict

def transformer_encoder_block(d_model, num_heads, dropout, use_rotary):
    """
    Transformer encoder block operating in d_model space (residual-safe).
    Input/output shape: (batch, timesteps, d_model)
    """
    def block(x):
        if use_rotary:
            x = RotaryEmbedding(sequence_axis=1, feature_axis=2)(x)
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )(x, x)
        attn_output = Dropout(dropout)(attn_output)
        out1 = Add()([x, attn_output])
        out1 = LayerNormalization()(out1)

        ff = Dense(4 * d_model, activation='relu')(out1)
        ff = Dropout(dropout)(ff)
        ff = Dense(d_model)(ff)
        out2 = Add()([out1, ff])
        out2 = LayerNormalization()(out2)
        return out2
    return block


def make_windows(X2d: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Build sliding windows over row-major time series.
    X2d: (n_rows, n_features), y: (n_rows,)
    Returns:
      X3: (n_rows - seq_len + 1, seq_len, n_features)
      y2: (n_rows - seq_len + 1,)
    """
    if len(X2d) < seq_len:
        raise ValueError(f"Not enough rows ({len(X2d)}) for seq_length={seq_len}")
    X3 = np.stack([X2d[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(X2d))], axis=0)
    y2 = y[seq_len - 1:]
    return X3, y2


def model(data, round_params):
    """
    Builds, trains, and evaluates a transformer regime classifier.
    Keeps round_results scalar-only to avoid Polars schema errors in UEL.
    """
    # --- Unpack parameters ---
    d_model = round_params['d_model']
    num_heads = round_params['num_heads']
    num_layers = round_params['num_layers']
    dropout = round_params['dropout']
    learning_rate = round_params['learning_rate']
    weight_decay = round_params.get('weight_decay', 0.0)
    batch_size = round_params['batch_size']
    epochs = round_params['epochs']
    seed = round_params.get('seed', 42)
    label_smoothing = round_params.get('label_smoothing', 0.0)
    early_stopping_patience = round_params.get('early_stopping_patience', 3)
    seq_length = round_params['seq_length']
    use_rotary = round_params.get('positional_encoding_type', 'rotary') == 'rotary'

    np.random.seed(seed)

    # --- Data to arrays ---
    X_train, X_val, X_test = data['x_train'], data['x_val'], data['x_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    # --- Robust windowing: cap seq_length by smallest split length ---
    seq_len_eff = min(seq_length, X_train.shape[0], X_val.shape[0], X_test.shape[0])
    X_train, y_train = make_windows(X_train, y_train, seq_len_eff)
    X_val,   y_val   = make_windows(X_val,   y_val,   seq_len_eff)
    X_test,  y_test  = make_windows(X_test,  y_test,  seq_len_eff)
    n_features = X_train.shape[2]
    seq_length = seq_len_eff  # ensure consistency for Input shape

    # --- Build model: project inputs to d_model, then encoder blocks ---
    input_layer = Input(shape=(seq_length, n_features), name='input')
    x = Dense(d_model, name='input_projection')(input_layer)
    for _ in range(num_layers):
        x = transformer_encoder_block(d_model, num_heads, dropout, use_rotary)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    model_tf = Model(inputs=input_layer, outputs=output)

    # --- Compile ---
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model_tf.compile(
        optimizer=optimizer,  # type: ignore
        loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=['accuracy']
    )

    # --- Train ---
    callbacks = [
        EarlyStopping(
            patience=early_stopping_patience,
            restore_best_weights=True,
            monitor='val_loss',
            verbose=0
        )
    ]
    model_tf.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0  # type: ignore
    )

    # --- Evaluate on windowed test ---
    test_probs = model_tf.predict(X_test, batch_size=batch_size, verbose=0).flatten() #type: ignore
    test_preds = (test_probs > 0.5).astype(int)

    # Use the window-aligned y_test for metrics (length equals len(test_preds))
    round_results = binary_metrics(data={'y_test': y_test}, preds=test_preds, probs=test_probs)

    # UEL-safe artifacts
    round_results['_preds'] = test_preds          # UEL will collect and pop this
    round_results['models'] = [model_tf]          # UEL will collect and pop this

    # Store validation arrays inside extras so they are not logged as columns
    val_probs = model_tf.predict(X_val, batch_size=batch_size, verbose=0).flatten() #type: ignore
    val_preds = (val_probs > 0.5).astype(int)
    round_results['extras'] = {
        'val_preds': val_preds,
        'val_probs': val_probs,
    }

    return round_results







# def prep(data, round_params):
#     # -------- 0) Unpack knobs --------
#     all_datetimes = data['datetime'].to_list()
#     lookback = round_params['lookback_window']
#     horizon = round_params['target_horizon']
#     price_thresh = round_params['target_threshold']
#     scaler_type = round_params['scaler_type']

#     # Ensure Polars datetime dtype
#     df = data.clone()
#     if df['datetime'].dtype != pl.Datetime:
#         df = df.with_columns(pl.col('datetime').cast(pl.Datetime))

#     all_datetimes = df['datetime'].to_list()

#     # -------- 1) Target engineering (Polars-native) --------
#     df = df.with_columns([
#         ((pl.col('close').shift(-horizon) - pl.col('close')) / pl.col('close')).alias('price_change'),
#     ])
#     df = df.with_columns([
#         (pl.when(pl.col('price_change') > price_thresh).then(1).otherwise(0)).alias('target')
#     ])

#     # Base feature columns
#     feature_cols = [
#         'open', 'high', 'low', 'close',
#         'volume', 'no_of_trades', 'maker_ratio',
#         'mean', 'std', 'median', 'iqr'
#     ]

#     # -------- 2) Optional cyclical features (Polars expr, not NumPy on Series) --------
#     if round_params['use_cyclical_features']:
#         # infer bar duration
#         kline_sec = (df['datetime'][1] - df['datetime'][0]).total_seconds()
#         exprs = []
#         names = []

#         if kline_sec < 3600:
#             exprs += [
#                 ((2 * np.pi * pl.col('datetime').dt.minute() / 60).sin()).alias('minute_sin'),
#                 ((2 * np.pi * pl.col('datetime').dt.minute() / 60).cos()).alias('minute_cos'),
#             ]
#             names += ['minute_sin', 'minute_cos']

#         exprs += [
#             ((2 * np.pi * pl.col('datetime').dt.hour() / 24).sin()).alias('hour_sin'),
#             ((2 * np.pi * pl.col('datetime').dt.hour() / 24).cos()).alias('hour_cos'),
#             ((2 * np.pi * pl.col('datetime').dt.weekday() / 7).sin()).alias('day_of_week_sin'),
#             ((2 * np.pi * pl.col('datetime').dt.weekday() / 7).cos()).alias('day_of_week_cos'),
#         ]
#         names += ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']

#         df = df.with_columns(exprs)
#         feature_cols.extend(names)

#     # -------- 3) Optional sequential features --------
#     if round_params['use_sequential_features']:
#         first_dt = df['datetime'][0]
#         df = df.with_columns([
#             ( (pl.col('datetime') - pl.lit(first_dt)).dt.total_days() ).alias('nth_day'),
#             ( ((pl.col('datetime') - pl.lit(first_dt)).dt.total_days() // 7) ).alias('nth_week'),
#             ( ((pl.col('datetime').dt.year() - first_dt.year) * 12
#                + (pl.col('datetime').dt.month() - first_dt.month)) ).alias('nth_month'),
#         ])
#         feature_cols.extend(['nth_day', 'nth_week', 'nth_month'])

#     # Remove rows made null by horizon shift
#     df = df.drop_nulls(subset=['price_change', 'target'])

#     # -------- 4) Sequential split (Loop utility) --------
#     train_df, val_df, test_df = split_sequential(df, (70, 15, 15))

#     # Keep a copy of test datetimes to build alignment for predictions
#     test_dt = test_df['datetime'].to_list()

#     # -------- 5) Build 3D sequences for Transformer --------
#     def _seq(pl_df, features, target, lb):
#         X, y = [], []
#         f_np = pl_df.select(features).to_numpy()
#         t_np = pl_df.select(target).to_numpy().ravel()
#         n = pl_df.height
#         for i in range(n - lb):
#             X.append(f_np[i:i+lb])
#             y.append(t_np[i + lb - 1])
#         return np.asarray(X), np.asarray(y)

#     X_train, y_train = _seq(train_df, feature_cols, 'target', lookback)
#     X_val,   y_val   = _seq(val_df,   feature_cols, 'target', lookback)
#     X_test,  y_test  = _seq(test_df,  feature_cols, 'target', lookback)

#     # -------- 6) Scale (fit on train only) --------
#     scaler_map = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}
#     scaler = scaler_map[scaler_type]()
#     n_features = X_train.shape[2]

#     scaler.fit(X_train.reshape(-1, n_features))
#     X_train = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
#     X_val   = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
#     X_test  = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

#     # -------- 7) Build _alignment that matches *prediction rows* --------
#     # We only predict from the (lookback-1)-th bar onward in each split window.
#     # So the test datetimes that pair with preds are:
#     test_pred_datetimes = test_dt[lookback:]  # length == len(y_test)

#     # Remaining datetimes = all train/val + ONLY the test_pred_datetimes
#     rem = (
#         train_df['datetime'].to_list()
#         + val_df['datetime'].to_list()
#         + test_pred_datetimes
#     )

#     # Missing datetimes are everything else
#     missing = sorted(set(all_datetimes) - set(rem))

#     # Polars logger anti-joins a DF built from missing; if it’s empty, its dtype
#     # can be problematic for .dt.cast_time_unit. Make it "safely non-empty"
#     # without dropping anything by adding a sentinel *outside* the test range.
#     if len(missing) == 0:
#         sentinel = test_pred_datetimes[0] - pl.duration(milliseconds=1)
#         missing = [sentinel]

#     alignment = {
#         'missing_datetimes': missing,
#         'first_test_datetime': test_pred_datetimes[0],
#         'last_test_datetime':  test_pred_datetimes[-1],
#     }

#     #    # -------- 8) Return the final data dict --------
#     out = {
#         'X_train': X_train, 'y_train': y_train,
#         'X_val':   X_val,   'y_val':   y_val,
#         'X_test':  X_test,  'y_test':  y_test,
#         '_scaler': scaler,
#         '_alignment': alignment,
#         '_feature_names': feature_cols,
#     }

#     # -------- 9) Debug checks --------
#     expected_len = len(y_test)
#     actual_len   = len(test_pred_datetimes)
#     if expected_len != actual_len:
#         print(">>> DEBUG MISMATCH in prep:")
#         print(f"y_test length: {expected_len}")
#         print(f"test_pred_datetimes length: {actual_len}")
#         print("This WILL cause downstream errors in perf_df vs price_df.")
#         raise ValueError("Mismatch between y_test and test_pred_datetimes")

#     return out

# =============================================================================
# 3. MODEL TRAINING & EVALUATION (CORRECTED VERSION)
# =============================================================================

# def _positional_encoding(seq_len, d_model):
#     """Generates sinusoidal positional encoding."""
#     pos = keras.ops.arange(0, seq_len, dtype="float32")[:, None]
#     i = keras.ops.arange(0, d_model, 2, dtype="float32")
    
#     # --- FIX 1: Replaced `1 / ...` with a type-safe Keras operation ---
#     d_model_float = keras.ops.cast(d_model, dtype="float32")
#     exponent = -((2.0 * i) / d_model_float)
#     angle_rates = keras.ops.power(10000.0, exponent)
#     angle_rads = pos * angle_rates
    
#     sines = keras.ops.sin(angle_rads)
#     cosines = keras.ops.cos(angle_rads)
    
#     pos_encoding = keras.ops.concatenate([sines, cosines], axis=-1)
#     if d_model % 2 != 0:
#         pos_encoding = keras.ops.pad(pos_encoding, [[0,0], [0,1]])
        
#     return keras.ops.expand_dims(pos_encoding, axis=0)

# def _transformer_encoder_block(inputs, d_model, num_heads, dropout_rate):
#     """A single Transformer Encoder block using Keras 3.x layers."""
#     attn_output = MultiHeadAttention(
#         num_heads=num_heads,
#         key_dim=d_model // num_heads
#     )(query=inputs, value=inputs, key=inputs)
#     attn_output = Dropout(dropout_rate)(attn_output)
#     x = LayerNormalization(epsilon=1e-6)(Add()([inputs, attn_output]))
    
#     ffn_output = Dense(units=d_model * 4, activation="relu")(x)
#     ffn_output = Dense(units=d_model)(ffn_output)
#     ffn_output = Dropout(dropout_rate)(ffn_output)
#     return LayerNormalization(epsilon=1e-6)(Add()([x, ffn_output]))
# # --- ADD THIS HELPER LAYER TO YOUR SFM FILE ---
# # --- ADD THIS HELPER LAYER TO YOUR SFM FILE ---

# class RotaryEmbedding(keras.layers.Layer):
#     """
#     Custom Rotary Positional Embedding layer.
#     """
#     def __init__(self, dim, seq_len, theta=10000.0, **kwargs):
#         super().__init__(**kwargs)
#         self.dim = dim
#         self.seq_len = seq_len
#         self.theta = theta

#         # --- FIX IS HERE ---
#         # The original code created frequencies for the half-dimension, then incorrectly
#         # concatenated them back to the full dimension. We must create the sin/cos
#         # embeddings with the half-dimension to match the split inputs in call().
        
#         # 1. Create frequencies for HALF the dimension
#         arange = keras.ops.arange(0, self.dim, 2, dtype="float32")
#         inv_freq = 1.0 / (self.theta ** (arange / self.dim))
        
#         # 2. Create the time sequence
#         t = keras.ops.arange(self.seq_len, dtype=inv_freq.dtype)
        
#         # 3. Calculate frequency matrix (shape will be [seq_len, dim/2])
#         freqs = keras.ops.einsum("i,j->ij", t, inv_freq)

#         # 4. Create sin and cos embeddings directly from the half-dim freqs
#         #    DO NOT concatenate them.
#         self.cos_emb = keras.ops.cos(freqs)
#         self.sin_emb = keras.ops.sin(freqs)

#     def call(self, inputs):
#         # Split the input into two halves along the feature dimension
#         x1, x2 = keras.ops.split(inputs, 2, axis=-1)
        
#         # Now the shapes will match:
#         # x1 shape: (batch, seq_len, 16)
#         # cos_emb shape: (seq_len, 16)
        
#         # Apply the rotation matrix properties
#         rotated_x1 = (x1 * self.cos_emb) - (x2 * self.sin_emb)
#         rotated_x2 = (x1 * self.sin_emb) + (x2 * self.cos_emb)
        
#         # Concatenate the rotated halves back together
#         return keras.ops.concatenate([rotated_x1, rotated_x2], axis=-1)

#     def compute_output_shape(self, input_shape):
#         return input_shape

# def _build_transformer_model(input_shape, round_params):
#     """Builds the complete Transformer classifier using Keras 3.x."""
#     d_model = round_params['d_model']
#     seq_len = input_shape[0] # This is the lookback_window
#     encoding_type = round_params['positional_encoding_type']

#     inputs = Input(shape=input_shape)
#     x = Dense(units=d_model, activation="relu")(inputs)
    
#     if encoding_type == 'sinusoidal':
#         x += _positional_encoding(seq_len, d_model)
#     elif encoding_type == 'rotary':
#         # Pass the sequence length to the layer during initialization
#         x = RotaryEmbedding(dim=d_model, seq_len=seq_len)(x)
    
#     for _ in range(round_params['num_encoder_layers']):
#         x = _transformer_encoder_block(
#             x, d_model, round_params['num_heads'], round_params['dropout_rate']
#         )
        
#     x = GlobalAveragePooling1D()(x)
#     x = Dropout(0.2)(x)
#     x = Dense(units=d_model // 2, activation="relu")(x)
#     outputs = Dense(units=1, activation="sigmoid")(x)
    
#     return Model(inputs=inputs, outputs=outputs)

# def model(data: dict, round_params):
#     """
#     Builds, trains, and evaluates the model with all corrections applied.
#     """
#     # ------------------ 1. SFM Rule Adherence: Validity Check ------------------
#     d_model = round_params['d_model']
#     num_heads = round_params['num_heads']
#     encoding_type = round_params['positional_encoding_type']

#     if d_model % num_heads != 0:
#         return {
#             'recall': None, 'precision': None, 'fpr': None, 'auc': None,
#             'accuracy': None, '_preds': np.array([]),
#             'extras': {'status': f'Skipped invalid arch: d_model={d_model}, num_heads={num_heads}'}
#         }
#     # Check 1: d_model must be divisible by num_heads for MultiHeadAttention
#     if d_model % num_heads != 0:
#         return {
#             'recall': None, 'precision': None, 'fpr': None, 'auc': None,
#             'accuracy': None, '_preds': np.array([]),
#             'extras': {'status': f'Skipped invalid arch: d_model={d_model}, num_heads={num_heads}'}
#         }
        
#     # Check 2: d_model must be even for RotaryEmbedding
#     if encoding_type == 'rotary' and d_model % 2 != 0:
#         return {
#             'recall': None, 'precision': None, 'fpr': None, 'auc': None,
#             'accuracy': None, '_preds': np.array([]),
#             'extras': {'status': f'Skipped invalid arch: RoPE requires even d_model, got {d_model}'}
#         }
    

#     # ------------------ 2. Unpack Data and Params ------------------
#     X_train, y_train = data['X_train'], data['y_train']
#     X_val, y_val = data['X_val'], data['y_val']
#     X_test, y_test = data['X_test'], data['y_test']

#     # ------------------ 3. Build and Compile the Model ------------------
#     input_shape = (X_train.shape[1], X_train.shape[2])
#     transformer_model = _build_transformer_model(input_shape, round_params)
    
#     optimizer = AdamW(
#         learning_rate=round_params['learning_rate'],
#         weight_decay=round_params['weight_decay']
#     )
    
#     transformer_model.compile(
#         optimizer=optimizer, # --- FIX 2: Ignore incorrect linter error --- # type: ignore
#         loss='binary_crossentropy',
#         metrics=['accuracy', keras.metrics.AUC(name='auc')]
#     )

#     # ------------------ 4. Train the Model ------------------
#     early_stopping = EarlyStopping(
#         monitor='val_loss',
#         patience=round_params['early_stopping_patience'],
#         restore_best_weights=True,
#         verbose=0
#     )
    
#     history = transformer_model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         batch_size=round_params['batch_size'],
#         epochs=150,
#         callbacks=[early_stopping],
#         verbose=0 #type: ignore
#     )

#     # ------------------ 5. Evaluate and Return Results (Corrected Signature) ------------------
#     test_preds_proba = transformer_model.predict(X_test, verbose=0).flatten() #type: ignore
    
#     # --- FIX 3: Prepare arguments exactly as `binary_metrics` expects ---
#     # a. Create binary predictions from probabilities
#     binary_preds = (test_preds_proba > 0.5).astype(int)
    
#     # b. Prepare the data dictionary argument
#     data_for_metrics = {'y_test': y_test}
    
#     # c. Call the function with the correct arguments
#     round_results = binary_metrics(
#         data=data_for_metrics,
#         preds=binary_preds.tolist(),
#         probs=test_preds_proba.tolist()
#     )

#     # d. Manually add extras to the results dictionary
#     round_results['extras'] = {
#         'val_loss': min(history.history['val_loss']),
#         'stopped_epoch': early_stopping.stopped_epoch
#     }
    
#     # Add artifacts for UEL collection
#     round_results['_preds'] = test_preds_proba
#     # round_results['models'] = transformer_model # Optional
    
#     return round_results