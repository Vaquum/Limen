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
import keras
from keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add, Embedding
)
from keras.models import Model
from keras.optimizers import Adam, AdamW
from keras.callbacks import EarlyStopping
import polars as pl
import numpy as np
from loop.manifest import Manifest

def add_cyclical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add cyclical features for hour, minute, and day to df"""
    dt = df['datetime'].to_pandas()
    hour = dt.dt.hour
    minute = dt.dt.minute
    day = dt.dt.day
    df = df.with_columns([
        pl.Series('sin_hour', np.sin(2 * np.pi * hour / 24)),
        pl.Series('cos_hour', np.cos(2 * np.pi * hour / 24)),
        pl.Series('sin_minute', np.sin(2 * np.pi * minute / 60)),
        pl.Series('cos_minute', np.cos(2 * np.pi * minute / 60)),
        pl.Series('sin_day', np.sin(2 * np.pi * day / 31)),
        pl.Series('cos_day', np.cos(2 * np.pi * day / 31))
    ])
    return df

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

def manifest():
    # No-op bar formation for normal klines
    def base_bar_formation(data: pl.DataFrame, **params) -> pl.DataFrame:
        return data

    required_cols = [
        'datetime', 'open', 'high', 'low', 'close', 'mean', 'std', 'median', 'iqr',
        'volume', 'maker_ratio', 'no_of_trades', 'open_liquidity', 'high_liquidity',
        'low_liquidity', 'close_liquidity', 'liquidity_sum', 'maker_volume', 'maker_liquidity'
    ]

    from sklearn.preprocessing import StandardScaler

    return (
        Manifest()
        .set_split_config(8, 1, 2)
        .set_bar_formation(base_bar_formation, bar_type='base')
        .set_required_bar_columns(required_cols)

        # Add cyclical features (all granular time encodings)
        .add_feature(add_cyclical_features)

        # Target regime: quantile flag on windowed forward return
        .with_target('target_regime')
            .add_fitted_transform(regime_target)
                .fit_param('_regime_cutoff', lambda df, prediction_window, target_quantile: 
                    np.quantile(
                        (df['close'].shift(-prediction_window) - df['close']) / df['close'], 
                        target_quantile
                    ), 
                    prediction_window='prediction_window', target_quantile='target_quantile')
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


# Helper function remains the same and is correct
def _create_sequences(data_df, feature_cols, target_col, lookback):
    """
    Helper to transform a 2D time-series DataFrame into 3D sequences.
    """
    X, y = [], []
    features_np = data_df[feature_cols].to_numpy()
    target_np = data_df[target_col].to_numpy()
    
    for i in range(len(data_df) - lookback ):
        X.append(features_np[i:(i + lookback)])
        y.append(target_np[i + lookback - 1]) # Target corresponds to the last step of the window
    
    return np.array(X), np.array(y)



def prep(data, round_params):
    # -------- 0) Unpack knobs --------
    all_datetimes = data['datetime'].to_list()
    lookback = round_params['lookback_window']
    horizon = round_params['target_horizon']
    price_thresh = round_params['target_threshold']
    scaler_type = round_params['scaler_type']

    # Ensure Polars datetime dtype
    df = data.clone()
    if df['datetime'].dtype != pl.Datetime:
        df = df.with_columns(pl.col('datetime').cast(pl.Datetime))

    all_datetimes = df['datetime'].to_list()

    # -------- 1) Target engineering (Polars-native) --------
    df = df.with_columns([
        ((pl.col('close').shift(-horizon) - pl.col('close')) / pl.col('close')).alias('price_change'),
    ])
    df = df.with_columns([
        (pl.when(pl.col('price_change') > price_thresh).then(1).otherwise(0)).alias('target')
    ])

    # Base feature columns
    feature_cols = [
        'open', 'high', 'low', 'close',
        'volume', 'no_of_trades', 'maker_ratio',
        'mean', 'std', 'median', 'iqr'
    ]

    # -------- 2) Optional cyclical features (Polars expr, not NumPy on Series) --------
    if round_params['use_cyclical_features']:
        # infer bar duration
        kline_sec = (df['datetime'][1] - df['datetime'][0]).total_seconds()
        exprs = []
        names = []

        if kline_sec < 3600:
            exprs += [
                ((2 * np.pi * pl.col('datetime').dt.minute() / 60).sin()).alias('minute_sin'),
                ((2 * np.pi * pl.col('datetime').dt.minute() / 60).cos()).alias('minute_cos'),
            ]
            names += ['minute_sin', 'minute_cos']

        exprs += [
            ((2 * np.pi * pl.col('datetime').dt.hour() / 24).sin()).alias('hour_sin'),
            ((2 * np.pi * pl.col('datetime').dt.hour() / 24).cos()).alias('hour_cos'),
            ((2 * np.pi * pl.col('datetime').dt.weekday() / 7).sin()).alias('day_of_week_sin'),
            ((2 * np.pi * pl.col('datetime').dt.weekday() / 7).cos()).alias('day_of_week_cos'),
        ]
        names += ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']

        df = df.with_columns(exprs)
        feature_cols.extend(names)

    # -------- 3) Optional sequential features --------
    if round_params['use_sequential_features']:
        first_dt = df['datetime'][0]
        df = df.with_columns([
            ( (pl.col('datetime') - pl.lit(first_dt)).dt.total_days() ).alias('nth_day'),
            ( ((pl.col('datetime') - pl.lit(first_dt)).dt.total_days() // 7) ).alias('nth_week'),
            ( ((pl.col('datetime').dt.year() - first_dt.year) * 12
               + (pl.col('datetime').dt.month() - first_dt.month)) ).alias('nth_month'),
        ])
        feature_cols.extend(['nth_day', 'nth_week', 'nth_month'])

    # Remove rows made null by horizon shift
    df = df.drop_nulls(subset=['price_change', 'target'])

    # -------- 4) Sequential split (Loop utility) --------
    train_df, val_df, test_df = split_sequential(df, (70, 15, 15))

    # Keep a copy of test datetimes to build alignment for predictions
    test_dt = test_df['datetime'].to_list()

    # -------- 5) Build 3D sequences for Transformer --------
    def _seq(pl_df, features, target, lb):
        X, y = [], []
        f_np = pl_df.select(features).to_numpy()
        t_np = pl_df.select(target).to_numpy().ravel()
        n = pl_df.height
        for i in range(n - lb):
            X.append(f_np[i:i+lb])
            y.append(t_np[i + lb - 1])
        return np.asarray(X), np.asarray(y)

    X_train, y_train = _seq(train_df, feature_cols, 'target', lookback)
    X_val,   y_val   = _seq(val_df,   feature_cols, 'target', lookback)
    X_test,  y_test  = _seq(test_df,  feature_cols, 'target', lookback)

    # -------- 6) Scale (fit on train only) --------
    scaler_map = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}
    scaler = scaler_map[scaler_type]()
    n_features = X_train.shape[2]

    scaler.fit(X_train.reshape(-1, n_features))
    X_train = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    # -------- 7) Build _alignment that matches *prediction rows* --------
    # We only predict from the (lookback-1)-th bar onward in each split window.
    # So the test datetimes that pair with preds are:
    test_pred_datetimes = test_dt[lookback:]  # length == len(y_test)

    # Remaining datetimes = all train/val + ONLY the test_pred_datetimes
    rem = (
        train_df['datetime'].to_list()
        + val_df['datetime'].to_list()
        + test_pred_datetimes
    )

    # Missing datetimes are everything else
    missing = sorted(set(all_datetimes) - set(rem))

    # Polars logger anti-joins a DF built from missing; if it’s empty, its dtype
    # can be problematic for .dt.cast_time_unit. Make it "safely non-empty"
    # without dropping anything by adding a sentinel *outside* the test range.
    if len(missing) == 0:
        sentinel = test_pred_datetimes[0] - pl.duration(milliseconds=1)
        missing = [sentinel]

    alignment = {
        'missing_datetimes': missing,
        'first_test_datetime': test_pred_datetimes[0],
        'last_test_datetime':  test_pred_datetimes[-1],
    }

    #    # -------- 8) Return the final data dict --------
    out = {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
        '_scaler': scaler,
        '_alignment': alignment,
        '_feature_names': feature_cols,
    }

    # -------- 9) Debug checks --------
    expected_len = len(y_test)
    actual_len   = len(test_pred_datetimes)
    if expected_len != actual_len:
        print(">>> DEBUG MISMATCH in prep:")
        print(f"y_test length: {expected_len}")
        print(f"test_pred_datetimes length: {actual_len}")
        print("This WILL cause downstream errors in perf_df vs price_df.")
        raise ValueError("Mismatch between y_test and test_pred_datetimes")

    return out



# def prep(data, round_params):
#     """
#     A deterministic data preparation pipeline (Final Version).
#     """
#     # ------------------ 1. Fulfill SFM Contract & Unpack Params ------------------
#     all_datetimes = data['datetime'].to_list() # Keep datetimes as a Polars DF

#     # Unpack knobs
#     lookback = round_params['lookback_window']
#     horizon = round_params['target_horizon']
#     price_thresh = round_params['target_threshold']
#     scaler_type = round_params['scaler_type']
    
#     processed_df = data.clone()



#     # ------------------ 2. Target Engineering ------------------
#     future_close = processed_df['close'].shift(-horizon)
#     price_change = (future_close - processed_df['close']) / processed_df['close']
#     processed_df = processed_df.with_columns([
#         price_change.alias('price_change'),
#         (pl.when(price_change > price_thresh).then(1).otherwise(0)).alias('target')
#     ])

#     feature_cols = [
#         'open', 'high', 'low', 'close',
#         'volume', 'no_of_trades', 'maker_ratio',
#         'mean', 'std', 'median', 'iqr'
#     ]

#     # Cyclical features
#     if round_params['use_cyclical_features']:
#         cyc_feature_exprs = []
#         cyc_feature_names = []
#         kline_duration = (processed_df['datetime'][1] - processed_df['datetime'][0]).total_seconds()

#         if kline_duration < 3600:
#             cyc_feature_exprs.extend([
#                 (np.sin(2 * np.pi * processed_df['datetime'].dt.minute() / 60)).alias('minute_sin'),
#                 (np.cos(2 * np.pi * processed_df['datetime'].dt.minute() / 60)).alias('minute_cos')
#             ])
#             cyc_feature_names.extend(['minute_sin', 'minute_cos'])

#         cyc_feature_exprs.extend([
#             (np.sin(2 * np.pi * processed_df['datetime'].dt.hour() / 24)).alias('hour_sin'),
#             (np.cos(2 * np.pi * processed_df['datetime'].dt.hour() / 24)).alias('hour_cos'),
#             (np.sin(2 * np.pi * processed_df['datetime'].dt.weekday() / 7)).alias('day_of_week_sin'),
#             (np.cos(2 * np.pi * processed_df['datetime'].dt.weekday() / 7)).alias('day_of_week_cos')
#         ])
#         cyc_feature_names.extend(['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'])

#         processed_df = processed_df.with_columns(cyc_feature_exprs)
#         feature_cols.extend(cyc_feature_names)

#     # Sequential features
#     if round_params['use_sequential_features']:
#         first_datetime = processed_df['datetime'][0]
#         processed_df = processed_df.with_columns([
#             (processed_df['datetime'] - first_datetime).dt.total_days().alias('nth_day'),
#             ((processed_df['datetime'] - first_datetime).dt.total_days() // 7).alias('nth_week'),
#             ((processed_df['datetime'].dt.year() - first_datetime.year) * 12 + \
#              (processed_df['datetime'].dt.month() - first_datetime.month)).alias('nth_month')
#         ])
#         feature_cols.extend(['nth_day', 'nth_week', 'nth_month'])

#     processed_df = processed_df.drop_nulls()
#         # ------------------ 3.  Chronological Split ------------------
#     split_data_list = split_sequential(processed_df, (70, 15, 15))
#     train_df, val_df, test_df = split_data_list[0], split_data_list[1], split_data_list[2]
#     # ------------------ 4. Chronological Split ------------------
#     cols_for_utility = feature_cols + ['datetime', 'target']
    
#     # This call produces a base dictionary with 2D data and the crucial _alignment block
#     base_data_dict = split_data_to_prep_output(
#         split_data=split_data_list,
#         cols=cols_for_utility,
#         all_datetimes=all_datetimes
#     )

#     # ------------------ 4. Manual 3D Sequencing (Required for Transformer) ------------------
    
#     # Now, we create the 3D sequences our Transformer needs using our helper.
#     X_train, y_train = _create_sequences(train_df, feature_cols, 'target', lookback)
#     X_val, y_val = _create_sequences(val_df, feature_cols, 'target', lookback)
#     X_test, y_test = _create_sequences(test_df, feature_cols, 'target', lookback)

#     # ------------------ 5. Scaling ------------------
    
#     scaler_map = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}
#     scaler = scaler_map[scaler_type]()
#     n_features = X_train.shape[2]

#     # Reshape, fit on train only, and transform all sets
#     scaler.fit(X_train.reshape(-1, n_features))
#     X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
#     X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
#     X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

#     # ------------------ 6. Construct Final Data Dictionary ------------------

# # Drop first lookback rows from alignment to match sequence lengths
#     import pandas as pd
#     import datetime
#     def trim_alignment(alignment_block, lookback):
#         new_alignment = {}
#         for split_name, df in alignment_block.items():
#             if df is None:
#                 new_alignment[split_name] = None
#                 continue
#             if isinstance(df, pd.Series):
#                 df = df.to_frame(name="datetime")  # enforce column
#             elif isinstance(df, datetime.datetime):
#                 df = pd.DataFrame({"datetime": [df]})
#             elif not isinstance(df, pd.DataFrame):
#                 raise TypeError(f"Unexpected type for alignment[{split_name}]: {type(df)}")

#             # Enforce datetime dtype
#             if "datetime" in df.columns:
#                 df["datetime"] = pd.to_datetime(df["datetime"])

#             new_alignment[split_name] = df.iloc[lookback:].reset_index(drop=True)
#         return new_alignment

#     final_data_dict = {
#         'X_train': X_train_scaled, 'y_train': y_train,
#         'X_val': X_val_scaled, 'y_val': y_val,
#         'X_test': X_test_scaled, 'y_test': y_test,
#         '_scaler': scaler,
#         '_alignment': trim_alignment(base_data_dict['_alignment'], lookback)
#     }
#     print(">>> DEBUG: Alignment after trimming")
#     for k, v in final_data_dict["_alignment"].items():
#         if v is not None and "datetime" in v.columns:
#             assert pd.api.types.is_datetime64_any_dtype(v["datetime"]), f"{k} datetime wrong type"

#     return final_data_dict

# =============================================================================
# 3. MODEL TRAINING & EVALUATION (CORRECTED VERSION)
# =============================================================================

def _positional_encoding(seq_len, d_model):
    """Generates sinusoidal positional encoding."""
    pos = keras.ops.arange(0, seq_len, dtype="float32")[:, None]
    i = keras.ops.arange(0, d_model, 2, dtype="float32")
    
    # --- FIX 1: Replaced `1 / ...` with a type-safe Keras operation ---
    d_model_float = keras.ops.cast(d_model, dtype="float32")
    exponent = -((2.0 * i) / d_model_float)
    angle_rates = keras.ops.power(10000.0, exponent)
    angle_rads = pos * angle_rates
    
    sines = keras.ops.sin(angle_rads)
    cosines = keras.ops.cos(angle_rads)
    
    pos_encoding = keras.ops.concatenate([sines, cosines], axis=-1)
    if d_model % 2 != 0:
        pos_encoding = keras.ops.pad(pos_encoding, [[0,0], [0,1]])
        
    return keras.ops.expand_dims(pos_encoding, axis=0)

def _transformer_encoder_block(inputs, d_model, num_heads, dropout_rate):
    """A single Transformer Encoder block using Keras 3.x layers."""
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(query=inputs, value=inputs, key=inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = LayerNormalization(epsilon=1e-6)(Add()([inputs, attn_output]))
    
    ffn_output = Dense(units=d_model * 4, activation="relu")(x)
    ffn_output = Dense(units=d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(Add()([x, ffn_output]))
# --- ADD THIS HELPER LAYER TO YOUR SFM FILE ---
# --- ADD THIS HELPER LAYER TO YOUR SFM FILE ---

class RotaryEmbedding(keras.layers.Layer):
    """
    Custom Rotary Positional Embedding layer.
    """
    def __init__(self, dim, seq_len, theta=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.seq_len = seq_len
        self.theta = theta

        # --- FIX IS HERE ---
        # The original code created frequencies for the half-dimension, then incorrectly
        # concatenated them back to the full dimension. We must create the sin/cos
        # embeddings with the half-dimension to match the split inputs in call().
        
        # 1. Create frequencies for HALF the dimension
        arange = keras.ops.arange(0, self.dim, 2, dtype="float32")
        inv_freq = 1.0 / (self.theta ** (arange / self.dim))
        
        # 2. Create the time sequence
        t = keras.ops.arange(self.seq_len, dtype=inv_freq.dtype)
        
        # 3. Calculate frequency matrix (shape will be [seq_len, dim/2])
        freqs = keras.ops.einsum("i,j->ij", t, inv_freq)

        # 4. Create sin and cos embeddings directly from the half-dim freqs
        #    DO NOT concatenate them.
        self.cos_emb = keras.ops.cos(freqs)
        self.sin_emb = keras.ops.sin(freqs)

    def call(self, inputs):
        # Split the input into two halves along the feature dimension
        x1, x2 = keras.ops.split(inputs, 2, axis=-1)
        
        # Now the shapes will match:
        # x1 shape: (batch, seq_len, 16)
        # cos_emb shape: (seq_len, 16)
        
        # Apply the rotation matrix properties
        rotated_x1 = (x1 * self.cos_emb) - (x2 * self.sin_emb)
        rotated_x2 = (x1 * self.sin_emb) + (x2 * self.cos_emb)
        
        # Concatenate the rotated halves back together
        return keras.ops.concatenate([rotated_x1, rotated_x2], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

def _build_transformer_model(input_shape, round_params):
    """Builds the complete Transformer classifier using Keras 3.x."""
    d_model = round_params['d_model']
    seq_len = input_shape[0] # This is the lookback_window
    encoding_type = round_params['positional_encoding_type']

    inputs = Input(shape=input_shape)
    x = Dense(units=d_model, activation="relu")(inputs)
    
    if encoding_type == 'sinusoidal':
        x += _positional_encoding(seq_len, d_model)
    elif encoding_type == 'rotary':
        # Pass the sequence length to the layer during initialization
        x = RotaryEmbedding(dim=d_model, seq_len=seq_len)(x)
    
    for _ in range(round_params['num_encoder_layers']):
        x = _transformer_encoder_block(
            x, d_model, round_params['num_heads'], round_params['dropout_rate']
        )
        
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(units=d_model // 2, activation="relu")(x)
    outputs = Dense(units=1, activation="sigmoid")(x)
    
    return Model(inputs=inputs, outputs=outputs)

def model(data: dict, round_params):
    """
    Builds, trains, and evaluates the model with all corrections applied.
    """
    # ------------------ 1. SFM Rule Adherence: Validity Check ------------------
    d_model = round_params['d_model']
    num_heads = round_params['num_heads']
    encoding_type = round_params['positional_encoding_type']

    if d_model % num_heads != 0:
        return {
            'recall': None, 'precision': None, 'fpr': None, 'auc': None,
            'accuracy': None, '_preds': np.array([]),
            'extras': {'status': f'Skipped invalid arch: d_model={d_model}, num_heads={num_heads}'}
        }
    # Check 1: d_model must be divisible by num_heads for MultiHeadAttention
    if d_model % num_heads != 0:
        return {
            'recall': None, 'precision': None, 'fpr': None, 'auc': None,
            'accuracy': None, '_preds': np.array([]),
            'extras': {'status': f'Skipped invalid arch: d_model={d_model}, num_heads={num_heads}'}
        }
        
    # Check 2: d_model must be even for RotaryEmbedding
    if encoding_type == 'rotary' and d_model % 2 != 0:
        return {
            'recall': None, 'precision': None, 'fpr': None, 'auc': None,
            'accuracy': None, '_preds': np.array([]),
            'extras': {'status': f'Skipped invalid arch: RoPE requires even d_model, got {d_model}'}
        }
    

    # ------------------ 2. Unpack Data and Params ------------------
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    # ------------------ 3. Build and Compile the Model ------------------
    input_shape = (X_train.shape[1], X_train.shape[2])
    transformer_model = _build_transformer_model(input_shape, round_params)
    
    optimizer = AdamW(
        learning_rate=round_params['learning_rate'],
        weight_decay=round_params['weight_decay']
    )
    
    transformer_model.compile(
        optimizer=optimizer, # --- FIX 2: Ignore incorrect linter error --- # type: ignore
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    # ------------------ 4. Train the Model ------------------
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=round_params['early_stopping_patience'],
        restore_best_weights=True,
        verbose=0
    )
    
    history = transformer_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=round_params['batch_size'],
        epochs=150,
        callbacks=[early_stopping],
        verbose=0 #type: ignore
    )

    # ------------------ 5. Evaluate and Return Results (Corrected Signature) ------------------
    test_preds_proba = transformer_model.predict(X_test, verbose=0).flatten() #type: ignore
    
    # --- FIX 3: Prepare arguments exactly as `binary_metrics` expects ---
    # a. Create binary predictions from probabilities
    binary_preds = (test_preds_proba > 0.5).astype(int)
    
    # b. Prepare the data dictionary argument
    data_for_metrics = {'y_test': y_test}
    
    # c. Call the function with the correct arguments
    round_results = binary_metrics(
        data=data_for_metrics,
        preds=binary_preds.tolist(),
        probs=test_preds_proba.tolist()
    )

    # d. Manually add extras to the results dictionary
    round_results['extras'] = {
        'val_loss': min(history.history['val_loss']),
        'stopped_epoch': early_stopping.stopped_epoch
    }
    
    # Add artifacts for UEL collection
    round_results['_preds'] = test_preds_proba
    # round_results['models'] = transformer_model # Optional
    
    return round_results