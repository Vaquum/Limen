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


def params():
    """
    Defines the parameter space for the Transformer binary classifier experiment.
    Every parameter is a "knob" we can turn.
    """
    return {
        # ---------------- Prep Knobs ----------------
        'lookback_window': [24, 72],
        'target_horizon': [6, 24],
        'target_threshold': [0.002, 0.005], # % price change for a "bullish" event
        'scaler_type': ['standard', 'minmax', 'robust'], # Use strings for serialization
        
        # ---------------- Feature Selection Knobs ----------------
        'use_cyclical_features': [True, False],
        'use_sequential_features': [True, False],
        
        # ---------------- Transformer Architecture Knobs ----------------
        # Coupled tuples: (name, d_model, num_heads) to ensure d_model % num_heads == 0
        'd_model': [32, 64], # The embedding dimension
        'num_heads': [2, 4, 8],

        'num_encoder_layers': [2, 4],
        'dropout_rate': [0.1, 0.2],
        'positional_encoding_type': ['sinusoidal', 'rotary'],
        
        # ---------------- Training & Optimization Knobs ----------------
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'batch_size': [32, 64, 128],
        'weight_decay': [0.0, 0.01, 0.1], 
        'early_stopping_patience': [5, 10], 
        # 'traininng_split_ratio': []
    }

# Helper function remains the same and is correct
def _create_sequences(data_df, feature_cols, target_col, lookback):
    """
    Helper to transform a 2D time-series DataFrame into 3D sequences.
    """
    X, y = [], []
    features_np = data_df[feature_cols].to_numpy()
    target_np = data_df[target_col].to_numpy()
    
    for i in range(len(data_df) - lookback):
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

# =============================================================================
# 3. MODEL TRAINING & EVALUATION (CORRECTED VERSION)
# =============================================================================

def _positional_encoding(seq_len, d_model):
    """Generates sinusoidal positional encoding."""
    pos = keras.ops.arange(0, seq_len, dtype="float32")[:, None]
    i = keras.ops.arange(0, d_model, 2, dtype="float32")
    
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
    Rotary Positional Embedding (RoPE) layer, implemented in pure Keras 3.0.

    This layer rotates input embeddings based on their relative position, offering
    a modern alternative to sinusoidal positional encoding.
    
    Args:
        dim (int): The dimension of the feature space, which must be even.
    """
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        if dim % 2 != 0:
            raise ValueError(f"The feature dimension must be even for RotaryEmbedding, but got {dim}")
        self.dim = dim
        
        # Calculate inverse frequencies for the sinusoidal basis
        inv_freq = 1.0 / (10000 ** (keras.ops.arange(0, dim, 2, dtype="float32") / dim))
        self.inv_freq = inv_freq

    def call(self, inputs):
        """Applies rotary embedding to the input tensor."""
        # inputs shape: (batch_size, sequence_length, feature_dimension)
        seq_len = keras.ops.shape(inputs)[1]
        t = keras.ops.arange(seq_len, dtype="float32")
        
        # Calculate frequency components
        freqs = keras.ops.einsum("i,j->ij", t, self.inv_freq)
        # freqs shape: (sequence_length, dim / 2)
        
        # Duplicate for both sine and cosine components
        emb = keras.ops.concatenate([freqs, freqs], axis=-1)
        # emb shape: (sequence_length, dim)

        # Create cosine and sine embeddings for rotation
        cos_emb = keras.ops.cos(emb)
        sin_emb = keras.ops.sin(emb)

        # Split the input into two halves for rotation
        x1, x2 = keras.ops.split(inputs, 2, axis=-1)
        
        # Apply the rotation matrix properties
        # rotated_x = x * cos - flip(x) * sin
        rotated_x1 = (x1 * cos_emb) - (x2 * sin_emb)
        rotated_x2 = (x1 * sin_emb) + (x2 * cos_emb)
        
        # Concatenate the rotated halves back together
        return keras.ops.concatenate([rotated_x1, rotated_x2], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

def _build_transformer_model(input_shape, round_params):
    """Builds the complete Transformer classifier using pure Keras 3.0."""
    d_model = round_params['d_model']
    num_heads = round_params['num_heads']
    
    inputs = Input(shape=input_shape)
    
    # Project input features into the Transformer's dimensional space (d_model)
    x = Dense(units=d_model, activation="relu")(inputs)
    
    # --- MODIFIED SECTION: Apply Positional Encoding ---
    encoding_type = round_params['positional_encoding_type']
    
    if encoding_type == 'sinusoidal':
        # Additive sinusoidal encoding
        x += _positional_encoding(input_shape[0], d_model)
    elif encoding_type == 'rotary':
        # Rotational encoding
        x = RotaryEmbedding(dim=d_model)(x)
    
    for _ in range(round_params['num_encoder_layers']):
        x = _transformer_encoder_block(
            x, d_model, round_params['num_heads'], round_params['dropout_rate']
        )
        
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(units=d_model // 2, activation="relu")(x)
    outputs = Dense(units=1, activation="sigmoid")(x)
    
    return Model(inputs=inputs, outputs=outputs)

def model(data_dict, round_params):
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
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']

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