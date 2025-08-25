'''
Transformer Binary Classifier - UEL Single File Model format
Bitcoin long-only trading strategy with transformer-based sequence modeling
'''

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.metrics.binary_metrics import binary_metrics


def params():
    """
    Focused parameter space for transformer-based binary classifier
    Optimized for high-throughput sweeps with practical ranges
    """
    p = {
        # === DATA & PREP ===
        "feature_set": ["ohlcv", "ohlcv+indicators"],
        "sampling_interval": [1, 5],  # minutes
        "lookback_window": [32, 64, 128],  # sequence length (candles, not hours)
        "prediction_horizon": [1, 2, 4],  # hours ahead
        "min_return_threshold": [1.0, 1.5, 2.0],  # % move for long class
        "normalization": ["standard", "robust"],

        # === TRANSFORMER ARCHITECTURE ===
        "d_model": [64, 128],
        "n_heads": [2, 4],
        "n_layers": [2, 3],
        "dropout": [0.1, 0.3],

        # === TRAINING ===
        "learning_rate": [1e-4, 5e-4],
        "batch_size": [32, 64],
        "weight_decay": [0, 1e-5],

        # === DECISION / STRATEGY ===
        "decision_threshold": [0.5, 0.6, 0.7],
        "confidence_margin": [0.05, 0.1],  # margin between long vs no-trade
        "trade_horizon": [30, 60],  # minutes holding period

        # === REGULARIZATION (light) ===
        "label_smoothing": [0.0, 0.1],
    }
    return p
        

def prep(data: pl.DataFrame, round_params: dict):
    """
    Prepares OHLCV + optional temporal/indicator features
    for a Transformer-based binary classifier.
    
    Args:
        data (pl.DataFrame): Input dataframe with at least ['datetime','open','high','low','close','volume'].
        round_params (dict): Parameters dict from params().
    
    Returns:
        dict: Preprocessed data in SFM-compatible format:
              x_train, y_train, x_val, y_val, x_test, y_test,
              plus feature names and scalers.
    """

    # Ensure datetime is parsed correctly
    if data.schema["datetime"] != pl.Datetime:
        try:
            data = data.with_columns(pl.col("datetime").str.to_datetime())
        except:
            data = data.with_columns(pl.from_epoch(pl.col("datetime"), time_unit="s").alias("datetime"))

    # --- Core Params ---
    lookback = round_params.get("sequence_length", 32)  # window size for transformer input
    horizon = round_params.get("prediction_horizon", 1) # how far ahead to predict
    ret_thresh = round_params.get("min_return_threshold", 2.0) / 100.0 # % return threshold for binary labeling
    norm_method = round_params.get("normalization", "standard") # scaler choice

    # --- Temporal Features ---
    time_feat = round_params.get("time_features", "intraday+weekly")
    if "intraday" in time_feat:
        data = data.with_columns([
            (np.sin(2 * np.pi * pl.col("datetime").dt.hour() / 24)).alias("sin_hour"),
            (np.cos(2 * np.pi * pl.col("datetime").dt.hour() / 24)).alias("cos_hour")
        ])
    if "weekly" in time_feat:
        data = data.with_columns([
            (np.sin(2 * np.pi * pl.col("datetime").dt.weekday() / 7)).alias("sin_weekday"),
            (np.cos(2 * np.pi * pl.col("datetime").dt.weekday() / 7)).alias("cos_weekday")
        ])
    if "flags" in time_feat:  # optional month start/end flags
        data = data.with_columns([
            (pl.col("datetime").dt.day() <= 2).alias("is_month_start"),
            (pl.col("datetime").dt.day() >= 28).alias("is_month_end"),
            (pl.col("datetime").dt.weekday() >= 5).alias("is_weekend")
        ])

    # --- Price Returns Feature (always include) ---
    data = data.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("return_1"),
        (pl.col("close") / pl.col("close").shift(5) - 1).alias("return_5")
    ])

    # --- Label Creation (Binary) ---
    # Label = 1 if forward return over horizon >= threshold, else 0
    data = data.with_columns([
        (pl.col("close").shift(-horizon) / pl.col("close") - 1).alias("future_return")
    ])
    data = data.with_columns([
        (pl.col("future_return") >= ret_thresh).cast(pl.Int8).alias("label")
    ])

    # Drop NaNs (from shifting operations)
    data = data.drop_nulls()

    # --- Feature Selection ---
    exclude_cols = ["datetime", "label", "future_return"]
    feature_cols = [c for c in data.columns if c not in exclude_cols]

    # --- Normalization ---
    if norm_method == "standard":
        scaler = StandardScaler()
    elif norm_method == "robust":
        scaler = RobustScaler()
    elif norm_method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization: {norm_method}")

    X = scaler.fit_transform(data[feature_cols].to_numpy())
    y = data["label"].to_numpy()

    # --- Sequence Building (Transformer expects [seq_len, features]) ---
    sequences = []
    targets = []
    for i in range(len(X) - lookback - horizon):
        seq_x = X[i:i+lookback]               # input window
        seq_y = y[i+lookback + horizon - 1]   # label aligned with prediction horizon
        sequences.append(seq_x)
        targets.append(seq_y)

    X = np.array(sequences)
    y = np.array(targets)

    # --- Train/Val/Test Split (simple sequential split for now) ---
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    data_dict = {
        "x_train": X[:train_end],
        "y_train": y[:train_end],
        "x_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "x_test": X[val_end:],
        "y_test": y[val_end:],
        "_feature_names": feature_cols,
        "_scaler": scaler,
    }

    return data_dict

def model(data, round_params):
    """
    Train and evaluate a bare-bones Transformer-based binary classifier.
    Compatible with UEL expectations (returns dict with models + extras).
    
    Args:
        data (dict): Output from prep(), containing train/val/test splits
        round_params (dict): Parameters for current permutation
    
    Returns:
        dict: UEL-compatible results with trained model and metrics
    """

    # --- Extract Data ---
    X_train = torch.tensor(data["x_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_val = torch.tensor(data["x_val"], dtype=torch.float32)
    y_val = torch.tensor(data["y_val"], dtype=torch.long)
    X_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    # --- Build Transformer Model ---
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model, n_heads, n_layers, dropout, seq_len, n_features):
            super().__init__()
            self.embedding = nn.Linear(n_features, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.fc = nn.Linear(d_model * seq_len, 2)  # binary classifier

        def forward(self, x):
            x = self.embedding(x)              # [B, seq_len, d_model]
            x = self.encoder(x)                # [B, seq_len, d_model]
            x = x.reshape(x.size(0), -1)       # flatten
            return self.fc(x)                  # [B, 2]

    # --- Model Init ---
    seq_len = round_params.get("lookback_window", 32)
    d_model = round_params.get("d_model", 64)
    n_heads = round_params.get("n_heads", 2)
    n_layers = round_params.get("n_layers", 2)
    dropout = round_params.get("dropout", 0.1)
    n_features = X_train.shape[-1]

    net = SimpleTransformer(d_model, n_heads, n_layers, dropout, seq_len, n_features)

    # --- Training Setup ---
    criterion = nn.CrossEntropyLoss(label_smoothing=round_params.get("label_smoothing", 0.0))
    optimizer = optim.Adam(
        net.parameters(),
        lr=round_params.get("learning_rate", 1e-4),
        weight_decay=round_params.get("weight_decay", 0.0),
    )

    batch_size = round_params.get("batch_size", 32)
    n_epochs = round_params.get("n_epochs", 20)  # short for sweeps

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # --- Train Loop ---
    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        net.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # validation loss
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = net(xb)
                val_loss += criterion(out, yb).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # --- Final Evaluation on Test ---
    net.eval()
    with torch.no_grad():
        logits = net(X_test)
        probs = F.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    # --- Metrics ---
    report = classification_report(y_test.numpy(), preds, output_dict=True)
    confmat = confusion_matrix(y_test.numpy(), preds)

    # Fix: Convert numpy arrays to lists and extract positive class probabilities
    metrics = binary_metrics(
        {**data, 'y_test': y_test.numpy().tolist()}, 
        preds.tolist(), 
        probs[:, 1].tolist()  # Extract probabilities for positive class (class 1)
    )

    # --- Return in UEL format ---
    return {
        "models": [net],
        "extras": {
            "classification_report": report,
            "confusion_matrix": confmat.tolist(),
            "val_loss": best_val_loss,
            "params_used": round_params,
        },
        **metrics,  # merge UEL binary metrics
    }

