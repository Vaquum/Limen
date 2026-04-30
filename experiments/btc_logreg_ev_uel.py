"""Single-file Limen experiment for BTC next-bar-up LogReg research.

Run from the repository root:

    uv run python experiments/btc_logreg_ev_uel.py

The file is intentionally self-contained: imports, local helper callables,
manifest, historical data retrieval, and the final UEL run all live here.
"""

from __future__ import annotations

import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import limen
from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.features import amihud_illiquidity
from limen.features import body_to_range
from limen.features import close_position
from limen.features import cyclical_time_features
from limen.features import garman_klass_volatility
from limen.features import kline_imbalance
from limen.features import parkinson_volatility
from limen.features import range_pct
from limen.features import realized_semivariance
from limen.features import volatility_of_volatility
from limen.features import vwap
from limen.features import wick_imbalance
from limen.indicators import atr
from limen.indicators import roc
from limen.indicators import wilder_rsi
from limen.indicators import window_return
from limen.scalers.registry import SCALER_REGISTRY


LOGGER = logging.getLogger(__name__)
RAW_MARKET_COLUMNS = {
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "mean",
    "std",
    "volume",
    "maker_ratio",
    "no_of_trades",
    "open_liquidity",
    "high_liquidity",
    "low_liquidity",
    "close_liquidity",
    "liquidity_sum",
    "maker_volume",
    "maker_liquidity",
}
EV_RETURN_COLUMN = "ev_forward_return"
EPSILON = 1e-12
BACKTEST_METRIC_KEYS = [
    "trade_win_rate_pct",
    "trade_expectancy_pct",
    "max_drawdown_pct",
    "total_return_gross_pct",
    "total_return_net_pct",
    "trade_return_mean_win_pct",
    "trade_return_mean_loss_pct",
    "mean_kelly_pct",
    "bars_total",
    "sharpe_per_bar",
    "bars_in_market_pct",
    "trades_count",
    "cost_round_trip_bps",
]


def consensus_logreg_features(
    data: pl.DataFrame | pl.LazyFrame,
    short_window: int = 8,
    medium_window: int = 32,
    long_window: int = 96,
) -> pl.DataFrame | pl.LazyFrame:
    """Add compact consensus features missing from the built-in helper set."""

    close = pl.col("close")
    volume = pl.col("volume")
    returns = close.pct_change()
    short_vol = returns.rolling_std(window_size=short_window)
    long_vol = returns.rolling_std(window_size=long_window)
    ema_short = close.ewm_mean(span=short_window, adjust=False)
    ema_medium = close.ewm_mean(span=medium_window, adjust=False)

    return data.with_columns(
        [
            (close / close.shift(short_window)).log().alias("log_return_short"),
            (close / close.shift(medium_window)).log().alias("log_return_medium"),
            (close / close.shift(long_window)).log().alias("log_return_long"),
            (short_vol / (long_vol + EPSILON)).alias("volatility_ratio"),
            (close / (ema_short + EPSILON)).log().alias("ema_distance_short"),
            (close / (ema_medium + EPSILON)).log().alias("ema_distance_medium"),
            ((volume + 1.0).log() - (volume.shift(1) + 1.0).log()).alias(
                "log_volume_change"
            ),
            (1.0 - (2.0 * pl.col("maker_ratio"))).alias("maker_pressure"),
            (
                (pl.col("high") - pl.col("low"))
                / (pl.col("close") + EPSILON)
                / ((pl.col("liquidity_sum") + 1.0).log() + EPSILON)
            ).alias("range_per_log_liquidity"),
        ]
    )


def next_bar_up_target(
    data: pl.DataFrame,
    column: str,
    horizon: int = 1,
    return_column: str = EV_RETURN_COLUMN,
) -> pl.DataFrame:
    """Create the pure next-bar-up label plus an auxiliary forward return."""

    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    forward_return = (pl.col("close").shift(-horizon) / pl.col("close")).log()
    return data.with_columns(
        [
            forward_return.alias(return_column),
            (forward_return > 0.0).cast(pl.Int8).alias(column),
        ]
    )


class CausalRollingRobustScaler:
    """Causal rolling median/IQR scaler for model features.

    The transform excludes raw market fields and the EV helper return from
    scaling. For validation/test splits, the scaler prepends the train tail
    solely to seed trailing statistics, then removes that prefix before
    returning the transformed split.
    """

    window: int = 2000
    min_samples: int = 128
    clip: float = 5.0

    def __init__(self, x_train: pl.DataFrame) -> None:
        self.train_height = x_train.height
        self.columns = [
            col
            for col in x_train.columns
            if col not in RAW_MARKET_COLUMNS
            and col != EV_RETURN_COLUMN
            and x_train[col].dtype.is_numeric()
        ]
        self.tail = x_train.tail(self.window)
        self._transform_calls = 0

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.columns or df.height == 0:
            return df

        use_prefix = self._transform_calls > 0
        self._transform_calls += 1
        prefix_len = self.tail.height if use_prefix else 0
        work = pl.concat([self.tail, df], how="vertical") if use_prefix else df

        exprs = []
        for col in self.columns:
            median = (
                pl.col(col)
                .rolling_median(
                    window_size=self.window,
                    min_samples=self.min_samples,
                )
                .shift(1)
            )
            q_low = (
                pl.col(col)
                .rolling_quantile(
                    0.25,
                    window_size=self.window,
                    min_samples=self.min_samples,
                )
                .shift(1)
            )
            q_high = (
                pl.col(col)
                .rolling_quantile(
                    0.75,
                    window_size=self.window,
                    min_samples=self.min_samples,
                )
                .shift(1)
            )
            iqr = q_high - q_low
            scaled = (
                (pl.col(col) - median)
                / pl.when(iqr.abs() > EPSILON).then(iqr).otherwise(1.0)
            ).clip(-self.clip, self.clip)
            exprs.append(scaled.alias(col))

        transformed = work.with_columns(exprs)
        self.tail = work.tail(self.window)
        if prefix_len:
            transformed = transformed.slice(prefix_len)
        return transformed


class CausalRollingRobustScaler1000(CausalRollingRobustScaler):
    window = 1000


class CausalRollingRobustScaler2000(CausalRollingRobustScaler):
    window = 2000


class CausalRollingRobustScaler5000(CausalRollingRobustScaler):
    window = 5000


SCALER_REGISTRY.update(
    {
        "rolling_robust_1000": CausalRollingRobustScaler1000,
        "rolling_robust_2000": CausalRollingRobustScaler2000,
        "rolling_robust_5000": CausalRollingRobustScaler5000,
    }
)


def _to_numpy(frame_or_series: Any) -> np.ndarray:
    if hasattr(frame_or_series, "to_numpy"):
        return frame_or_series.to_numpy()
    return np.asarray(frame_or_series)


def _model_columns(frame: pl.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in RAW_MARKET_COLUMNS
        and col != EV_RETURN_COLUMN
        and frame[col].dtype.is_numeric()
    ]


def _safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return math.nan
    return float(roc_auc_score(y_true, probs))


def _safe_log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return math.nan
    return float(log_loss(y_true, np.clip(probs, 1e-6, 1.0 - 1e-6), labels=[0, 1]))


def _expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y_true[mask].mean()) - float(probs[mask].mean()))
    return float(ece)


def _winsorized_mean(values: np.ndarray, q: float = 0.1) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return 0.0
    lo, hi = np.quantile(clean, [q, 1.0 - q])
    return float(np.clip(clean, lo, hi).mean())


def _scalar_payoff_estimates(
    returns: np.ndarray,
    shrinkage_lambda: float,
) -> tuple[float, float]:
    positive = returns[returns > 0.0]
    negative = -returns[returns < 0.0]
    unconditional = _winsorized_mean(np.abs(returns))
    gain = _winsorized_mean(positive)
    loss = _winsorized_mean(negative)
    gain = ((1.0 - shrinkage_lambda) * gain) + (shrinkage_lambda * unconditional)
    loss = ((1.0 - shrinkage_lambda) * loss) + (shrinkage_lambda * unconditional)
    return max(gain, EPSILON), max(loss, EPSILON)


def _bucketed_payoff_estimates(
    val_probs: np.ndarray,
    val_returns: np.ndarray,
    test_probs: np.ndarray,
    shrinkage_lambda: float,
    bucket_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    fallback_gain, fallback_loss = _scalar_payoff_estimates(
        val_returns,
        shrinkage_lambda,
    )
    fallback = (
        np.full_like(test_probs, fallback_gain, dtype=float),
        np.full_like(test_probs, fallback_loss, dtype=float),
    )
    mask = np.isfinite(val_probs) & np.isfinite(val_returns)
    if bucket_count <= 1 or mask.sum() < max(50, bucket_count * 10):
        return fallback

    probs = val_probs[mask]
    returns = val_returns[mask]
    edges = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, bucket_count + 1)))
    if edges.size < 3:
        return fallback

    bucket_ids = np.digitize(probs, edges[1:-1], right=True)
    gains = np.full(edges.size - 1, fallback_gain, dtype=float)
    losses = np.full(edges.size - 1, fallback_loss, dtype=float)
    for bucket in range(edges.size - 1):
        bucket_returns = returns[bucket_ids == bucket]
        if bucket_returns.size < 10:
            continue
        gains[bucket], losses[bucket] = _scalar_payoff_estimates(
            bucket_returns,
            shrinkage_lambda,
        )

    test_bucket_ids = np.digitize(test_probs, edges[1:-1], right=True)
    test_bucket_ids = np.clip(test_bucket_ids, 0, len(gains) - 1)
    return gains[test_bucket_ids], losses[test_bucket_ids]


def _cost_multiplier(val_frame: pl.DataFrame, test_frame: pl.DataFrame) -> np.ndarray:
    val_range = (
        (val_frame["high"] - val_frame["low"]) / (val_frame["close"] + EPSILON)
    ).to_numpy()
    test_range = (
        (test_frame["high"] - test_frame["low"]) / (test_frame["close"] + EPSILON)
    ).to_numpy()
    val_liquidity = np.log1p(val_frame["liquidity_sum"].to_numpy())
    test_liquidity = np.log1p(test_frame["liquidity_sum"].to_numpy())

    range_ref = max(float(np.nanmedian(val_range[np.isfinite(val_range)])), EPSILON)
    liquidity_ref = max(
        float(np.nanmedian(val_liquidity[np.isfinite(val_liquidity)])),
        EPSILON,
    )
    range_ratio = test_range / range_ref
    liquidity_ratio = liquidity_ref / (test_liquidity + EPSILON)
    multiplier = (0.7 * range_ratio) + (0.3 * liquidity_ratio)
    return np.clip(multiplier, 0.5, 3.0)


def _apply_sigmoid_calibration(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    scores: list[np.ndarray],
    calibration_c: float,
) -> list[np.ndarray]:
    if np.unique(y_val).size < 2:
        return [1.0 / (1.0 + np.exp(-score)) for score in scores]

    calibrator = LogisticRegression(C=calibration_c, solver="lbfgs", max_iter=1000)
    calibrator.fit(val_scores.reshape(-1, 1), y_val)
    return [calibrator.predict_proba(score.reshape(-1, 1))[:, 1] for score in scores]


def _apply_isotonic_calibration(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    scores: list[np.ndarray],
) -> list[np.ndarray]:
    if np.unique(y_val).size < 2 or y_val.size < 4000:
        return [1.0 / (1.0 + np.exp(-score)) for score in scores]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_scores, y_val)
    return [calibrator.predict(score) for score in scores]


def _stateful_ev_predictions(
    probs: np.ndarray,
    ev: np.ndarray,
    p_enter: float,
    p_exit: float,
    ev_enter: float,
    ev_exit: float,
) -> np.ndarray:
    state = 0
    preds = np.zeros_like(probs, dtype=np.int8)
    effective_p_exit = min(p_exit, p_enter - 0.02)

    for i, (prob, edge) in enumerate(zip(probs, ev, strict=True)):
        if state == 0 and edge > ev_enter and prob >= max(0.5, p_enter):
            state = 1
        elif state == 1 and (edge < ev_exit or prob < effective_p_exit):
            state = 0
        preds[i] = state

    return preds


def _kelly_fraction(
    ev: np.ndarray,
    variance_returns: np.ndarray,
    kelly_fraction: float,
    max_exposure: float,
) -> np.ndarray:
    clean = variance_returns[np.isfinite(variance_returns)]
    variance = float(np.nanvar(clean)) if clean.size else EPSILON
    downside = np.minimum(clean, 0.0)
    downside_semivariance = float(np.mean(downside**2)) if downside.size else variance
    robust_variance = max(variance, downside_semivariance, EPSILON)
    raw = np.maximum(ev, 0.0) / robust_variance
    return np.clip(raw * kelly_fraction, 0.0, max_exposure)


def _empty_backtest_metrics() -> dict[str, float]:
    return {f"backtest_{key}": math.nan for key in BACKTEST_METRIC_KEYS}


def _backtest_metrics(
    preds: np.ndarray,
    data: dict,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, float]:
    if "price_data_for_backtest" not in data:
        return _empty_backtest_metrics()

    price_pd = data["price_data_for_backtest"].to_pandas()
    bt_input = {
        "predictions": np.asarray(preds).astype(int),
        "open": price_pd["open"].to_numpy(),
        "close": price_pd["close"].to_numpy(),
    }
    bt_input["price_change"] = bt_input["close"] - bt_input["open"]
    try:
        result = backtest_snapshot(
            pl.DataFrame(bt_input).to_pandas(),
            execution_lag_bars=1,
            fee_bps=fee_bps,
            slip_bps=slippage_bps,
        )
    except ValueError:
        return _empty_backtest_metrics()
    if result.empty:
        return _empty_backtest_metrics()
    return {
        f"backtest_{key}": float(value)
        for key, value in result.iloc[0].to_dict().items()
    }


def ev_logreg_binary(
    data: dict,
    *,
    penalty: str = "l2",
    C: float = 0.1,
    l1_ratio: float = 0.0,
    class_weight_mode: str = "none",
    max_iter: int = 1000,
    tol: float = 1e-4,
    calibration_method: str = "sigmoid",
    calibration_c: float = 1.0,
    p_enter: float = 0.52,
    p_exit: float = 0.48,
    ev_enter_bps: float = 0.0,
    ev_exit_bps: float = -1.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
    prob_shrinkage: float = 0.5,
    payoff_shrinkage: float = 0.3,
    payoff_bucket_count: int = 5,
    kelly_fraction: float = 0.1,
    max_exposure: float = 0.5,
    random_state: int = 42,
) -> dict:
    """Fit calibrated LogReg and evaluate EV-gated long-only predictions."""

    feature_cols = _model_columns(data["x_train"])
    X_train = _to_numpy(data["x_train"].select(feature_cols))
    X_val = _to_numpy(data["x_val"].select(feature_cols))
    X_test = _to_numpy(data["x_test"].select(feature_cols))
    y_train = _to_numpy(data["y_train"]).astype(int).ravel()
    y_val = _to_numpy(data["y_val"]).astype(int).ravel()
    y_test = _to_numpy(data["y_test"]).astype(int).ravel()

    solver = "lbfgs" if penalty == "l2" else "saga"
    model_kwargs: dict[str, Any] = {
        "penalty": penalty,
        "C": C,
        "solver": solver,
        "fit_intercept": True,
        "class_weight": None if class_weight_mode == "none" else "balanced",
        "max_iter": max_iter,
        "tol": tol,
        "random_state": random_state,
    }
    if solver == "saga":
        model_kwargs["n_jobs"] = -1
    if penalty == "elasticnet":
        model_kwargs["l1_ratio"] = l1_ratio

    clf = LogisticRegression(**model_kwargs)
    clf.fit(X_train, y_train)

    val_scores = clf.decision_function(X_val)
    test_scores = clf.decision_function(X_test)
    if calibration_method == "isotonic":
        val_probs, test_probs = _apply_isotonic_calibration(
            val_scores,
            y_val,
            [val_scores, test_scores],
        )
    elif calibration_method == "none":
        val_probs = clf.predict_proba(X_val)[:, 1]
        test_probs = clf.predict_proba(X_test)[:, 1]
    else:
        val_probs, test_probs = _apply_sigmoid_calibration(
            val_scores,
            y_val,
            [val_scores, test_scores],
            calibration_c,
        )

    val_returns = _to_numpy(data["x_val"][EV_RETURN_COLUMN]).astype(float).ravel()
    test_returns = _to_numpy(data["x_test"][EV_RETURN_COLUMN]).astype(float).ravel()
    gain_hat, loss_hat = _bucketed_payoff_estimates(
        val_probs,
        val_returns,
        test_probs,
        payoff_shrinkage,
        payoff_bucket_count,
    )

    shrunk_test_probs = 0.5 + (prob_shrinkage * (test_probs - 0.5))
    cost_multiplier = _cost_multiplier(data["x_val"], data["x_test"])
    cost = (2.0 * (fee_bps + (slippage_bps * cost_multiplier))) / 10_000.0
    test_ev = (
        (shrunk_test_probs * gain_hat)
        - ((1.0 - shrunk_test_probs) * loss_hat)
        - cost
    )
    preds = _stateful_ev_predictions(
        shrunk_test_probs,
        test_ev,
        p_enter,
        p_exit,
        ev_enter_bps / 10_000.0,
        ev_exit_bps / 10_000.0,
    )
    target_fraction = _kelly_fraction(
        test_ev,
        val_returns,
        kelly_fraction,
        max_exposure,
    )

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    results: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "auc": round(_safe_auc(y_test, test_probs), 4),
        "brier": round(float(brier_score_loss(y_test, test_probs)), 6),
        "log_loss": round(_safe_log_loss(y_test, test_probs), 6),
        "ece_10": round(_expected_calibration_error(y_test, test_probs, n_bins=10), 6),
        "val_brier": round(float(brier_score_loss(y_val, val_probs)), 6),
        "p_up_mean": round(float(np.mean(test_probs)), 6),
        "p_up_std": round(float(np.std(test_probs)), 6),
        "signal_rate_pct": round(float(np.mean(preds) * 100.0), 3),
        "ev_mean_bps": round(float(np.mean(test_ev) * 10_000.0), 4),
        "ev_signal_mean_bps": round(
            float(np.mean(test_ev[preds == 1]) * 10_000.0) if preds.any() else math.nan,
            4,
        ),
        "gain_hat_mean_bps": round(float(np.mean(gain_hat) * 10_000.0), 4),
        "loss_hat_mean_bps": round(float(np.mean(loss_hat) * 10_000.0), 4),
        "cost_round_trip_mean_bps_model": round(float(np.mean(cost) * 10_000.0), 4),
        "cost_round_trip_max_bps_model": round(float(np.max(cost) * 10_000.0), 4),
        "kelly_target_mean_pct": round(float(np.mean(target_fraction) * 100.0), 4),
        "kelly_target_max_pct": round(float(np.max(target_fraction) * 100.0), 4),
        "feature_count": len(feature_cols),
        "_preds": preds,
    }
    results.update(_backtest_metrics(preds, data, fee_bps, slippage_bps))
    return results


def params() -> dict[str, list[Any]]:
    """Search space for the single-file SFD."""

    return {
        "roc_period": [1, 4, 8, 16],
        "return_window": [4, 16, 64],
        "short_window": [8, 16],
        "medium_window": [32, 64],
        "long_window": [96, 128],
        "vol_window": [16, 32, 64],
        "vol_of_vol_window": [32, 64],
        "imbalance_window": [8, 16, 32],
        "rsi_period": [14, 28],
        "atr_period": [14, 28],
        "feature_groups": [
            "all",
            "returns|volatility|consensus",
            "returns|structure|liquidity",
            "returns|time|consensus",
        ],
        "scaler_type": ["rolling_robust_1000", "rolling_robust_2000"],
        "penalty": ["l2", "elasticnet"],
        "C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        "l1_ratio": [0.0, 0.1, 0.3, 0.5],
        "class_weight_mode": ["none", "balanced"],
        "max_iter": [1000],
        "tol": [0.0001, 0.001],
        "calibration_method": ["sigmoid"],
        "calibration_c": [0.3, 1.0, 3.0],
        "p_enter": [0.5, 0.51, 0.52, 0.55],
        "p_exit": [0.45, 0.48, 0.5],
        "ev_enter_bps": [0.0, 1.0, 2.0],
        "ev_exit_bps": [-2.0, 0.0],
        "fee_bps": [1.0, 4.0, 6.0],
        "slippage_bps": [1.0, 2.0, 4.0],
        "prob_shrinkage": [0.5, 0.7, 1.0],
        "payoff_shrinkage": [0.2, 0.3, 0.5],
        "payoff_bucket_count": [1, 5, 10],
        "kelly_fraction": [0.05, 0.1, 0.15, 0.2],
        "max_exposure": [0.25, 0.5, 1.0],
        "random_state": [42],
    }


def manifest() -> Manifest:
    """Manifest-driven SFD wiring for the BTC LogReg EV experiment."""

    return (
        Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={"kline_size": 3600, "n_rows": 6000},
        )
        .set_test_data_source(
            method=HistoricalData.get_spot_klines,
            params={"kline_size": 3600, "n_rows": 3000},
        )
        .set_split_config(8, 1, 2)
        .add_feature(
            consensus_logreg_features,
            group="consensus",
            short_window="short_window",
            medium_window="medium_window",
            long_window="long_window",
        )
        .add_indicator(roc, group="returns", period="roc_period")
        .add_indicator(window_return, group="returns", period="return_window")
        .add_indicator(atr, group="volatility", period="atr_period")
        .add_indicator(wilder_rsi, group="structure", period="rsi_period")
        .add_feature(parkinson_volatility, group="volatility", window="vol_window")
        .add_feature(garman_klass_volatility, group="volatility", window="vol_window")
        .add_feature(realized_semivariance, group="volatility", window="vol_window")
        .add_feature(
            volatility_of_volatility,
            group="volatility",
            volatility_window="short_window",
            window="vol_of_vol_window",
        )
        .add_feature(range_pct, group="structure")
        .add_feature(close_position, group="structure")
        .add_feature(body_to_range, group="structure")
        .add_feature(wick_imbalance, group="structure")
        .add_feature(vwap, group="liquidity")
        .add_feature(kline_imbalance, group="liquidity", window="imbalance_window")
        .add_feature(amihud_illiquidity, group="liquidity")
        .add_feature(cyclical_time_features, group="time")
        .with_target("next_bar_up")
        .add_transform(
            next_bar_up_target,
            column="target_column",
            horizon=1,
        )
        .done()
        .set_scaler_from_params("scaler_type")
        .with_model(ev_logreg_binary)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    historical = HistoricalData()
    data = historical.get_spot_klines(kline_size=3600, n_rows=6000)

    output_dir = Path("experiment_outputs") / "btc_logreg_ev_uel"
    output_csv = output_dir / "btc-logreg-ev-uel.csv"
    output_csv.unlink(missing_ok=True)
    random.seed(42)
    np.random.seed(42)

    uel = limen.UniversalExperimentLoop(
        data=data,
        sfd=sys.modules[__name__],
        experiment_dir=output_dir,
    )
    uel.run(
        experiment_name="btc-logreg-ev-uel",
        n_permutations=100,
        prep_each_round=True,
        random_search=True,
    )
    LOGGER.info("Final rows:\n%s", uel.experiment_log.tail(5))
