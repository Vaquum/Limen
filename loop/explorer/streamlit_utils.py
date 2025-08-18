# streamlit_utils.py

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Display / formatting utils
# -------------------------

def pretty_label(key: str) -> str:
    """
    Convert a column key like `class_weight` -> 'Class Weight'.
    """
    return " ".join(w.capitalize() for w in key.replace("_", " ").split())


def format_value(value, dtype) -> str:
    """
    Streamlit-safe, human-friendly formatter used in the details view.
    Mirrors the logic you had in streamlit_v3.py.
    """
    if pd.isna(value):
        return "—"

    # Integer
    if pd.api.types.is_integer_dtype(dtype):
        try:
            return f"{int(value):,}"
        except Exception:
            return f"{value}"

    # Float
    if pd.api.types.is_float_dtype(dtype):
        try:
            av = abs(float(value))
        except Exception:
            return f"{value}"

        if av >= 1000:
            return f"{float(value):,.2f}"
        if av >= 1:
            return f"{float(value):,.3g}"
        if av >= 0.01:
            return f"{float(value):.4f}"
        return f"{float(value):.6g}"

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        try:
            return f"{pd.to_datetime(value)}"
        except Exception:
            return f"{value}"

    # Fallback
    return f"{value}"


# ---------------------------------
# Quantile binning (fixed cut points)
# ---------------------------------

QUANTILE_BREAKS = [0.01, 0.25, 0.50, 0.75, 0.99]
QUANTILE_LABELS = ["≤1%", "1–25%", "25–50%", "50–75%", "75–99%", "≥99%"]


def quantile_bins_fixed(s: pd.Series) -> tuple[np.ndarray, list[str]]:
    """
    Returns (edges, labels) for fixed quantiles:
    1%, 25%, 50%, 75%, 99% with open tails [-inf, +inf].

    Edges are forced strictly increasing to avoid pandas cut() errors
    when repeated values collapse adjacent quantiles.
    """
    s = pd.to_numeric(s, errors="coerce")
    qs = s.quantile(QUANTILE_BREAKS).to_numpy()

    # Build edges: [-np.inf, q01, q25, q50, q75, q99, +np.inf]
    edges = np.asarray(np.concatenate(([-np.inf], qs, [np.inf])), dtype=float)

    # Ensure strictly increasing (handles equal quantiles)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)

    return edges, QUANTILE_LABELS


# -----------------------------
# Column typing / discovery utils
# -----------------------------

def split_num_cat(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Returns (numeric_columns, categorical_columns)."""
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    return num_cols, cat_cols


def is_numeric(series: pd.Series) -> bool:
    """Shorthand for a common check."""
    return pd.api.types.is_numeric_dtype(series)


def coerce_numeric(series: pd.Series) -> pd.Series:
    """To numeric with NaNs for non-coercible values."""
    return pd.to_numeric(series, errors="coerce")


# ----------------------
# Table plumbing helpers
# ----------------------

def add_drilldown_links(df: pd.DataFrame, rid_col: str = "rid") -> pd.DataFrame:
    """
    Insert a 'view' column with /?row=<rid> links for drill-down.
    Returns a copy.
    """
    out = df.copy()
    if rid_col not in out.columns:
        raise KeyError(f"Column '{rid_col}' not found for drilldown links.")
    out.insert(1, "view", [f"/?row={i}" for i in out[rid_col]])
    return out


def numeric_range(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Safe min/max for slider bounds with float coercion."""
    s = coerce_numeric(df[col])
    return float(s.min()), float(s.max())


def progress_col_config(df: pd.DataFrame, numeric_columns: list[str]) -> dict:
    """
    Build a st.dataframe column_config mapping for numeric ProgressColumn
    with per-column min/max hints.
    """
    cfg = {}
    for c in numeric_columns:
        s = pd.to_numeric(df[c], errors="coerce")
        cmin = float(s.min())
        cmax = float(s.max())
        # Avoid equal min/max which breaks progress bars
        if cmin == cmax:
            cmin -= 1.0
            cmax += 1.0
        cfg[c] = st.column_config.ProgressColumn(
            c,
            help=f"{c} (min={cmin:.3g}, max={cmax:.3g})",
            min_value=cmin,
            max_value=cmax,
            format="%.4g",
        )
    return cfg


# -----------------------------
# Smoothing / normalization utils
# -----------------------------

def rolling_mean_by_group(
    long_df: pd.DataFrame,
    value_col: str,
    group_col: str,
    window: int,
) -> pd.Series:
    """
    Rolling mean per group; min_periods=1 to match the app's UX.
    Returns a Series aligned with long_df.
    """
    if window <= 1:
        return long_df[value_col]
    return (
        long_df.groupby(group_col, observed=False)[value_col]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )


def minmax_scale_long(
    long_df: pd.DataFrame,
    value_col: str,
    group_col: str,
    out_range: tuple[float, float] = (-1.0, 1.0),
) -> pd.Series:
    """
    Min-max scale per group to a target range (default [-1, 1]).
    Constant series map to the midpoint of the range.
    """
    g = long_df.groupby(group_col, observed=False)[value_col]
    vmin = g.transform("min")
    vmax = g.transform("max")
    num = long_df[value_col] - vmin
    den = (vmax - vmin).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled01 = np.where(den == 0, 0.5, num / den)  # 0..1
    lo, hi = out_range
    return lo + (hi - lo) * scaled01