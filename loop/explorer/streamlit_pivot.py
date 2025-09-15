# streamlit_pivot.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

def render_pivot_table(
    df_filt: pd.DataFrame,
    *,
    pivot_rows: str | None,
    pivot_cols: str | None,
    pivot_val: str,
    agg: str,
    quantile_rows: bool,
    quantile_cols: bool,
    quantile_bins_fn,
) -> None:
    '''
    Compute and render a pivot table with optional quantile binning.
    Args:
        df_filt (pd.DataFrame): Klines dataset with categorical and numeric columns
        pivot_rows (str | None): Row dimension column name
        pivot_cols (str | None): Column dimension column name
        pivot_val (str): Value column to aggregate
        agg (str): Aggregation function name
        quantile_rows (bool): Whether to quantile-bin the row dimension
        quantile_cols (bool): Whether to quantile-bin the column dimension
        quantile_bins_fn (callable): Function that returns fixed quantile bin edges and labels
    Returns:
        None: None
    '''
    if not (pivot_rows or pivot_cols):
        st.warning("Select at least one pivot row or column to build the pivot table.")
        return

    work_df = df_filt.copy()
    idx = [pivot_rows] if pivot_rows else None
    cols = [pivot_cols] if pivot_cols else None

    row_header_prefix = None
    col_header_prefix = None

    # Rows -> Quantiles (fixed cuts), only if numeric and opted-in
    if pivot_rows and quantile_rows and pd.api.types.is_numeric_dtype(work_df[pivot_rows]):
        edges, labels = quantile_bins_fn(work_df[pivot_rows])
        qname = pivot_rows + "_q"
        work_df[qname] = pd.cut(work_df[pivot_rows], bins=edges, labels=labels, include_lowest=True)
        idx = [qname]
        row_header_prefix = f"{pivot_rows} (quantiles)"

    # Columns -> Quantiles (fixed cuts), only if numeric and opted-in
    if pivot_cols and quantile_cols and pd.api.types.is_numeric_dtype(work_df[pivot_cols]):
        edges, labels = quantile_bins_fn(work_df[pivot_cols])
        qname = pivot_cols + "_q"
        work_df[qname] = pd.cut(work_df[pivot_cols], bins=edges, labels=labels, include_lowest=True)
        cols = [qname]
        col_header_prefix = f"{pivot_cols} (quantiles)"

    # Map aggregation names to functions supported by pandas
    def _iqr_func(x: pd.Series) -> float:
        return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

    agg_map = {
        "min": "min",
        "max": "max",
        "sum": "sum",
        "mean": "mean",
        "std": "std",
        "median": "median",
        "iqr": _iqr_func,
        "count": "count",
    }

    pivot = pd.pivot_table(
        work_df,
        index=idx,
        columns=cols,
        values=pivot_val,
        aggfunc=agg_map.get(agg, agg),
        observed=False,
    )

    if isinstance(pivot.columns, pd.MultiIndex):
        pivot.columns = [" | ".join(map(str, tpl)).strip() for tpl in pivot.columns]

    # Prefix columns (when quantiled by columns)
    if cols and col_header_prefix:
        pivot.columns = [f"{col_header_prefix}: {c}" for c in pivot.columns]

    out = pivot.reset_index()

    # Rename quantiled row index for clarity
    if row_header_prefix and idx:
        out = out.rename(columns={idx[0]: row_header_prefix})

    st.subheader("Pivot table")
    st.dataframe(out, use_container_width=True, hide_index=True)