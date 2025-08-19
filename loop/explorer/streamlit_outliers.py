import pandas as pd
import polars as pl
import streamlit as st

from loop.transforms import (
    winsorize_transform,
    mad_transform,
    quantile_trim_transform,
    zscore_transform,
)


def render_outlier_controls(df: pd.DataFrame) -> str:
    """
    Render a compact icon toolbar and, when Outliers icon is active, a dropdown to select
    the outlier method. Returns the selected method name.
    """
    if "_show_outliers" not in st.session_state:
        st.session_state["_show_outliers"] = False  # default collapsed
    if "_show_time" not in st.session_state:
        st.session_state["_show_time"] = False

    # Icon toolbar (only first active for now)
    # Toolbar now lives in streamlit_toolbar.render_toolbar; keep this module focused on Outliers

    method = "None"
    has_numeric = len(df.select_dtypes("number").columns) > 0
    if has_numeric and st.session_state["_show_outliers"]:
        method = st.sidebar.selectbox(
            "Outlier Method",
            ["None", "Winsorize", "MAD Z-Score", "Quantile Trim", "Z-Score"],
            index=0,
        )
    return method


def apply_outlier_transform(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Apply the selected outlier method to all numeric columns (polars-backed),
    returning a pandas DataFrame.
    """
    if method == "None":
        return df

    # Convert to polars
    pl_df = pl.from_pandas(df)
    if method == "Winsorize":
        pl_out = winsorize_transform(pl_df)
    elif method == "MAD Z-Score":
        pl_out = mad_transform(pl_df)
    elif method == "Quantile Trim":
        pl_out = quantile_trim_transform(pl_df)
    elif method == "Z-Score":
        pl_out = zscore_transform(pl_df)
    else:
        pl_out = pl_df

    return pl_out.to_pandas()


