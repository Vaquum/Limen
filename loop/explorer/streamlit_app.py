# streamlit_v3.py

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px  # keep if other modules rely on px; safe to leave

from streamlit_styles import streamlit_styles
import streamlit_charts as charts
from streamlit_sidebar import build_sidebar
from streamlit_pivot import render_pivot_table
from streamlit_heatmap import render_corr_heatmap
from streamlit_table import prepare_table_data, render_table
from streamlit_details import render_details_view

import os, sys, argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--data", default=os.environ.get("DATA_PARQUET", "data.parquet"))
args, _ = parser.parse_known_args()

parquet_path = args.data
print(f"[streamlit] Using parquet: {parquet_path}", file=sys.stderr)
if not os.path.exists(parquet_path):
    raise FileNotFoundError(f"Parquet not found: {parquet_path}")


# NEW: shared helpers
from streamlit_utils import (
    pretty_label,
    format_value,
    quantile_bins_fixed,
)

st.set_page_config(page_title="Loop Explorer", layout="wide")
pd.set_option("styler.render.max_elements", 10_000_000)

# --------------------------------------------------------
# ---------- Global CSS (cards + compact sidebar dividers)
# --------------------------------------------------------
sidebar_container_gap_rem = 0.45
sidebar_divider_gap_rem   = 0.30

# Inject styles from the helper
st.markdown(
    streamlit_styles(sidebar_container_gap_rem, sidebar_divider_gap_rem),
    unsafe_allow_html=True,
)

# --------------
# ---- Load data
# --------------
#df = pd.read_parquet(os.environ.get("DATA_PARQUET", "data.parquet"))
# NEW
df = pd.read_parquet(parquet_path)

# ----------------
# ---- Detail view
# ----------------
render_details_view(
    df=df,
    pretty_label_fn=pretty_label,
    format_value_fn=format_value,
)

# -----------------------------
# Base view + custom columns
# -----------------------------
df_base = df.copy()
df_base.insert(0, "rid", range(len(df_base)))

if "custom_cols" not in st.session_state:
    st.session_state["custom_cols"] = []

if st.session_state["custom_cols"]:
    for _name, _expr in st.session_state["custom_cols"]:
        try:
            df_base[_name] = pd.eval(_expr, engine="python", local_dict=df_base.to_dict("series"))
        except Exception:
            pass

num_cols = df_base.select_dtypes("number").columns.tolist()
cat_cols = df_base.select_dtypes(exclude="number").columns.tolist()

# -----------------
# ---- Sidebar UI
# -----------------
sidebar_state = build_sidebar(
    df_base=df_base,
    num_cols=num_cols,
    cat_cols=cat_cols,
    sidebar_divider_gap_rem=sidebar_divider_gap_rem,
)

# ----------
# Show Table
# ----------
show_table         = sidebar_state["show_table"]
numeric_filter_col = sidebar_state["numeric_filter_col"]
num_range          = sidebar_state["num_range"]
fmt_mode           = sidebar_state["fmt_mode"]

# ----------
# Show Chart
# ----------
show_chart       = sidebar_state["show_chart"]
chart_type       = sidebar_state["chart_type"]
xcol             = sidebar_state["xcol"]
ycol             = sidebar_state["ycol"]
ycols            = sidebar_state["ycols"]
hue_col          = sidebar_state["hue_col"]
size_col         = sidebar_state["size_col"]
normalize_line   = sidebar_state["normalize_line"]
smoothing_window = sidebar_state["smoothing_window"]
area_normalize_100 = sidebar_state.get("area_normalize_100", True)

# ------------------------
# Show Correlation Heatmap
# ------------------------
show_corr = sidebar_state["show_corr"]
filter_outliers  = sidebar_state.get("filter_outliers", False)

# ----------------
# Show Pivot Table
# ----------------
show_pivot     = sidebar_state["show_pivot"]
pivot_rows     = sidebar_state["pivot_rows"]
pivot_cols     = sidebar_state["pivot_cols"]
pivot_val      = sidebar_state["pivot_val"]
agg            = sidebar_state["agg"]
quantile_rows  = sidebar_state["quantile_rows"]
quantile_cols  = sidebar_state["quantile_cols"]

# ---------------------------
# ---- Table data & render
# ---------------------------
df_filt, df_display = prepare_table_data(
    df_base=df_base,
    show_table=show_table,
    numeric_filter_col=numeric_filter_col,
    num_range=num_range,
    filter_outliers=filter_outliers,
)

if show_table:
    render_table(df_display, fmt_mode)

# -----------
# ---- Charts
# -----------
if show_chart:
    if chart_type == "Line" and xcol and ycols:
        charts.plot_line(
            df_filt,
            xcol=xcol,
            ycols=ycols,
            smoothing_window=smoothing_window,
            normalize_line=normalize_line,
            hue_col=hue_col,
            size_col=size_col,
        )

    elif chart_type == "Area" and xcol and ycols:
        charts.plot_area(
            df_filt,
            xcol=xcol,
            ycols=ycols,
            smoothing_window=smoothing_window,
            normalize_100=area_normalize_100,
        )

    elif chart_type == "Scatter" and xcol and ycol:
        charts.plot_scatter(
            df_filt,
            xcol=xcol,
            ycol=ycol,
            hue_col=hue_col,
            size_col=size_col,
        )

    elif chart_type == "Box" and xcol and ycol:
        charts.plot_box(
            df_filt,
            xcol=xcol,
            ycol=ycol,
            hue_col=hue_col,
        )

    else:
        if ycol:
            charts.plot_histogram(
                df_filt,
                ycol=ycol,
                hue_col=hue_col,
            )

# ------------------------
# ---- Correlation Heatmap
# ------------------------
if show_corr:
    render_corr_heatmap(df_filt, num_cols)

# ----------------
# ---- Pivot Table
# ----------------
if show_pivot and pivot_val:
    render_pivot_table(
        df_filt,
        pivot_rows=pivot_rows,
        pivot_cols=pivot_cols,
        pivot_val=pivot_val,
        agg=agg,
        quantile_rows=quantile_rows,
        quantile_cols=quantile_cols,
        quantile_bins_fn=quantile_bins_fixed,  # <-- from utils
    )