import os, sys, argparse
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from loop.explorer.streamlit_styles import streamlit_styles
import loop.explorer.streamlit_charts as charts
from loop.explorer.streamlit_sidebar import build_sidebar
from loop.explorer.streamlit_pivot import render_pivot_table
from loop.explorer.streamlit_heatmap import render_corr_heatmap
from loop.explorer.streamlit_table import prepare_table_data, render_table
from loop.explorer.streamlit_details import render_details_view
from loop.explorer.streamlit_utils import pretty_label, format_value, quantile_bins_fixed
from loop.explorer.streamlit_outliers import render_outlier_controls, apply_outlier_transform
from loop.explorer.streamlit_time import render_time_controls, apply_time_filter
from loop.explorer.streamlit_trend import render_trend_controls, apply_trend_regime


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--data', default=os.environ.get('DATA_PARQUET', 'data.parquet'))
args, _ = parser.parse_known_args()

parquet_path = args.data
# Avoid writing to stderr during Streamlit boot to prevent BrokenPipeError in
# certain headless environments. The information is not critical for runtime.
if not os.path.exists(parquet_path):
    raise FileNotFoundError(f"Parquet not found: {parquet_path}")

st.set_page_config(page_title='Loop Explorer', layout='wide')

# -----------------------------
# Handle toolbar toggle via query param (?toggle=...)
# Compatible with both new and experimental query param APIs
# -----------------------------
def _get_query_params() -> dict:
    try:
        qp = st.query_params
        return dict(qp) if isinstance(qp, dict) else {}
    except Exception:
        try:
            return {k: v[0] if isinstance(v, list) and v else v for k, v in st.experimental_get_query_params().items()}
        except Exception:
            return {}

def _set_query_params(params: dict) -> None:
    try:
        st.query_params.update(params)
    except Exception:
        try:
            st.experimental_set_query_params(**params)
        except Exception:
            pass

qp = _get_query_params()
_toggle = qp.get('toggle')
if _toggle:
    # Map toggle names to session flags
    mapping = {
        'outliers': '_show_outliers',
        'time': '_show_time',
        'trend': '_show_trend',
        'dataset': '_show_dataset',
    }
    flag = mapping.get(_toggle)
    if flag:
        st.session_state[flag] = not st.session_state.get(flag, False)
    # Clear the query param to keep URL clean
    qp.pop('toggle', None)
    _set_query_params(qp)
    st.rerun()
pd.set_option('styler.render.max_elements', 10_000_000)

# --------------------------------------------------------
# ---------- Global CSS (cards + compact sidebar dividers)
# --------------------------------------------------------
sidebar_container_gap_rem = 0.0
sidebar_divider_gap_rem   = 0.30

st.markdown(
    streamlit_styles(sidebar_container_gap_rem, sidebar_divider_gap_rem),
    unsafe_allow_html=True,
)

# --------------
# ---- Load data
# --------------
@st.cache_data(show_spinner=False)
def _load_parquet(path: str, mtime: float | None) -> pd.DataFrame:
    # mtime is unused beyond caching; included to bust cache when file changes
    return pd.read_parquet(path)

def _mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

dataset_name = st.session_state.get('dataset_select', 'Historical Data')
name_to_path = {
    'Historical Data': parquet_path,
    'Experiment Log': '/tmp/experiment_log.parquet',
    'Confusion Metrics': '/tmp/confusion_metrics.parquet',
    'Backtest Results': '/tmp/backtest_results.parquet',
}
selected_path = name_to_path.get(dataset_name, parquet_path)

# Strict loading: try primary (and legacy alias if defined); otherwise, raise
legacy_map = {
    'Confusion Metrics': '/tmp/experiment_confusion_metrics.parquet',
    'Backtest Results': '/tmp/experiment_backtest_results.parquet',
}
candidates = [selected_path]
if dataset_name in legacy_map:
    candidates.append(legacy_map[dataset_name])

existing_path = next((p for p in candidates if os.path.exists(p)), None)
if not existing_path:
    st.error(f'Missing dataset parquet for "{dataset_name}". Tried: {candidates}')
    raise FileNotFoundError(f'Parquet not found for {dataset_name}: {candidates}')

df = _load_parquet(existing_path, _mtime(existing_path))

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
df_base.insert(0, 'rid', range(len(df_base)))

if 'custom_cols' not in st.session_state:
    st.session_state['custom_cols'] = []

if st.session_state['custom_cols']:
    for _name, _expr in st.session_state['custom_cols']:
        try:
            df_base[_name] = pd.eval(_expr, engine='python', local_dict=df_base.to_dict('series'))
        except Exception:
            pass

num_cols = df_base.select_dtypes('number').columns.tolist()
cat_cols = df_base.select_dtypes(exclude='number').columns.tolist()

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
show_table         = sidebar_state['show_table']
numeric_filter_col = sidebar_state['numeric_filter_col']
num_range          = sidebar_state['num_range']
fmt_mode           = sidebar_state['fmt_mode']
selected_columns   = sidebar_state.get('selected_columns', [])

# ----------
# Show Chart
# ----------
show_chart       = sidebar_state['show_chart']
chart_type       = sidebar_state['chart_type']
xcol             = sidebar_state['xcol']
ycol             = sidebar_state['ycol']
ycols            = sidebar_state['ycols']
hue_col          = sidebar_state['hue_col']
size_col         = sidebar_state['size_col']
normalize_line   = sidebar_state['normalize_line']
smoothing_window = sidebar_state['smoothing_window']
area_normalize_100 = sidebar_state.get('area_normalize_100', True)
normalize_counts_hist = sidebar_state.get('normalize_counts_hist', False)
normalize_data_hist = sidebar_state.get('normalize_data_hist', False)

# ------------------------
# Show Correlation Heatmap
# ------------------------
show_corr = sidebar_state['show_corr']

# ----------------
# Global Outlier Control
# ----------------
outlier_method = render_outlier_controls(df_base)
df_out = apply_outlier_transform(df_base, outlier_method)

# ----------------
# Global Time Control
# ----------------
time_settings = render_time_controls(df_out)
df_out2 = apply_time_filter(df_out, time_settings)

# ----------------
# Global Trend Control
# ----------------
# Toggle is in the toolbar; controls render only when active
trend_settings = render_trend_controls(df_out2)
df_with_regime, df_out3 = apply_trend_regime(df_out2, trend_settings)

# ---------------------------
# ---- Table data & render
# ---------------------------
df_filt, df_display = prepare_table_data(
    df_base=df_out3,
    show_table=show_table,
    numeric_filter_col=numeric_filter_col,
    num_range=num_range,
)

if show_table and selected_columns:
    # Respect user selection fully. If empty, do not render the table.
    keep = [c for c in selected_columns if c in df_display.columns]
    if keep:
        render_table(df_display[keep], fmt_mode, column_order=keep)

# -----------
# ---- Charts
# -----------
if show_chart:
    if chart_type == 'Line' and xcol and ycols:
        charts.plot_line(
            df_filt,
            xcol=xcol,
            ycols=ycols,
            smoothing_window=smoothing_window,
            normalize_line=normalize_line,
            hue_col=hue_col,
            size_col=size_col,
        )

    elif chart_type == 'Area' and xcol and ycols:
        charts.plot_area(
            df_filt,
            xcol=xcol,
            ycols=ycols,
            smoothing_window=smoothing_window,
            normalize_100=area_normalize_100,
        )

    elif chart_type == 'Scatter' and xcol and ycol:
        charts.plot_scatter(
            df_filt,
            xcol=xcol,
            ycol=ycol,
            hue_col=hue_col,
            size_col=size_col,
        )

    elif chart_type == 'Box' and xcol and ycol:
        charts.plot_box(
            df_filt,
            xcol=xcol,
            ycol=ycol,
            hue_col=hue_col,
        )

    else:
        # Histogram supports multi y
        if ycols:
            charts.plot_histogram(
                df_filt,
                ycols=ycols,
                normalize_data=normalize_data_hist,
                normalize_counts=normalize_counts_hist,
                hue_col=hue_col,
            )
        elif ycol:
            charts.plot_histogram(
                df_filt,
                ycol=ycol,
                normalize_data=normalize_data_hist,
                normalize_counts=normalize_counts_hist,
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
show_pivot     = sidebar_state['show_pivot']
pivot_val      = sidebar_state['pivot_val']
if show_pivot and pivot_val:
    render_pivot_table(
        df_filt,
        pivot_rows=sidebar_state['pivot_rows'],
        pivot_cols=sidebar_state['pivot_cols'],
        pivot_val=pivot_val,
        agg=sidebar_state['agg'],
        quantile_rows=sidebar_state['quantile_rows'],
        quantile_cols=sidebar_state['quantile_cols'],
        quantile_bins_fn=quantile_bins_fixed,
    )