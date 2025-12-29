from __future__ import annotations
import pandas as pd
import streamlit as st
from loop.explorer.streamlit_toolbar import render_toolbar


def _tight_divider(gap_rem: float) -> None:
    st.markdown(
        f"<hr style='margin:{gap_rem}rem 0; border:0; border-top:1px solid rgba(0,0,0,.12);' />",
        unsafe_allow_html=True,
    )


def build_sidebar(
    df_base: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    sidebar_divider_gap_rem: float = 0.30,
):
    """
    Render sidebar controls and return the collected UI state.
    Args:
        df_base (pd.DataFrame): Klines dataset with base columns for selection and preview
        num_cols (list[str]): Numeric column names
        cat_cols (list[str]): Categorical column names
        sidebar_divider_gap_rem (float): Vertical spacing between sections (rem)
    Returns:
        dict: Sidebar state dictionary consumed by the main app
    """

    with st.sidebar:
        # One-shot reset for the custom column creator. This sets the widget
        # value before instantiation, avoiding Streamlit's constraint about
        # modifying a widget key after it has been created.
        if st.session_state.pop("_reset_enable_custom_col", False):
            st.session_state["enable_custom_col"] = False

        # Top toolbar icons
        render_toolbar()

        # Dataset selector
        if st.session_state.get("_show_dataset", False):
            ds = st.selectbox(
                "Dataset",
                [
                    "Historical Data",
                    "Experiment Log",
                    "Confusion Metrics",
                    "Backtest Results",
                ],
                index=0,
                key="dataset_select",
            )
            st.session_state["dataset_name"] = ds
            _tight_divider(sidebar_divider_gap_rem)

        # --- Show Table + its options (persisted via explicit keys)
        show_table = st.checkbox("**Show Table**", value=False, key="table_show")

        numeric_filter_col = None
        num_range = None
        fmt_mode = "Normal"
        selected_columns: list[str] = []
        if show_table:
            numeric_filter_col = st.selectbox(
                "Filter by Column Value",
                [""] + num_cols,
                index=0,
                key="table_numeric_filter_col",
            )
            if numeric_filter_col:
                series = df_base[numeric_filter_col]
                if pd.api.types.is_integer_dtype(series):
                    col_min = int(series.min())
                    col_max = int(series.max())
                    num_range = st.slider(
                        f"{numeric_filter_col} range",
                        min_value=col_min,
                        max_value=col_max,
                        value=(col_min, col_max),
                        step=1,
                        key="table_num_range",
                    )
                else:
                    col_min = float(series.min())
                    col_max = float(series.max())
                    num_range = st.slider(
                        f"{numeric_filter_col} range",
                        min_value=col_min,
                        max_value=col_max,
                        value=(col_min, col_max),
                        key="table_num_range",
                    )
            fmt_mode = st.radio(
                "Table Type",
                ["Normal", "Inline Bars"],
                horizontal=False,
                key="table_fmt_mode",
            )

            # Column visibility control with select/unselect all toggle
            all_columns = df_base.columns.tolist()
            # Toggle controls default only; we avoid setting widget state programmatically
            select_all = st.checkbox(
                "Select/Unselect All Columns", value=True, key="table_select_all_toggle"
            )
            seed_default = all_columns if select_all else []

            selected_columns = st.multiselect(
                "Columns",
                options=all_columns,
                default=seed_default,
            )

        _tight_divider(sidebar_divider_gap_rem)

        # --- Show Chart + its options
        show_chart = st.checkbox("**Show Chart**", value=False)

        chart_type = "Histogram"
        chart_title = ""
        xcol = ""
        ycol = ""
        ycols = None
        hue_col = None
        size_col = None
        normalize_line = False
        smoothing_window = 1  # default
        area_normalize_100 = True

        # Initialize persistent Y selections in session state
        if "selected_ycol" not in st.session_state:
            st.session_state["selected_ycol"] = ""
        if "selected_ycols" not in st.session_state:
            st.session_state["selected_ycols"] = []

        if show_chart:
            chart_type = st.radio(
                "Chart Type",
                ["Histogram", "Line", "Area", "Scatter", "Box"],
                horizontal=True,
            )
            chart_title = st.text_input(
                "Chart Title",
                value="",
                key="chart_title",
                placeholder="Enter chart title (optional)",
            )
            if chart_type != "Histogram":
                xcol = st.selectbox("X-axis", [""] + df_base.columns.tolist(), index=0)

            if chart_type in ("Line", "Area"):
                # Multi-select persists via key; no default to avoid selection flicker
                ycols = st.multiselect("Y-axis", num_cols, key="ycols_line_area")
                # Persist selection and sync single to first of multi if available
                st.session_state["selected_ycols"] = ycols
                if ycols:
                    st.session_state["selected_ycol"] = ycols[0]
                smoothing_window = st.slider(
                    "Smoothing Window",
                    min_value=1,
                    max_value=200,
                    value=1,
                    step=1,
                    help="Rolling mean window; 1 = no smoothing",
                )
                if chart_type == "Line":
                    normalize_line = st.checkbox("*Normalize Data*", value=False)
                elif chart_type == "Area":
                    area_normalize_100 = st.checkbox("*Normalize to 100%*", value=True)
            elif chart_type == "Histogram":
                # Histogram: multi-select Y columns; persist via key only
                ycols = st.multiselect("Y-axis", num_cols, key="ycols_hist")
                normalize_counts = st.checkbox(
                    "*Normalize Counts*", value=False, key="normalize_counts_hist"
                )
                normalize_data_hist = st.checkbox(
                    "*Normalize Data*",
                    value=False,
                    key="normalize_data_hist",
                    help="Scale each selected series to [-1, 1] using per-series min–max (same behavior as Line).",
                )
                st.session_state["selected_ycols"] = ycols
                if ycols:
                    st.session_state["selected_ycol"] = ycols[0]
            else:
                # Scatter/Box: single Y select
                default_single = st.session_state["selected_ycol"]
                if not default_single and st.session_state["selected_ycols"]:
                    default_single = st.session_state["selected_ycols"][0]
                options = [""] + num_cols
                default_index = (
                    options.index(default_single) if default_single in options else 0
                )
                ycol = st.selectbox("Y-axis", options, index=default_index)
                st.session_state["selected_ycol"] = ycol

            if chart_type == "Scatter":
                hue_col = st.selectbox("Hue", [""] + df_base.columns.tolist(), index=0)
                size_col = st.selectbox("Size", [""] + num_cols, index=0)

        _tight_divider(sidebar_divider_gap_rem)

        # --- Correlation heatmap toggle
        show_corr = st.checkbox("**Show Correlation Heatmap**", value=False)

        heatmap_selected_cols = []
        if show_corr:
            heatmap_selected_cols = st.multiselect(
                "Heatmap Columns",
                options=num_cols,
                default=num_cols,
                key="heatmap_selected_cols",
            )

        _tight_divider(sidebar_divider_gap_rem)

        # --- Pivot controls
        show_pivot = st.checkbox("**Show Pivot Table**", value=False)
        pivot_rows = pivot_cols = pivot_val = agg = None
        quantile_rows = False
        quantile_cols = False

        if show_pivot:
            pivot_rows = st.selectbox("Pivot Rows", [""] + cat_cols + num_cols, index=0)
            if pivot_rows:
                quantile_rows = st.checkbox(
                    "*Transform to Quantiles*",
                    value=False,
                    key="q_rows",
                    help="Bin the selected pivot row into fixed quantile buckets at 1%, 25%, 50%, 75%, 99% (tails included).",
                )

            pivot_cols = st.selectbox(
                "Pivot Columns", [""] + cat_cols + num_cols, index=0
            )
            if pivot_cols:
                quantile_cols = st.checkbox(
                    "*Transform to Quantiles*",
                    value=False,
                    key="q_cols",
                    help="Bin the selected pivot column into fixed quantile buckets at 1%, 25%, 50%, 75%, 99% (tails included).",
                )

            pivot_val = st.selectbox("Pivot Value", num_cols)
            agg = st.selectbox(
                "Aggregation",
                ["min", "max", "sum", "mean", "std", "median", "iqr", "count"],
            )
            pivot_heatmap = st.checkbox(
                "Render Pivot as Heatmap", value=False, key="pivot_heatmap"
            )

        _tight_divider(sidebar_divider_gap_rem)

        # --- Custom column (MVP) ---
        enable_custom = st.checkbox(
            "**Create Custom Column**", value=False, key="enable_custom_col"
        )
        new_col_name = ""
        new_expr = ""
        if enable_custom:
            new_col_name = st.text_input(
                "New Column Name",
                placeholder="e.g. pct_change",
                key="custom_col_name",
                autocomplete="off",
            )
            new_expr = st.text_input(
                "Expression",
                placeholder="(close - open) / open * 100",
                key="custom_col_expr",
                autocomplete="off",
            )
            if new_col_name and new_expr:
                try:
                    test_series = pd.eval(
                        new_expr, engine="python", local_dict=df_base.to_dict("series")
                    )
                    df_base[new_col_name] = test_series
                    if "custom_cols" not in st.session_state:
                        st.session_state["custom_cols"] = []
                    if (new_col_name, new_expr) not in st.session_state["custom_cols"]:
                        st.session_state["custom_cols"].append((new_col_name, new_expr))
                    # Mark for one-shot success note just below the Expression field
                    st.session_state["_custom_added_name"] = new_col_name
                    # Keep the creator open so the success message is visible below the input
                    # The table below is updated in this same render via df_base mutation
                except Exception as e:
                    st.error(f"Could not create '{new_col_name}': {e}")

        # One-shot transient success note with soft fade directly under the Expression field
        _added_once = st.session_state.pop("_custom_added_name", None)
        if _added_once:
            st.markdown(
                (
                    '<div style="margin-top:6px; color:#FCE2EB; animation:fadeMsg 2.5s ease-in-out forwards;">'
                    f"Column {_added_once} added."
                    "</div>"
                    "<style>@keyframes fadeMsg {0%{opacity:0;} 15%{opacity:1;} 85%{opacity:1;} 100%{opacity:0;}}</style>"
                ),
                unsafe_allow_html=True,
            )

    # Return everything the main script needs
    return dict(
        show_table=show_table,
        numeric_filter_col=numeric_filter_col,
        num_range=num_range,
        fmt_mode=fmt_mode,
        selected_columns=selected_columns,
        show_chart=show_chart,
        chart_type=chart_type,
        chart_title=chart_title,
        xcol=xcol,
        ycol=ycol,
        ycols=ycols,
        hue_col=hue_col,
        size_col=size_col,
        normalize_line=normalize_line,
        smoothing_window=smoothing_window,
        area_normalize_100=area_normalize_100,
        show_corr=show_corr,
        heatmap_selected_cols=heatmap_selected_cols,
        normalize_counts_hist=locals().get("normalize_counts", False),
        normalize_data_hist=locals().get("normalize_data_hist", False),
        show_pivot=show_pivot,
        pivot_rows=pivot_rows,
        pivot_cols=pivot_cols,
        pivot_val=pivot_val,
        agg=agg,
        quantile_rows=quantile_rows,
        quantile_cols=quantile_cols,
        pivot_heatmap=locals().get("pivot_heatmap", False),
    )
