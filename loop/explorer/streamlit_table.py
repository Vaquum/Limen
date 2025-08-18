# streamlit_table.py

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

def prepare_table_data(
    df_base: pd.DataFrame,
    show_table: bool,
    numeric_filter_col: str | None,
    num_range: tuple[float, float] | None,
    filter_outliers: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the numeric range filter (if any) and inserts the drill-down link column.
    Returns (df_filt, df_display).
    """
    df_filt = df_base
    if filter_outliers:
        num_cols = df_filt.select_dtypes("number").columns.tolist()
        if num_cols:
            mask = pd.Series(True, index=df_filt.index)
            for c in num_cols:
                series = pd.to_numeric(df_filt[c], errors="coerce")
                q = series.quantile([0.25, 0.75])
                q1, q3 = float(q.iloc[0]), float(q.iloc[1])
                iqr = q3 - q1
                if not np.isfinite(iqr) or iqr == 0:
                    continue
                lo = q1 - 1.5 * iqr
                hi = q3 + 1.5 * iqr
                mask &= series.between(lo, hi) | series.isna()
            df_filt = df_filt[mask]
    if show_table and numeric_filter_col and num_range:
        lo, hi = num_range
        df_filt = df_filt[(df_filt[numeric_filter_col] >= lo) & (df_filt[numeric_filter_col] <= hi)]

    # Add drill-down link
    df_view = df_filt.copy()
    df_view.insert(1, "view", [f"/?row={i}" for i in df_view["rid"]])

    return df_filt, df_view


def render_table(
    df_display: pd.DataFrame,
    fmt_mode: str,
) -> None:
    """
    Renders the table in either "Inline Bars" or "Normal" mode.
    """
    numeric_visible = df_display.select_dtypes("number").columns.tolist()

    if fmt_mode == "Inline Bars":
        colcfg = {
            "view": st.column_config.LinkColumn("", help="Open row details", display_text="🔎 view"),
            "rid": st.column_config.NumberColumn("rid", help="Row id", width="small"),
        }
        for c in numeric_visible:
            cmin = float(df_display[c].min())
            cmax = float(df_display[c].max())
            if cmin == cmax:
                cmin -= 1.0
                cmax += 1.0
            colcfg[c] = st.column_config.ProgressColumn(
                c,
                help=f"{c} (min={cmin:.3g}, max={cmax:.3g})",
                min_value=cmin,
                max_value=cmax,
                format="%.4g",
            )
        st.dataframe(df_display, use_container_width=True, hide_index=True, column_config=colcfg)
    else:
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "view": st.column_config.LinkColumn("", help="Open row details", display_text="🔎 view"),
                "rid": st.column_config.NumberColumn("rid", help="Row id", width="small"),
            },
        )