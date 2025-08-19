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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    '''
    Compute filtered table data and add a drill-down link column.
    
    Args:
        df_base (pd.DataFrame): Klines dataset with 'rid' column for drill-down
        show_table (bool): Whether to apply table-specific filters
        numeric_filter_col (str | None): Numeric column to filter by range
        num_range (tuple[float, float] | None): Inclusive range for the numeric filter
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The filtered DataFrame and the display DataFrame
    '''
    
    df_filt = df_base
    if show_table and numeric_filter_col and num_range:
        lo, hi = num_range
        df_filt = df_filt[(df_filt[numeric_filter_col] >= lo) & (df_filt[numeric_filter_col] <= hi)]

    # Add drill-down link
    df_view = df_filt.copy()
    df_view.insert(1, 'view', [f"/?row={i}" for i in df_view["rid"]])

    # Pretty-print datetime to second resolution for display only
    if 'datetime' in df_view.columns:
        try:
            s = pd.to_datetime(df_view['datetime'], utc=True, errors='coerce')
            # Render as naive UTC to seconds
            df_view['datetime'] = s.dt.tz_convert('UTC').dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            # If conversion fails, leave original values
            pass

    return df_filt, df_view


def render_table(
    df_display: pd.DataFrame,
    fmt_mode: str,
) -> None:
    
    '''
    Render the table in either inline bar or normal mode.
    
    Args:
        df_display (pd.DataFrame): Klines dataset with a 'rid' column for drill-down
        fmt_mode (str): Display mode selector ('Inline Bars' or 'Normal')
    
    Returns:
        None: None
    '''
    
    numeric_visible = df_display.select_dtypes('number').columns.tolist()

    if fmt_mode == 'Inline Bars':
        colcfg = {
            'view': st.column_config.LinkColumn('', help='Open row details', display_text='view'),
            'rid': st.column_config.NumberColumn('rid', help='Row id', width='small'),
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
                format='%.4g',
            )
        st.dataframe(df_display, use_container_width=True, hide_index=True, column_config=colcfg)
    else:
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                'view': st.column_config.LinkColumn('', help='Open row details', display_text='view'),
                'rid': st.column_config.NumberColumn('rid', help='Row id', width='small'),
            },
        )