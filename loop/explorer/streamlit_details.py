# streamlit_details.py
from __future__ import annotations
import pandas as pd
import streamlit as st

def render_details_view(
    df: pd.DataFrame,
    pretty_label_fn,
    format_value_fn,
) -> None:
    
    '''
    Render the detail cards when the URL contains the row query parameter.
    
    Args:
        df (pd.DataFrame): Klines dataset with selected row accessible by 'rid'
        pretty_label_fn (callable): Function to convert column keys to labels
        format_value_fn (callable): Function to format values for display
    
    Returns:
        None: None
    '''
    
    params = st.query_params
    if 'row' not in params:
        return

    try:
        raw = params['row']
        sval = raw[0] if isinstance(raw, list) else raw
        rid = int(str(sval))
        row = df.iloc[rid]
    except Exception:
        st.error('Invalid row id.')
        st.stop()

    st.markdown(f"<div class='lux-title'>Row {rid} details</div>", unsafe_allow_html=True)
    # Same-tab back: clear the 'row' query param and rerun
    if st.button('← Back to table'):
        # Build params without 'row'
        try:
            qp = dict(st.query_params)
        except Exception:
            qp = {}
        qp.pop('row', None)
        # Prefer overwriting the full query param set to ensure removal
        try:
            # New API path: clear then repopulate
            try:
                st.query_params.clear()
                for k, v in qp.items():
                    st.query_params[k] = v
            except Exception:
                st.experimental_set_query_params(**qp)
        except Exception:
            pass
        # Ensure Show Table remains selected when returning
        st.session_state['table_show'] = True
        st.rerun()
    # ---------- Subtitle: date range, bar interval, sample size ----------
    def _format_timeresolution(td: pd.Timedelta | None) -> str:
        if td is None or pd.isna(td):
            return "N/A"
        total_seconds = int(td.total_seconds())
        if total_seconds <= 0:
            return "N/A"
        if total_seconds % 86400 == 0:
            days = total_seconds // 86400
            return f"{days}d"
        if total_seconds % 3600 == 0:
            hours = total_seconds // 3600
            return f"{hours}h"
        if total_seconds % 60 == 0:
            minutes = total_seconds // 60
            return f"{minutes}m"
        return f"{total_seconds}s"

    subtitle_parts: list[str] = ["All values formatted for readability"]

    # Date range and inferred bar interval from 'datetime' column if present
    if 'datetime' in df.columns:
        try:
            dt_series = pd.to_datetime(df['datetime'], utc=True, errors='coerce').dropna()
            if not dt_series.empty:
                dt_min = dt_series.min().tz_convert('UTC').tz_localize(None)
                dt_max = dt_series.max().tz_convert('UTC').tz_localize(None)
                subtitle_parts.append(f"Range: {dt_min:%Y-%m-%d} → {dt_max:%Y-%m-%d}")
                # Use median difference as robust bar interval
                diffs = dt_series.sort_values().diff().dropna()
                bar_td = diffs.median() if not diffs.empty else None
                subtitle_parts.append(f"Bar: {_format_timeresolution(bar_td)}")
        except Exception:
            pass

    # Sample size (rows in df)
    try:
        subtitle_parts.append(f"Sample: {len(df):,}")
    except Exception:
        pass

    subtitle_html = " \u00B7 ".join(subtitle_parts)
    st.markdown(f"<div class='lux-subtle' style='margin:6px 0 14px;'>{subtitle_html}</div>", unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (k, v) in enumerate(row.items()):
        dtype = df.dtypes[k]
        label = pretty_label_fn(k)
        val   = format_value_fn(v, dtype)
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="lux-card">
                  <div class="split-card">
                    <div class="split-left">
                      <div class="lux-label">{label}</div>
                      <div class="lux-value">{val}</div>
                    </div>
                    <div class="split-right">
                      <!-- reserved -->
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.stop()