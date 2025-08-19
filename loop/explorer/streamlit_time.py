from __future__ import annotations
from datetime import timedelta
import pandas as pd
import streamlit as st


def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    # Only accept a column literally named 'datetime'
    return ['datetime'] if 'datetime' in df.columns else []


def render_time_controls(df: pd.DataFrame) -> dict:
    """Render time controls if a datetime column exists. Returns settings dict."""
    dt_cols = _detect_datetime_columns(df)
    if not st.session_state.get("_show_time", False):
        return {"enabled": False}
    if not dt_cols:
        with st.sidebar:
            st.info("No datetime column detected.")
        return {"enabled": False}

    with st.sidebar:
        raw = df['datetime'].dropna()
        # Parse to UTC-aware then drop tz for slider bounds
        s = pd.to_datetime(raw, utc=True).dropna()
        start = s.min().tz_convert('UTC').tz_localize(None).to_pydatetime()
        end = s.max().tz_convert('UTC').tz_localize(None).to_pydatetime()

        # Presets
        preset = st.radio('Time Window', ['All', 'YTD', '30D', '7D', '24H', '48H', 'Custom'], horizontal=True)
        if preset != 'All':
            if preset == 'YTD':
                bound_lo = end.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            elif preset == '30D':
                bound_lo = end - timedelta(days=30)
            elif preset == '7D':
                bound_lo = end - timedelta(days=7)
            elif preset == '24H':
                bound_lo = end - timedelta(hours=24)
            elif preset == '48H':
                bound_lo = end - timedelta(hours=48)
            else:
                bound_lo = start
        else:
            bound_lo = start

        # Range selector (always shown so user can fine-tune)
        sel_start, sel_end = st.slider('Time Range', min_value=start, max_value=end, value=(bound_lo, end))

    return {
        'enabled': True,
        'start': sel_start,
        'end': sel_end,
    }


def apply_time_filter(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if not settings.get('enabled'):
        return df
    # Parse entire column uniformly to UTC-naive for masking
    parsed = pd.to_datetime(df['datetime'], utc=True)
    s2 = parsed.dt.tz_convert('UTC').dt.tz_localize(None)
    mask = (s2 >= settings['start']) & (s2 <= settings['end'])
    return df.loc[mask]


