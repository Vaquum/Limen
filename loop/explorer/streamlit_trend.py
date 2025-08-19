from __future__ import annotations
import pandas as pd
import polars as pl
import streamlit as st

from loop.features.ma_slope_regime import ma_slope_regime
from loop.features.price_vs_band_regime import price_vs_band_regime
from loop.features.breakout_percentile_regime import breakout_percentile_regime
from loop.features.window_return_regime import window_return_regime
from loop.features.hh_hl_structure_regime import hh_hl_structure_regime


def render_trend_controls(df: pd.DataFrame) -> dict:
    
    '''
    Render trend regime control panel and return selected settings.
    
    Args:
        df (pd.DataFrame): Klines dataset with 'datetime' and price columns
    
    Returns:
        dict: Trend configuration settings for downstream application
    '''
    
    if 'datetime' not in df.columns or not st.session_state.get('_show_time', False):
        # Only enable when datetime exists; tie visibility to toolbar use-case if needed
        pass

    if '_show_trend' not in st.session_state:
        st.session_state['_show_trend'] = False

    # Trend icon lives with the toolbar; toggled externally. Here we just render controls if active.
    if not st.session_state.get('_show_trend', False):
        return {'enabled': False}

    with st.sidebar:
        method = st.selectbox('Trend Method', [
            'None', 'MA Slope', 'Price vs Band', 'Breakout Percentile', 'Window Return', 'HH/HL Structure'
        ], index=0)

        if method == 'None':
            return {'enabled': False}

        period = st.number_input('Period', min_value=2, max_value=1000, value=24, step=1)
        up_only = st.checkbox('Up-Only', value=True)

        params: dict = {}
        if method == 'MA Slope':
            params['threshold'] = st.number_input('Slope Threshold (normalized)', value=0.0, step=0.1, format='%0.1f')
            params['normalize_by_std'] = st.checkbox('Normalize by STD', value=True)
        elif method == 'Price vs Band':
            params['band'] = st.selectbox('Band Type', ['std', 'dev_std'], index=0)
            params['k'] = st.number_input('k (band multiplier)', value=0.75, step=0.05, format='%0.2f')
        elif method == 'Breakout Percentile':
            params['p_hi'] = st.slider('Upper Percentile', 0.5, 0.99, 0.85, 0.01, format='%0.2f')
            params['p_lo'] = st.slider('Lower Percentile', 0.01, 0.5, 0.15, 0.01, format='%0.2f')
        elif method == 'Window Return':
            params['r_hi'] = st.number_input('Upper Return Threshold', value=0.0, step=0.001, format='%0.3f')
            params['r_lo'] = st.number_input('Lower Return Threshold', value=0.0, step=0.001, format='%0.3f')
        elif method == 'HH/HL Structure':
            params['score_threshold'] = st.number_input('Score Threshold', value=4, step=1)

    return {
        'enabled': True,
        'method': method,
        'period': int(period),
        'up_only': up_only,
        **params,
    }


def apply_trend_regime(df: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    '''
    Compute trend regime according to settings and return annotated and filtered DataFrames.
    
    Args:
        df (pd.DataFrame): Klines dataset with price columns used by regime features
        settings (dict): Trend configuration produced by render_trend_controls
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrame with 'regime' and filtered DataFrame by regime
    '''
    
    if not settings.get('enabled'):
        return df, df

    pl_df = pl.from_pandas(df)
    method = settings['method']
    period = settings['period']

    if method == 'MA Slope':
        pl_out = ma_slope_regime(pl_df, period=period, threshold=float(settings.get('threshold', 0.0)), normalize_by_std=bool(settings.get('normalize_by_std', True)))
        regime_col = 'regime_ma_slope'
    elif method == 'Price vs Band':
        pl_out = price_vs_band_regime(pl_df, period=period, band=str(settings.get('band', 'std')), k=float(settings.get('k', 0.75)))
        regime_col = 'regime_price_band'
    elif method == 'Breakout Percentile':
        pl_out = breakout_percentile_regime(pl_df, period=period, p_hi=float(settings.get('p_hi', 0.85)), p_lo=float(settings.get('p_lo', 0.15)))
        regime_col = 'regime_breakout_pct'
    elif method == 'Window Return':
        pl_out = window_return_regime(pl_df, period=period, r_hi=float(settings.get('r_hi', 0.0)), r_lo=float(settings.get('r_lo', 0.0)))
        regime_col = 'regime_window_return'
    else:
        pl_out = hh_hl_structure_regime(pl_df, window=period, score_threshold=int(settings.get('score_threshold', 4)))
        regime_col = 'regime_hh_hl'

    out_df = pl_out.with_columns(pl.col(regime_col).alias('regime')).to_pandas()
    
    if settings.get('up_only', True):
        filt_df = out_df.loc[out_df['regime'] == 'Up']
    else:
        filt_df = out_df
    return out_df, filt_df


