# streamlit_heatmap.py
from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st

def render_corr_heatmap(df_filt: pd.DataFrame, num_cols: list[str]) -> None:
    
    '''
    Compute and render a correlation heatmap for selected numeric columns.
    
    Args:
        df_filt (pd.DataFrame): Klines dataset with numeric columns to correlate
        num_cols (list[str]): User-selected column names to include in the correlation matrix
    
    Returns:
        None: None
    '''
    
    if len(num_cols) < 2:
        st.info("Select at least 2 columns for the correlation heatmap.")
        return

    corr = df_filt[num_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=False,
        # Custom palette (low → high)
        color_continuous_scale=[
            "#C4E8F4",  # palette-1
            "#FCE2EB",  # palette-2
            "#EAA3C8",  # palette-3
            "#DC65A6",  # palette-4
            "#F16068",  # palette-5
            "#BCABD3",  # palette-6
            "#DDD941",  # palette-7
        ],
        origin="lower",
        aspect="auto",
    )
    # Match table text scale (~20% above global) -> use ~1.2em equivalent
    # Semi-bold effect via HTML <b> and larger font size
    fig_corr.update_traces(
        text=corr.round(2).astype(str).values,
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=16, family='Lexend, "IBM Plex Sans", Arial, sans-serif'),
        opacity=0.85,
        selector=dict(type="heatmap"),
    )
    fig_corr.update_xaxes(side="top")
    fig_corr.update_layout(
        height=800,
        margin=dict(l=60, r=40, t=60, b=60),
        yaxis=dict(tickfont=dict(size=16), tickangle=0),
        xaxis=dict(tickfont=dict(size=16)),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        shapes=[
            dict(
                type="rect",
                x0=i - 0.5, x1=i + 0.5,
                y0=j - 0.5, y1=j + 0.5,
                line=dict(color="grey", width=0.3),
            )
            for i in range(len(corr.columns))
            for j in range(len(corr.columns))
        ],
    )
    st.plotly_chart(fig_corr, use_container_width=True)