# streamlit_heatmap.py
from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st

def render_corr_heatmap(df_filt: pd.DataFrame, num_cols: list[str]) -> None:
    """Render the correlation heatmap exactly like in the main file (no logic changes)."""
    if len(num_cols) < 2:
        st.info("Not enough numeric columns for correlation heatmap.")
        return

    corr = df_filt[num_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        origin="lower",
        aspect="auto",
    )
    # Match table text scale (~20% above global) -> use ~1.2em equivalent
    fig_corr.update_traces(textfont=dict(size=16), selector=dict(type="heatmap"))
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