# streamlit_charts.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Internal helpers (module-local)
# -----------------------------
def _rolling_mean_per_series(df_long: pd.DataFrame, series_col: str, value_col: str, window: int) -> pd.Series:
    """Apply rolling mean per series; window=1 -> passthrough."""
    if window <= 1:
        return df_long[value_col]
    return (
        df_long.groupby(series_col)[value_col]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )


# -----------------------------
# Public chart renderers
# -----------------------------
def plot_line(
    df_filt: pd.DataFrame,
    *,
    xcol: str,
    ycols: list[str],
    smoothing_window: int = 1,
    normalize_line: bool = False,
    hue_col: str | None = None,
    size_col: str | None = None,
) -> None:
    """
    Line chart.
    - If normalize_line=True: per-series min-max scaling to [-1, 1] in long form,
      smoothing applied to the scaled series.
    - Else: wide form, smoothing per column before plotting.
    """
    common_args = {}
    if hue_col:
        common_args["color"] = hue_col
    if size_col:
        common_args["size"] = size_col

    if normalize_line:
        # Long form
        plot_long = df_filt[[xcol] + ycols].melt(
            id_vars=[xcol],
            value_vars=ycols,
            var_name="Series",
            value_name="Value"
        )
        plot_long["Value"] = pd.to_numeric(plot_long["Value"], errors="coerce")

        # Per-series min/max
        stats = plot_long.groupby("Series")["Value"].agg(vmin="min", vmax="max").reset_index()
        plot_long = plot_long.merge(stats, on="Series", how="left")

        # Scale to [-1, 1]; constants -> 0
        denom = (plot_long["vmax"] - plot_long["vmin"]).to_numpy()
        num   = (plot_long["Value"] - plot_long["vmin"]).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled01 = np.where(denom == 0, 0.5, num / denom)
            plot_long["minmax_scaled"] = -1.0 + 2.0 * scaled01

        # Sort + optional smoothing on the scaled series
        plot_long = plot_long.sort_values(["Series", xcol])
        plot_long["minmax_scaled"] = _rolling_mean_per_series(
            plot_long, "Series", "minmax_scaled", smoothing_window
        )

        st.plotly_chart(px.line(plot_long, x=xcol, y="minmax_scaled", color="Series"), use_container_width=True)

    else:
        # Wide form; sort then smooth each y
        plot_df = df_filt[[xcol] + ycols].copy().sort_values(xcol)
        if smoothing_window > 1:
            for col in ycols:
                plot_df[col] = plot_df[col].rolling(window=smoothing_window, min_periods=1).mean()

        st.plotly_chart(px.line(plot_df, x=xcol, y=ycols, **common_args), use_container_width=True)


def plot_area(
    df_filt: pd.DataFrame,
    *,
    xcol: str,
    ycols: list[str],
    smoothing_window: int = 1,
    normalize_100: bool = True,
) -> None:
    """
    100% stacked area chart (aka normalized stacked area):
    - Uses groupnorm="fraction" to normalize each x-slice.
    - Formats y-axis as percent.
    - Keeps legend order stable by sorting ycols.
    """
    ycols_sorted = sorted(ycols)

    # Long form
    plot_long = df_filt[[xcol] + ycols_sorted].melt(
        id_vars=[xcol],
        value_vars=ycols_sorted,
        var_name="Series",
        value_name="Value"
    )
    plot_long["Value"] = pd.to_numeric(plot_long["Value"], errors="coerce")

    # Sort + optional smoothing
    plot_long = plot_long.sort_values(["Series", xcol])
    plot_long["Value"] = _rolling_mean_per_series(plot_long, "Series", "Value", smoothing_window)

    if normalize_100:
        fig = px.area(
            plot_long,
            x=xcol, y="Value", color="Series",
            groupnorm="fraction",
            category_orders={"Series": ycols_sorted},
        )
        fig.update_layout(yaxis=dict(tickformat=".0%"))
    else:
        fig = px.area(
            plot_long,
            x=xcol, y="Value", color="Series",
            category_orders={"Series": ycols_sorted},
        )
    st.plotly_chart(fig, use_container_width=True)


def plot_scatter(
    df_filt: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    hue_col: str | None = None,
    size_col: str | None = None,
) -> None:
    common_args = {}
    if hue_col:
        common_args["color"] = hue_col
    if size_col:
        common_args["size"] = size_col
    st.plotly_chart(px.scatter(df_filt, x=xcol, y=ycol, **common_args), use_container_width=True)


def plot_box(
    df_filt: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    hue_col: str | None = None,
) -> None:
    st.plotly_chart(px.box(df_filt, x=xcol, y=ycol, color=hue_col if hue_col else None), use_container_width=True)


def plot_histogram(
    df_filt: pd.DataFrame,
    *,
    ycol: str,
    hue_col: str | None = None,
) -> None:
    args = {"color": hue_col} if hue_col else {}
    st.plotly_chart(px.histogram(df_filt, x=ycol, **args), use_container_width=True)