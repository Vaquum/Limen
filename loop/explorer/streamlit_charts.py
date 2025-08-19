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


def _apply_plot_theme(fig):
    fig.update_layout(font=dict(size=16))
    fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=15))
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=15))
    fig.update_layout(legend=dict(font=dict(size=15)))
    return fig

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

        fig = px.line(plot_long, x=xcol, y="minmax_scaled", color="Series")
        _apply_plot_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Wide form; sort then smooth each y
        plot_df = df_filt[[xcol] + ycols].copy().sort_values(xcol)
        if smoothing_window > 1:
            for col in ycols:
                plot_df[col] = plot_df[col].rolling(window=smoothing_window, min_periods=1).mean()

        fig = px.line(plot_df, x=xcol, y=ycols, **common_args)
        _apply_plot_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


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
    _apply_plot_theme(fig)
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
    fig = px.scatter(df_filt, x=xcol, y=ycol, **common_args)
    _apply_plot_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def plot_box(
    df_filt: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    hue_col: str | None = None,
) -> None:
    fig = px.box(df_filt, x=xcol, y=ycol, color=hue_col if hue_col else None)
    _apply_plot_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def plot_histogram(
    df_filt: pd.DataFrame,
    *,
    ycol: str | None = None,
    ycols: list[str] | None = None,
    normalize_data: bool = False,
    normalize_counts: bool = False,
    hue_col: str | None = None,
) -> None:
    args = {"color": hue_col} if hue_col else {}
    if ycols:
        # Build a long-form DataFrame to overlay multiple series in one histogram
        # Ensure consistent order and avoid duplicate keys on rapid changes
        cols = list(dict.fromkeys(ycols))
        long_df = df_filt[cols].copy().melt(value_vars=cols, var_name="Series", value_name="Value")
        if normalize_data:
            # Per-series min-max scaling to [-1, 1] like Line normalize
            def minmax_scale(s: pd.Series) -> pd.Series:
                s = pd.to_numeric(s, errors='coerce')
                vmin, vmax = np.nanmin(s.values), np.nanmax(s.values)
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
                    return pd.Series(np.zeros(len(s)), index=s.index)
                scaled01 = (s - vmin) / (vmax - vmin)
                return -1.0 + 2.0 * scaled01

            long_df['Value'] = (
                long_df.groupby('Series')['Value']
                .transform(minmax_scale)
            )
        # Default overlay (native numeric bins)
        fig = px.histogram(long_df, x='Value', color='Series', opacity=0.6, histnorm='probability' if normalize_counts else None)
        fig.update_layout(barmode='overlay')
        _apply_plot_theme(fig)
        key = f"hist_multi_{'_'.join(cols)}_{int(normalize_counts)}_{int(normalize_data)}"
        st.plotly_chart(fig, use_container_width=True, key=key)
        return
    if not ycol:
        return
    if normalize_data:
        s = pd.to_numeric(df_filt[ycol], errors='coerce')
        vmin, vmax = np.nanmin(s.values), np.nanmax(s.values)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax != vmin:
            s = -1.0 + 2.0 * ((s - vmin) / (vmax - vmin))
        fig = px.histogram(x=s, histnorm='probability' if normalize_counts else None, **args)
    else:
        fig = px.histogram(df_filt, x=ycol, histnorm='probability' if normalize_counts else None, **args)
    _apply_plot_theme(fig)
    st.plotly_chart(fig, use_container_width=True, key=f"hist_single_{ycol}_{int(normalize_counts)}_{int(normalize_data)}")