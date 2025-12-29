from __future__ import annotations
import streamlit as st


def _icon_button(col, glyph: str, tooltip: str, toggle_flag: str, idx: int) -> None:
    """
    Render a square icon button in the sidebar that toggles a session flag.

    Args:
        col: Streamlit column to render into
        glyph (str): Button label glyph
        tooltip (str): Unused tooltip label (reserved)
        toggle_flag (str): Session state key to toggle
        idx (int): Column index for CSS targeting

    Returns:
        None: None
    """

    def _toggle():
        st.session_state[toggle_flag] = not st.session_state.get(toggle_flag, False)

    # Use a simple glyph/emoji label directly
    col.button(glyph, on_click=_toggle, key=f"tb_{toggle_flag}")

    # Style the nth column's first button inside the current horizontal block of the sidebar.
    base_selector = (
        f"[data-testid='stSidebar'] "
        f"[data-testid='stHorizontalBlock'] "
        f"[data-testid='column']:nth-of-type({idx}) "
        f"button"
    )
    # Dimensions and base look
    col.markdown(
        f"""
        <style>
        {base_selector} {{
            width: 56px !important; height: 56px !important;
            min-width: 56px !important; min-height: 56px !important;
            aspect-ratio: 1 / 1 !important;
            border-radius: 12px !important;
            padding: 0 !important;
            display: inline-flex !important; align-items: center; justify-content: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Typography for the glyph label
    col.markdown(
        f"""
        <style>
        {base_selector} {{
            font-size: 28px !important;
            color: #7B61FF !important; /* primary accent purple */
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_toolbar() -> None:
    """
    Render the top toolbar with icons for Outliers, Time, Trend, and Dataset.

    Returns:
        None: None
    """

    # Initialize flags if missing
    st.session_state.setdefault("_show_outliers", False)
    st.session_state.setdefault("_show_time", False)
    st.session_state.setdefault("_show_trend", False)
    st.session_state.setdefault("_show_dataset", False)

    c1, c2, c3, c4 = st.sidebar.columns(4)
    _icon_button(c1, "◆", "Outliers", "_show_outliers", 1)
    _icon_button(c2, "◷", "Time", "_show_time", 2)
    _icon_button(c3, "▲", "Trend", "_show_trend", 3)
    _icon_button(c4, "■", "Dataset", "_show_dataset", 4)
