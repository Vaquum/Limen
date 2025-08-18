# streamlit_styles.py

def streamlit_styles(sidebar_container_gap_rem: float = 0.45,
                     sidebar_divider_gap_rem: float = 0.30) -> str:
    """
    Return the global CSS for the app. Call with:
        st.markdown(streamlit_styles(sidebar_container_gap_rem, sidebar_divider_gap_rem),
                    unsafe_allow_html=True)
    """
    return f"""
    <style>
      /* Sidebar spacing */
      [data-testid="stSidebar"] .stElementContainer {{
          margin-top: {sidebar_container_gap_rem}rem !important;
          margin-bottom: 0 !important;
          padding-top: 0 !important;
          padding-bottom: 0 !important;
      }}
      [data-testid="stSidebar"] hr {{
          margin: {sidebar_divider_gap_rem}rem 0 !important;
          border: 0;
          border-top: 1px solid rgba(0,0,0,.12);
      }}

      /* Detail cards */
      .lux-card {{
        background: #fff;
        border: 1px solid rgba(30,40,60,0.10);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 6px 24px rgba(31,36,48,0.06);
      }}
      .lux-label {{
        font-size: 13px;
        font-weight: 600;
        letter-spacing: .01em;
        color: #5f6b7a;
        margin-bottom: 8px;
        text-transform: none;
      }}
      .lux-value {{
        font-size: 22px;
        font-weight: 700;
        color: #1F2430;
        word-break: break-word;
        line-height: 1.25;
      }}
      .lux-back a {{
        color: #6D5AF7; font-weight: 600; text-decoration: none;
      }}
      .lux-back a:hover {{ text-decoration: underline; }}
      .lux-title {{
        font-size: 38px; font-weight: 800; margin: 0 0 6px 0;
        letter-spacing: -0.02em;
      }}
      .lux-subtle {{ color: #778295; }}

      /* Split card layout for detail view */
      .split-card {{
        display: flex;
        gap: 12px;
        position: relative;
        align-items: flex-start;
      }}
      .split-card::before {{
        content: "";
        position: absolute;
        top: 10px;
        bottom: 10px;
        left: 50%;
        width: 1px;
        background: linear-gradient(
          180deg,
          rgba(0,0,0,0) 0%,
          rgba(0,0,0,0.12) 30%,
          rgba(0,0,0,0.12) 70%,
          rgba(0,0,0,0) 100%
        );
      }}
      .split-left  {{ flex: 1 1 50%; min-width: 0; }}
      .split-right {{ flex: 1 1 50%; min-width: 0; }}
    </style>
    """
