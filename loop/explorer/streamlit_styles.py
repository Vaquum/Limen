# IMPORTANT! -> Leave double quotes in this file exceptionally! 


def streamlit_styles(sidebar_container_gap_rem: float = 0.45,
                     sidebar_divider_gap_rem: float = 0.30) -> str:
    
    '''
    Compute CSS string for global Streamlit styles.
    Args:
        sidebar_container_gap_rem (float): Vertical spacing at the top of sidebar content
        sidebar_divider_gap_rem (float): Vertical spacing for sidebar dividers
    Returns:
        str: CSS string for use with st.markdown(..., unsafe_allow_html=True)
    '''
    
    return f"""
    <style>
      /* Palette variables */
      :root {{
        --palette-1: #C4E8F4; /* light blue */
        --palette-2: #FCE2EB; /* blush */
        --palette-3: #EAA3C8; /* pink */
        --palette-4: #DC65A6; /* magenta accent */
        --palette-5: #F16068; /* coral */
        --palette-6: #BCABD3; /* lavender */
        --palette-7: #DDD941; /* yellow */
      }}
      /* Import fonts: IBM Plex for text, Lexend for numerals */
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=Lexend:wght@400;600;700&display=swap');

      /* Global font scaling (~20%) */
      html, body, [data-testid="stAppViewContainer"] {{
          font-size: 19px;
          font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, sans-serif;
      }}

      /* Sidebar spacing */
      [data-testid="stSidebar"] .stElementContainer {{
          margin-top: {sidebar_container_gap_rem}rem !important;
          margin-bottom: 0 !important;
          padding-top: 0 !important;
          padding-bottom: 0 !important;
      }}
      /* Make sidebar ~20% wider than default (21rem → 25.2rem) */
      [data-testid="stSidebar"] {{
          width: 16rem !important;
          min-width: 16rem !important;
      }}
      [data-testid="stSidebar"] hr {{
          margin: {sidebar_divider_gap_rem}rem 0 !important;
          border: 0;
          border-top: 1px solid rgba(0,0,0,.12);
      }}

      /* Ensure toolbar buttons can hide text cleanly when icon is used */
      [data-testid="stSidebar"] [data-testid="stButton"] > button {{
          line-height: 0 !important;
      }}

      /* Detail cards */
      .lux-card {{
        background: #2F2D36;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        margin-bottom: 10px;
      }}
      .lux-label {{
        font-size: 16px;
        font-weight: 600;
        letter-spacing: .01em;
        color: #5f6b7a;
        margin-bottom: 8px;
        text-transform: none;
      }}
      .lux-value {{
        font-size: 26px;
        font-weight: 700;
        color: #ECE8F2;
        word-break: break-word;
        line-height: 1.25;
        /* Numerals use Lexend with tabular figures for alignment */
        font-family: "Lexend", "IBM Plex Sans", Arial, sans-serif;
        font-variant-numeric: tabular-nums lining-nums;
        font-feature-settings: "tnum" 1, "lnum" 1;
      }}
      .lux-back a {{
        color: var(--palette-4); font-weight: 600; text-decoration: none;
      }}
      .lux-back a:hover {{ text-decoration: underline; }}
      /* Match table 'view' links to theme purple */
      [data-testid="stDataFrame"] a[href^='/?row='] {{
        color: var(--palette-4) !important;
        text-decoration: none;
        font-weight: 600;
      }}
      [data-testid="stDataFrame"] a[href^='/?row=']:hover {{
        text-decoration: underline;
      }}

      /* Enlarge table text (~20% above global) */
      [data-testid="stDataFrame"] * {{
        font-size: 1.2em !important;
        /* Prefer Lexend for numeric-heavy tables */
        font-family: "Lexend", "IBM Plex Sans", Arial, sans-serif !important;
        font-variant-numeric: tabular-nums lining-nums;
        font-feature-settings: "tnum" 1, "lnum" 1;
      }}
      /* Keep built-in column visibility menu visible (avoid Popper warnings) */
      /* Inline Bars (ProgressColumn) color override */
      [data-testid="stDataFrame"] [role="progressbar"] > div {{
        background-color: var(--palette-6) !important;
      }}
      [data-testid="stDataFrame"] [role="progressbar"] {{
        border-radius: 6px !important;
        overflow: hidden;
      }}
      .lux-title {{
        font-size: 46px; font-weight: 800; margin: 0 0 6px 0;
        letter-spacing: -0.02em;
      }}
      .lux-subtle {{ color: #778295; }}

      /* Plotly charts: use Lexend for tick and hover numerals */
      .js-plotly-plot .xtick text,
      .js-plotly-plot .ytick text,
      .js-plotly-plot .hovertext text {{
        font-family: "Lexend", "IBM Plex Sans", Arial, sans-serif !important;
        font-variant-numeric: tabular-nums lining-nums;
        font-feature-settings: "tnum" 1, "lnum" 1;
      }}

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
