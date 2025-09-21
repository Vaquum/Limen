### Testing the Streamlit Explorer UI (Loop)

This guide captures a practical, repeatable method for validating the `loop.explorer` Streamlit UI end to end using our MCP tools.

### Scope
- Focus: `loop/explorer/streamlit_app.py` and flows reachable via `dev_snippets.test_explorer_locally()`
- Out of scope: model training internals and non‑explorer submodules

### Philosophy
- **Radical simplicity**: Fewer moving parts, fewer flakes
- **Determinism first**: Stable selectors, fixed waits, and explicit teardown
- **Actionable evidence**: Every assertion is backed by a screenshot or a visible UI state

### Environment (mandatory)
- Python: `python3.10`
- Virtual env: `.venv`
- Launch the app from an interactive Python cell or the shell, always exporting `PYTHONPATH` to the repo root

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
python -c 'import dev_snippets; addr = dev_snippets.test_explorer_locally(); print(addr)'
# If you need manual launch instead:
python -m streamlit run loop/explorer/streamlit_app.py --server.address 0.0.0.0 --server.port 5600 --server.headless true -- --data /tmp/historical_data.parquet
```

NOTE: `dev_snippets.test_explorer_locally()` produces parquet(s) in `/tmp/` and returns the host/port. Replace `0.0.0.0` with `localhost` for browser access.

### Boot rules (to avoid flakes)
- After first open, wait at least 15 seconds before interacting
- First readiness check: wait for `header[data-testid='stHeader']`
- Treat the app as ready only after both: page opened AND header selector present

### Tools and stable selectors
- Use the Playwright Driver tools:
  - `open(url)`
  - `wait_for_selector(selector, timeout)`
  - `click(selector)`
  - `type(selector, text)`
  - `screenshot(fullPage)`
- Prefer these UI anchors:
  - `header[data-testid='stHeader']` – app readiness
  - `div[data-testid='stSidebar']` – sidebar scope
  - `div[data-testid='stDataFrame']` – table rendered

Selector tips:
- Dropdown options can conflict with table header text (e.g., 'open', 'close'). Scoping clicks near the control or scrolling to the control first improves reliability
- Radio/checkbox labels are reliable: `text=Show Table`, `text=Show Chart`, `text=Show Correlation Heatmap`, `text=Show Pivot Table`, `text=Create Custom Column`

### Canonical smoke flow (sanity)
1) Launch via dev_snippets and derive `http://localhost:{port}`
2) `open(url)` and `wait_for_selector('header[data-testid=\'stHeader\']')`
3) Enable `Show Table`; assert `div[data-testid='stDataFrame']` visible; capture screenshot
4) Toggle `Show Chart` (default Histogram):
   - Select at least one Y series; toggle normalize counts/data; screenshot
5) Switch chart types:
   - Line: set `X= datetime`, `Y= close`, set smoothing; screenshot
   - Area: set multiple `Y` (e.g., open, high, low, close), enable 100% normalize; screenshot
   - Scatter: set `X= open`, `Y= close`, optional `Hue`/`Size`; screenshot
   - Box: set `X= datetime (or binned)`, `Y= close`; screenshot
6) Toggle `Show Correlation Heatmap`; scroll to confirm render; screenshot
7) Toggle `Show Pivot Table`:
   - Choose Rows, Columns, Value, and Aggregation; optional quantile transforms; screenshot
8) `Create Custom Column`:
   - Name: `pct_change`; Expr: `(close - open) / open * 100`; expect success message and table refresh; screenshot
9) Toolbar toggles:
   - Click the four glyph buttons (◆, ◷, ▲, ■) to reveal Outliers/Time/Trend/Dataset; verify visibility of the new sections
10) Teardown: close browser and kill Streamlit processes

### Assertions (minimal, high‑signal)
- Ready state: header present
- Table state: `stDataFrame` present
- Chart state: presence of Plotly canvas after required controls selected
- Heatmap state: rotated column labels visible
- Pivot state: pivot container rendered after rows/cols/value set
- Custom column: success toast visible or presence of new column in table/controls on rerun

### Flake mitigation
- Always wait 15s on the first page open
- Use `wait_for_selector` for every major transition
- Scroll before clicking controls below the fold
- Prefer label text over dynamic role‑based selectors
- If a click collides with a table header of the same text, open the dropdown first (click on the control), then target the listbox item

### Artifacts and reporting
- Capture full‑page screenshots after each major assertion (table, each chart type, heatmap, pivot, custom column)
- Summarize findings succinctly: what rendered, what options chosen, any warnings

### Known behaviors
- Streamlit emits deprecation messages for `use_container_width` (cosmetic)
- Initial stderr printing during boot can trigger `BrokenPipeError` in some headless setups; avoid writing to stderr during app boot

### Teardown
- Close the Playwright browser using the driver tool
- Kill Streamlit processes: `ps aux | grep 'streamlit run .*streamlit_app.py' | awk '{print $2}' | xargs kill -9`

### Quality checklist (pre‑commit)
- [ ] App launched via `.venv` with `PYTHONPATH` set
- [ ] Waited 15s before first interaction
- [ ] Header readiness verified
- [ ] Table rendered and captured
- [ ] All chart types rendered and captured
- [ ] Heatmap rendered and captured
- [ ] Pivot table rendered and captured
- [ ] Custom column created and visible
- [ ] Toolbar toggles verified
- [ ] Browser closed and Streamlit shutdown


