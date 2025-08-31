<the_system_we_already_have>everything contained in this repository</the_system_we_already_have>

<WHAT_WE_WILL_WORK_ON>

- I will give you tasks which involve either investigating or improving Loop
- All of the work we will do in small tasks iteratively
- All of our work will exclusively focus on `loop.explorer` sub-module improvements
- Never create a new environment, always use the one mentioned in `Rules.md`

<IMPORTANT> Always verify changes "visually" using `Playwrite MCP` after first running `import dev_snippets; dev_snippets.test_explorer_locally()` to start the Streamlit app. This is the only way to work together, any other way would be extremely harmful. **NOTE:** It will not work if you run it from command line, you must run it from python interactive cell with the given command. When you run it, at end it will return host and port in dictionary, just capture those and you're good to go with Playwright, though you'll of course have to change `0.0.0.0` to `localhost` to have the correct host.

<YOUR_CURRENT_TASK>

Ok, I can see now that while you can call the tools, open is the only one that actually works. So let's first now focus on making `mcp/playwright_driver` work properly. All the tool calls must work. So...

1) Start the standard virtual environment with `source .venv/bin/activate`
2) Set `PYTHONPATH` with `export PYTHONPATH=$(pwd):$PYTHONPATH`
3) Run the following in an interactive python shell in `.venv`:

```python

import dev_snippets;

address = dev_snippets.test_explorer_locally()

port = address['port']

url = f"http://localhost/{port}"

print(url)

```

4) Use all the tools one by one in playwright_driver and playwright/browser-tools to ensure they work in an expected way
5) Analyze the findings of each tool
6) Evaluate their usefulness for debugging
7) Report back your findings

**NOTE:** Whenever your first open browser with the `url`, it can take about 10 seconds before everything loads