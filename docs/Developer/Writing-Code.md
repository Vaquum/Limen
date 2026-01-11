# Code Style

## Philohopy

### Radical Simplicity
Simplicity isn’t just preferred—it’s the governing law. Choose the most straightforward design, the clearest algorithm, the leanest interface. Anything that adds complexity must earn its place.

### Consistency Over Creativity
Uniform style ([PEP 8](https://legacy.python.org/dev/peps/pep-0008/) + [house rules](#guidelines)) removes mental friction. Identical filename ↔ function name pairs, single quotes (double only in f-strings), upper-case constants, lower-case variables—these small disciplines let readers focus on logic, not format.

### Self-Evident Code
Code should explain itself. Docstrings are exhaustive; inline comments are rare and reserved for genuinely non-obvious decisions. No embedded examples—documentation and tests live elsewhere.

### Type-Safe Explicitness
Every declaration is type-hinted; magic numbers become named constants. When intent is machine-checkable, make it so.

### Modular Brevity
Functions grow past ~50 lines? They graduate into their own file. Smaller scope, smaller surface area, easier reasoning.

### Test-And-Doc First-Class
A contribution isn’t complete until it ships with thorough tests and matching docs. Quality gates are as important as the feature itself.

### LLM as a Drafting Tool, Not an Autopilot
Use language models to explore and iterate, but never drop raw output in the repo. The final pass—simplification, polish, and accountability—is always human.

## Guidelines

**NOTE:** Never simply copy-paste LLM code. Work with LLMs iteratively to simplify and clean your contribution, and always do the final simplification yourself. 

When making code contributions to Loop, always follow these guidelines:

- Follow PEP8 for style guidance, except when stated otherwise here
- Make declarations with type hints
- Never add comments unless it is critical to have it
- Never add examples
- Add comprehensive docstrings following the standard format (see codebase for examples)
- Add one empty line above docstring, and one empty line below docstring
- Add tests to new code
- Add docs to new code
- Use single quotes, except with f-strings where double quotes are always used
- Choose empty line (over no empty line) when in doubt
- Make functions over 50 lines its own file, except when there is good reason for not to
- Make the filename and the function name identical
- Make magic numbers constants
- Make constants uppercase
- Make variables lowercase
- Pass ruff + mypy with zero errors before every PR
- Order imports: stdlib → third-party → local; never use * imports
- Use logging.getLogger(__name__)—never print() in library code
- Avoid mutable default arguments (def fn(a=[]): ... is forbidden)
- Handle resources with context managers (with), not manual open/close
- Catch specific exceptions only; no bare except: blocks
- Break on exception except exceptionally
- Use pathlib over os.path for file paths
- Limit external dependencies; prefer the standard library or existing project deps
- Expose public API explicitly via __all__; prefix internal names with _