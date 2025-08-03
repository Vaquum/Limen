# Code Style

## Flavor

Loop's flavor is simplicity. Simplicity is the fundamental principle governing everything else. **Make everything as simple as possible.**

## Guidelines

**NOTE:** Never simply copy-paste LLM code. Work with LLMs iteratively to simplify and clean your contribution, and always do the final simplification yourself. 

When making code contributions to Loop, always follow these guidelines:

- Always follow PEP8 for style guidance, except when stated otherwise here
- Make declarations with type hints
- Never add comments unless it is critical to have it
- Never add examples
- Add comprehensive docstrings following the standard format (see codebase for examples)
- Add tests to new code
- Add docs to new code
- Use single quotes, except with f-strings where double quotes are always used
- Choose empty line (over no empty line) when in doubt
- Make functions over 50 lines its own file, except when there is good reason for not to
- Make the filename and the function name identical
- Make magic numbers constants
- Make constants uppercase
- Make variables lowercase