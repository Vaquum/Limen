# Writing Tests

## When to add tests?

If you are adding new code, add tests that cover that code. If you are adding an Foundational SFD, there is no requirement to write additional tests, simply drop it to the test harness in [`tests/test_foundational_sfd.py`]('../tests/test_foundational_sfd.py').

## Standards

Here is a few guidelines to ensure that our tests are as readable and maintainable as possible

- Keep comments to minimum
- Make sure that everything fails hard (if one thing fails all tests fail)
- Always use the standard modular setup in `tests/run.py`
- Add common utils to `tests/utils`
- Never add any fallbacks to tests
- Never add any printouts to tests

**NOTE:** Simply add tests to [`tests/run.py`]('../tests/run.py') and additional Foundational SFD tests to [`tests/test_foundational_sfd.py`]('../tests/test_foundational_sfd.py').

Run tests with: `python tests/run.py` or `python -m tests.run`