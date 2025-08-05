# Writing Tests

## When to add tests?

If you are adding new code, add tests that cover that code. If you are adding an SFM, there is no requirement to write additional tests, simply drop it to the test harness in [`loop/tests/test_sfm.py`]('../loop/tests/test_sfm.py'). 

## Standards

Here is a few guidelines to ensure that our tests are as readable and maintainable as possible

- Keep comments to minimum
- Make sure that everything fails hard (if one thing fails all tests fail)
- Always use the standard modular setup in `tests/run.py`
- Add common utils to `tests/utils`
- Never add any fallbacks to tests
- Never add any printouts to tests

**NOTE:** Simply by adding test to [`loop/tests/run.py`]('../loop/tests/run.py'), and additional to [`loop/tests/test_sfm.py`]('../loop/tests/test_sfm.py').