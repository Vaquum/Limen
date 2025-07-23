Here is a few guidelines to ensure that our tests are as readable and maintainable as possible

- Keep comments to minimum
- Make sure that everything fails hard (if one thing fails all tests fail)
- Always use the standard modular setup in `tests/run.py`
- Add common utils to `tests/utils`

Make sure that each individual test is wrapped in the following manner:

```
try:
    some_test_function()
    print(f'    ✅ {some_test_function.__name__}: PASSED')
except Exception as e:
    print(f'    ❌ {some_test_function.__name__}: FAILED - {e}')
```

**NOTE:** No other printouts.