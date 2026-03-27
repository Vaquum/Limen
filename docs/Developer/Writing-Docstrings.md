# Writing Docstrings

This page defines Limen's current docstring expectations for public code.

Use it when you add or change a public function, class, or method. The goal is not decorative docstrings. The goal is code-true guidance that saves future contributors and users from reading the implementation to answer basic questions.

## When A Docstring Is Required

Update or add a docstring when you change:

- a public function
- a public class
- a public method with real behavioral surface
- a public helper whose arguments, outputs, or side effects matter to callers

If behavior changed, the docstring should change with it.

## House Style

Follow the style already used across Limen:

```python
def example(data: pl.DataFrame, period: int = 14) -> pl.DataFrame:

    '''
    Short summary of what the function does.

    Args:
        data (pl.DataFrame): Input data
        period (int): Main configuration parameter

    Returns:
        pl.DataFrame: Output description
    '''
```

Current expectations:

- use a short summary first
- document meaningful arguments under `Args:`
- document return behavior under `Returns:`
- keep the wording direct and concrete

## What Good Docstrings Include

A strong Limen docstring should answer:

- what the function or class does
- what inputs it expects
- what it returns or mutates
- what columns it adds or requires when the surface is DataFrame-based
- any important caveat a caller needs to know

## What To Avoid

- repeating the function name without adding meaning
- documenting obvious assignments line by line
- leaving stale parameter names after a refactor
- describing future intended behavior as if it already exists
- hiding important side effects such as overwriting a column or depending on train-fitted state

## DataFrame-Specific Guidance

Much of Limen's public surface is Polars-based. For those functions, docstrings should be especially clear about:

- required input columns
- output column names or naming patterns
- whether rows are filtered, appended, or only transformed
- whether the helper is stateless or fitted on train data

## Review Standard

When reviewing a docstring, ask:

- is it still true after this change
- could a caller understand the surface without opening the implementation
- does it name the important columns, params, or artifacts
- is it concise enough to stay readable

## Read Next

- [Developer Home](README.md)
- [Documentation System Contract](Documentation-System.md)
