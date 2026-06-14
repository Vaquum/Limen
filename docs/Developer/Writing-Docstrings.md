# Writing docstrings

This page defines Limen's current docstring expectations for public code.

This page applies when a public function, class, or method changes. The goal is code-true guidance that lets future contributors and users answer basic questions without opening the implementation.

## When a docstring is required

Update or add a docstring when a change touches:

- a public function
- a public class
- a public method with real behavioral surface
- a public helper whose arguments, outputs, or side effects matter to callers

If behavior changed, the docstring should change with it.

## House style

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

## What docstrings include

A Limen docstring should answer:

- what the function or class does
- what inputs it expects
- what it returns or mutates
- what columns it adds or requires when the surface is DataFrame-based
- caveats that affect caller behavior

## What to avoid

- repeating the function name without adding meaning
- documenting obvious assignments line by line
- leaving stale parameter names after a refactor
- describing future intended behavior as if it already exists
- hiding important side effects such as overwriting a column or depending on train-fitted state

## DataFrame-specific guidance

Much of Limen's public surface is Polars-based. For those functions, docstrings should specify:

- required input columns
- output column names or naming patterns
- whether rows are filtered, appended, or only transformed
- whether the helper is stateless or fitted on train data

## Review standard

Docstring review checks that the text remains true after the change, names the relevant columns, params, or artifacts, and gives callers enough information without opening the implementation.

## Read next

- [Developer Home](README.md)
- [Documentation System Contract](Documentation-System.md)
