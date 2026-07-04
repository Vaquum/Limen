# `limen.cohort.sfc`

> Single-file cohort selectors that choose which experiment permutations a `Cohort` aggregates.

## Canonical docs

- [Cohort](../../../docs/Cohort.md)

## What this package owns

Owns the built-in selector strategies that turn completed-experiment artefacts into a list of permutation ids for `Cohort`, plus the `BUILTIN_SELECTORS` registry that names them.
Does **not** own cohort aggregation itself (owned by `limen.cohort.Cohort`) or experiment execution.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `BUILTIN_SELECTORS` | Name-to-selector registry | Maps `'all'`, `'top_n'`, `'backtest_pareto'`, `'diverse_metrics'` |
| `all.select` | Use every round | Default; preserves omitted-`permutation_ids` behaviour |
| `top_n.select` | Rank by one numeric `results.csv` column | Requires `results.csv` in context |
| `backtest_pareto.select` | Trading-metric Pareto selection | Uses backtest return/risk columns |
| `diverse_metrics.select` | Spread selection across metric space | Favours complementary rather than similar rounds |

## Adjacent modules

- `limen.cohort.Cohort` consumes the ids these selectors return.
- `limen.experiment` produces the artefacts (`results.csv`, backtest columns) selectors read.

## Quick orientation

```text
sfc/
├── all.py               # Select every permutation
├── top_n.py             # Rank by a numeric results column
├── backtest_pareto.py   # Pareto front over trading metrics
└── diverse_metrics.py   # Diverse spread across metric space
```

## Things to know

- Every selector shares the signature `select(context, *, ...)` and returns a `list[int | str]` of permutation ids.
- `context` is a `SelectorContext` (`dict[str, Any]`) carrying artefacts such as `context['results']`; each selector validates the keys it needs and raises `ValueError` when they are missing.
- `top_n` requires `results.csv` data; passing an empty or wrong-typed frame raises rather than silently returning nothing.

## Read next

- [Cohort](../../../docs/Cohort.md)
- [Trainer](../../../docs/Trainer.md)
