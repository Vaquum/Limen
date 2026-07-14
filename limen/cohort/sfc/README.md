# `limen.cohort.sfc`

> Single-file cohort selectors that choose which experiment permutations a `Cohort` aggregates.

## Canonical docs

- [Cohort](../../../docs/Cohort.md)

## What this package owns

Owns the built-in selector strategies that turn completed-experiment artifacts into a list of permutation ids for `Cohort`, plus the `BUILTIN_SELECTORS` registry that names them.
Does **not** own cohort aggregation itself (owned by `limen.cohort.Cohort`) or experiment execution.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `BUILTIN_SELECTORS` | Name-to-selector registry | Maps `'all'`, `'top_n'`, `'backtest_pareto'`, `'diverse_metrics'` |
| `all.select` | Use every round | Default; preserves omitted-`permutation_ids` behavior |
| `top_n.select` | Rank by one numeric `results.csv` column | Requires `results.csv` in context |
| `backtest_pareto.select` | Trading-metric Pareto selection | Uses backtest return/risk columns |
| `diverse_metrics.select` | Spread selection across metric space | Favors complementary rather than similar rounds |

## Adjacent modules

- `limen.cohort.Cohort` consumes the ids these selectors return.
- `limen.experiment` produces the artifacts (`results.csv`, backtest columns) selectors read.

## Quick orientation

```text
sfc/
├── all.py               # Select every permutation
├── top_n.py             # Rank by a numeric results column
├── backtest_pareto.py   # Pareto front over trading metrics
└── diverse_metrics.py   # Diverse spread across metric space
```

## Things to know

- Every selector shares the signature `select(context, *, ...)`. The selector result must ultimately contain nonempty string permutation IDs because `Cohort` rejects non-string entries during binding.
- `context` is a `SelectorContext` (`dict[str, Any]`) carrying artifacts such as `context['results']`; each selector validates the keys it needs and raises `ValueError` when they are missing.
- `top_n` requires a Polars `results.csv` frame. A frame with no usable numeric ranking rows returns `[]`; `Cohort` then rejects the empty selection. A missing or wrong-typed frame raises immediately.

## Read next

- [Cohort](../../../docs/Cohort.md)
- [Trainer](../../../docs/Trainer.md)
