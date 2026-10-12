# `limen.experiment.param_search`

> Search strategies that yield parameter combinations from a mutable `ParamDomain`.

## Canonical docs

- [Advanced Search](../../../docs/Advanced-Search.md)

## What this package owns

Owns the pluggable search strategies the advanced experiment path uses to enumerate parameter permutations, plus the registry that names them.
Does **not** own the parameter domain (`limen.experiment.param_domain`), the search queue (`limen.experiment.msq`), or the loop that drives them.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `SearchStrategy` | Abstract base for strategies | Iterators that yield `dict[str, Any]` combinations and observe domain mutations |
| `GridStrategy` | Exhaustive grid over the domain | Deterministic full sweep |
| `RandomStrategy` | Sampled combinations | Accepts a `seed` for reproducibility |
| `STRATEGY_REGISTRY` | Name-to-class lookup | Resolves the strategy named in a manifest |

## Adjacent modules

- `limen.experiment.param_domain.ParamDomain` supplies the domain strategies iterate; strategies register as observers so they react to domain mutations.
- `limen.experiment` (UEL, MSQ, reducers) drives the strategy and feeds feedback back into the domain.

## Quick orientation

```text
param_search/
├── search_strategy.py   # SearchStrategy abstract base
├── grid_strategy.py     # Exhaustive grid
├── random_strategy.py   # Random sampling
└── registry.py          # STRATEGY_REGISTRY
```

## Things to know

- A strategy is an iterator: it yields one parameter `dict` per permutation and stops when the domain is exhausted.
- Strategies hold a reference to their `ParamDomain` and register as observers, so reducers that mutate the domain mid-run change what the strategy yields next.
- `RandomStrategy` is only reproducible when constructed with an explicit `seed`.

## Read next

- [Advanced Search](../../../docs/Advanced-Search.md)
- [Reducers And Feedback](../../../docs/Reducers-And-Feedback.md)
