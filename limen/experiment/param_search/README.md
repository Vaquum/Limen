# `limen.experiment.param_search`

> Search strategies that yield parameter combinations from a mutable `ParamDomain`.

## Canonical docs

- [Advanced Search](../../../docs/Advanced-Search.md)

## In the research projects

Both projects run `search` over recorded observations. The [Python project](../../../docs/examples/python-research/README.md) constructs strategies over `ParamDomain`; [Compiled project](../../../docs/examples/compiled-research/README.md) resolves them from its declaration. The assertions compare completed candidate identities and keep candidate-generation seeds separate from model randomness.

## What this package owns

Owns the pluggable search strategies the advanced experiment path uses to enumerate parameter permutations, plus the registry that names them.
Does **not** own the parameter domain (`limen.experiment.param_domain`), the search queue (`limen.experiment.msq`), or the loop that drives them.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `SearchStrategy` | Abstract base for strategies | Iterators that yield `dict[str, Any]` combinations and observe domain mutations |
| `GridStrategy` | Exhaustive grid over the domain | Deterministic full sweep |
| `RandomStrategy` | Sampled combinations | Accepts a `seed` for reproducibility |
| `SobolStrategy` | Scrambled Sobol candidate sampling | Requires a nonnegative integer `seed`; discrete projection is not exhaustive |
| `STRATEGY_REGISTRY` | Name-to-class lookup | Resolves the strategy named in a manifest |

## Adjacent modules

- `limen.experiment.param_domain.ParamDomain` supplies the domain strategies iterate; strategies register as observers so they react to domain mutations.
- `limen.experiment` (UEL, MSQ, reducers) drives the strategy and feeds feedback back into the domain.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `search_strategy.py` | SearchStrategy abstract base |
| `grid_strategy.py` | Exhaustive grid |
| `random_strategy.py` | Random sampling |
| `sobol_strategy.py` | Seeded scrambled Sobol sampling |
| `registry.py` | STRATEGY_REGISTRY |

<!-- reference: package-experiment-param_search-layout source=limen/experiment/param_search/README.md region=layout:package role=layout -->
```text
param_search/
├── README.md
├── __init__.py
├── _grid_permutation.py
├── grid_strategy.py
├── random_strategy.py
├── registry.py
├── search_strategy.py
└── sobol_strategy.py
```

## Things to know

- A strategy is an iterator that yields one parameter `dict` per proposal. Grid stops at exhaustion; sampled strategies can fail when bounded duplicate retries find no novel combination.
- Strategies hold a reference to their `ParamDomain` and register as observers, so reducers that mutate the domain mid-run change what the strategy yields next.
- `RandomStrategy` is only reproducible when constructed with an explicit `seed`.
- `SobolStrategy(domain, seed=42)` is also exported from this package; its mapping, limits and checkpoint contract are documented in [Advanced Search](../../../docs/Advanced-Search.md#seeded-sobol-sampling).

## Read next

- [Advanced Search](../../../docs/Advanced-Search.md)
- [Reducers And Feedback](../../../docs/Reducers-And-Feedback.md)
