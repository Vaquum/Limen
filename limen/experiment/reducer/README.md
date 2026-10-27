# `limen.experiment.reducer`

> Feedback reducers that prune, trim, or refocus the parameter domain as an experiment runs.

## Canonical docs

- [Reducers And Feedback](../../../docs/Reducers-And-Feedback.md)

## In the research projects

The `search` scenarios in the [Python project](../../../docs/examples/python-research/README.md) and the [Compiled project](../../../docs/examples/compiled-research/README.md) each run a separate four-candidate grid with a three-round `BudgetReducer`. Their audit and checkpoint records establish the actual trim; they do not claim that pruning improves model quality.

## What this package owns

Owns the reducer strategies that consume in-flight experiment results and mutate the search — trimming to budget, pruning correlated or saturated regions, and refocusing on promising areas — plus the registry and shared filter/action constants.
Does **not** own the parameter domain or the search strategies it steers.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `REDUCER_REGISTRY` | Name-to-reducer lookup | Resolves reducers named in a manifest |
| `BudgetReducer` | Cap the remaining permutation budget | Trim via `TRIM_RANDOM` or `TRIM_WORST_FIRST` |
| `CorrelationReducer` | Remove wrong-direction numeric values or suggest low-impact keeps | Uses bootstrap correlation and sign stability |
| `FocusReducer` | Refocus the domain on promising regions | Narrows around strong rounds |
| `SaturationReducer` | Down-sample saturated values | Uses reversible named filters |
| `SanityReducer` | Guard against degenerate configurations | Filters obviously bad combinations |
| `PruningStrategy` | Base pruning behavior | Emits `ACTION_SUGGEST`-style actions |

## Adjacent modules

- `limen.experiment.param_domain` is the domain reducers mutate.
- `limen.experiment.param_search` strategies observe those mutations and change what they yield next.
- `limen.experiment` (UEL) invokes reducers between rounds with accumulated results.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `budget_reducer.py` | BudgetReducer (+ TRIM_* constants) |
| `correlation_reducer.py` | CorrelationReducer |
| `focus_reducer.py` | FocusReducer |
| `saturation_reducer.py` | SaturationReducer |
| `sanity_reducer.py` | SanityReducer |
| `pruning_strategy.py` | PruningStrategy (+ ACTION_SUGGEST) |
| `filter_types.py` | FILTER_* constants |
| `registry.py` | REDUCER_REGISTRY |

<!-- reference: package-experiment-reducer-layout source=limen/experiment/reducer/README.md region=layout:package role=layout -->
```text
reducer/
├── README.md
├── __init__.py
├── _forest_analysis.py
├── budget_reducer.py
├── correlation_reducer.py
├── filter_types.py
├── focus_reducer.py
├── pruning_strategy.py
├── registry.py
├── sanity_reducer.py
└── saturation_reducer.py
```

## Things to know

- Reducers inspect the log and `MSQ` but return declarative interventions without mutating either. `FeedbackController` dispatches those interventions to `MSQ`, then notifies the active search strategy.
- `filter_types.py` defines the `FILTER_*` constants (`FILTER_KEEP_VALUES`, `FILTER_KEEP_BETWEEN`, `FILTER_EXCLUDE_VALUE`, `FILTER_SAMPLE`) shared across reducers.
- Reducers are optional in CLI and Python/MSQ runs, configured through YAML or the UEL constructor.

## Read next

- [Reducers And Feedback](../../../docs/Reducers-And-Feedback.md)
- [Advanced Search](../../../docs/Advanced-Search.md)
