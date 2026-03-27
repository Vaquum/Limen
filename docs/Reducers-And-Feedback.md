# Reducers And Feedback

Reducers and feedback are the control layer inside Limen's advanced search path. They are what let a run react to its own results instead of sweeping the full original domain unchanged.

This layer is coordinated by `FeedbackController`, which can collect interventions from three sources:

- pruning strategies, also called reducers
- an in-process `intra_callback`
- an optional JSON intervention file

## Feedback Cycle Order

Each feedback trigger runs sources in this order:

1. pruning strategies
2. `intra_callback`
3. intervention file

The order matters because later sources see the `MSQ` after earlier sources have already modified it.

Two other behavior rules matter:

- failures are isolated per source, so one bad source does not block the others
- suggestion interventions are logged but not dispatched to the queue

## The Intervention Surface

The `MSQ` layer currently supports these intervention families:

| Operation | Effect |
|---|---|
| `remove_is` | remove one exact parameter value |
| `remove_ge`, `remove_le` | remove numeric values above or below a threshold |
| `keep_is`, `keep_between` | narrow a parameter to one value or a numeric range |
| `inject_value` | add a new value into the parameter domain |
| `trim` | reduce the remaining queue budget |
| `set_filter`, `clear_filter` | apply or remove named multi-combination filters |
| `remove_custom` | direct callback-only custom combo filtering |

Reducers usually return declarative dicts. The `intra_callback` is different: it receives direct `MSQ` access and can call queue methods itself.

## Built-In Reducers

### `BudgetReducer`

`BudgetReducer` is resource-driven. It does not analyze model quality; it analyzes whether the remaining search still fits the configured budget.

On a live local run in this repo, `BudgetReducer(max_permutations=4)` trimmed a requested `6`-round experiment down to `4` completed rows and wrote the trim to both:

- `audit.jsonl`
- `checkpoint.json`

In direct `worst_first` analysis, it emitted:

```python
[
    {'op': 'remove_is', 'param': 'a', 'value': 1, ...},
    {'op': 'remove_is', 'param': 'b', 'value': 'x', ...},
    {'op': 'trim', 'target_count': 5, ...},
]
```

### `FocusReducer`

`FocusReducer` reacts to a breakthrough score and narrows the search around the current winner.

In a live direct reducer run, a breakthrough above `0.9` emitted:

- one `keep_between` filter for a numeric parameter
- one `keep_values` filter for a categorical parameter
- multiple `inject_value` variations around the winning numeric value

This is the exploitation-style reducer in the current package.

### `SaturationReducer`

`SaturationReducer` looks for parameter values whose metric variance has collapsed and then keeps only a sample of combinations for continued monitoring.

In a live direct run, it emitted:

```python
{
    'op': 'set_filter',
    'key': 'saturation_a_0',
    'filter_type': 'sample',
    'filter_params': {'param': 'a', 'value': 0, 'fraction': 0.1},
}
```

### `SanityReducer`

`SanityReducer` is the defensive reducer. It looks for broken or unreliable parameter values, such as:

- high NaN or null rates in the target metric
- zero-metric collapse
- timeout-heavy values
- warning-heavy values

In a live direct run with `zero_metric_threshold=0.5`, it emitted suggestion-only interventions such as:

```python
{
    'op': 'remove_is',
    'param': 'a',
    'value': 1,
    'action': 'suggest',
    'reason': 'zero_metric rate 1.00 for a=1',
}
```

Because that is a suggestion, it is logged but not applied.

### `CorrelationReducer`

`CorrelationReducer` uses bootstrap parameter-to-metric correlation analysis.

In a live direct run, it detected a wrong-direction parameter and emitted:

```python
{
    'op': 'remove_is',
    'param': 'a',
    'value': 1,
    'reason': 'wrong-direction (negative) correlation ...',
}
```

It can also emit suggestion-style `keep_is` interventions for low-impact parameters.

## Feedback Sources Beyond Reducers

### `intra_callback`

`intra_callback` receives `(log, msq)` and can call queue methods directly.

Use it when you want:

- arbitrary Python-side control logic
- direct queue mutation that is too custom for declarative reducer output
- experiment-time hooks without creating a reducer class

### `interventions.json`

When `experiment_dir` is set, `FeedbackController` watches:

```text
<experiment_dir>/interventions.json
```

This file is polled by modification time. If it changes, Limen reads the JSON list and applies those interventions during the next feedback cycle.

Example:

```json
[
  {"op": "keep_is", "param": "solver", "value": "lbfgs"},
  {"op": "inject_value", "param": "C", "value": 7.5}
]
```

## Audit Trail

Each feedback trigger writes one JSONL entry to `audit.jsonl` when the advanced path is using an `experiment_dir`.

A live local run in this repo recorded:

- the round number
- applied interventions
- suggestion interventions
- any isolated source errors
- a compact `msq_state_after`

That audit trail is the main way to explain why the search changed mid-run.

## One Concrete Mixed Feedback Cycle

In a live local controller run in this repo, one trigger applied all three sources in the same cycle:

- pruning strategy: `remove_is a=3`
- `intra_callback`: `inject_value a=99`
- intervention file: `keep_is b='x'`

All three were:

- applied to the queue
- passed to `strategy.update_from_feedback(...)`
- written into `audit.jsonl`

On the next trigger, changing the file to inject `a=123` produced a second audit entry with the updated file-driven intervention.

## Read Next

- Continue to [Advanced Search](Advanced-Search.md) for the full artifact-rich run path around this feedback system.
- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) for where feedback triggers are scheduled during a run.
- Continue to [Trainer](Trainer.md) if you want to use the finished artifacts downstream.
