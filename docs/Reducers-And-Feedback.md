# Reducers and feedback

Reducers and feedback are the control layer inside Limen's advanced search path. They are what let a run react to its own results instead of sweeping the full original domain unchanged.

This layer is coordinated by `FeedbackController`, which can collect interventions from three sources:

- pruning strategies, also called reducers
- an in-process `intra_callback`
- an optional JSON intervention file

## Prerequisites

- the artifact-backed [Advanced Search](Advanced-Search.md) path
- a positive `feedback_interval`
- enough completed rounds for the selected reducer's observation threshold
- review of `audit.jsonl` after every adaptive run

## Feedback cycle order

Each feedback trigger runs sources in this order:

1. pruning strategies
2. `intra_callback`
3. intervention file

The order matters because later sources see the `MSQ` after earlier sources have already modified it.

Two other behavior rules matter:

- failures are isolated per source, so one bad source does not block the others
- suggestion interventions are logged but not dispatched to the queue

## The intervention surface

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

Reducers return declarative dicts by default. The `intra_callback` receives direct `MSQ` access and can call queue methods itself.

## Built-in reducers

### `BudgetReducer`

`BudgetReducer` is resource-driven. It does not analyze model quality; it analyzes whether the remaining search still fits the configured budget.

In an artifact-backed advanced run, `BudgetReducer(max_permutations=4)` trims a requested `6`-round experiment down to `4` completed rows and writes the trim to both:

- `audit.jsonl`
- `checkpoint.json`

In direct `worst_first` analysis, it emitted:

```python
[
    {'op': 'remove_is', 'param': 'a', 'value': 1, 'reason': 'sanity rule'},
    {'op': 'remove_is', 'param': 'b', 'value': 'x', 'reason': 'sanity rule'},
    {'op': 'trim', 'target_count': 5, 'reason': 'budget limit'},
]
```

### `FocusReducer`

`FocusReducer` reacts to a breakthrough score and narrows the search near the current winner.

In a live direct reducer run, a breakthrough above `0.9` emitted:

- one `keep_between` filter for a numeric parameter
- one `keep_values` filter for a categorical parameter
- multiple `inject_value` variations near the winning numeric value

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
    'reason': 'wrong-direction negative correlation on auc',
}
```

It can also emit suggestion-style `keep_is` interventions for low-impact parameters.

## When to use which reducer

The five reducers answer different questions. Pick by the problem you have, not by wanting "more feedback".

| Reducer | Reach for it when | What it does | Reversible |
|---|---|---|---|
| `BudgetReducer` | the sweep must finish inside a walltime or permutation cap | trims the remaining queue once, when the budget is projected to overrun | no — a trim is permanent |
| `SanityReducer` | you want to stop wasting rounds on broken parameter values | hard-removes values whose metric is mostly null; flags zero-metric, timeout, and warning-heavy values as suggestions | removals no; suggestions are advisory |
| `CorrelationReducer` | you want to prune values that move the metric the wrong way or barely at all | removes wrong-direction values, suggests keeps for low-impact ones | removals no; suggestions advisory |
| `FocusReducer` | a breakthrough score has appeared and you want to exploit near the winner | narrows each parameter around the best round and injects nearby variations | yes — named filters, snaps back after a timeout |
| `SaturationReducer` | a value's metric has stopped varying and no longer earns full budget | down-samples saturated values, keeping a monitoring sample | yes — restores a value if it starts varying again |

`SanityReducer` is defensive and safe to run throughout. `BudgetReducer` is a resource cap. The other three change *what* the search explores, so they carry more risk of steering the run — introduce them one at a time and read `audit.jsonl` to see their effect.

Reducers combine, but with no conflict resolution. Within a feedback trigger they run in the order of the list they were constructed in, each analyzing the same pre-trigger `MSQ`; their interventions are dispatched afterwards with last-write-wins at the queue. `FocusReducer` and `SaturationReducer` use namespaced filter keys, so they do not clobber each other, but two exploitation reducers can still fight (e.g. a `BudgetReducer` trim shrinking the queue that a `FocusReducer` is injecting into). See [Feedback cycle order](#feedback-cycle-order) and [One concrete mixed feedback cycle](#one-concrete-mixed-feedback-cycle).

## Tuning guidelines

Every reducer takes `metric` (the column it reasons over) and `active` (default `True`). The knobs below are the ones that change behavior most; each reducer validates its ranges and raises `ValueError` on bad input.

`BudgetReducer` — a resource cap, not a quality judge:

| Parameter | Default | Effect |
|---|---|---|
| `max_walltime_hours` | `None` | wall-clock ceiling; the trim target is the smaller of the walltime and permutation projections |
| `max_permutations` | `None` | hard cap on completed rounds |
| `check_after_pct` | `0.1` | fraction of budget consumed before a trim is allowed; larger waits for a steadier throughput estimate, smaller reacts earlier on noisier estimates |
| `trim_strategy` | `'random'` | `'random'` downsamples blindly; `'worst_first'` removes the worst-scoring values first and requires `metric` |

`SanityReducer` — only the null-rate path removes; the rest suggest:

| Parameter | Default | Effect |
|---|---|---|
| `nan_threshold` | `0.1` | null/NaN rate above which a value is hard-removed; smaller is stricter |
| `min_observations` | `1` | trials required before a value can be removed, guarding against tiny samples |
| `zero_metric_threshold` / `execution_time_threshold` / `warning_threshold` | `None` | opt-in suggestion detectors; each stays off until set |

`CorrelationReducer` — needs enough rounds to be meaningful:

| Parameter | Default | Effect |
|---|---|---|
| `min_observations` | `50` | rounds required before it acts; larger is more conservative |
| `negative_correlation_threshold` | `-0.3` | wrong-direction trigger; nearer `0` prunes more readily, more negative prunes only strong offenders |
| `prune_threshold` | `0.05` | absolute correlation below this marks a value low-impact (a `keep_is` suggestion) |
| `sign_stability_threshold` | `0.8` | bootstrap sign agreement required before acting; larger demands higher confidence |

`FocusReducer` — exploitation around a breakthrough:

| Parameter | Default | Effect |
|---|---|---|
| `breakthrough_threshold` | required | metric level that activates focus; set near the top of the expected range so it only fires on a real breakthrough |
| `focus_range_pct` | `0.2` | half-width of the numeric window kept around the winner; larger explores wider, smaller searches tighter |
| `focus_timeout` | `5` | rounds without improvement before it snaps back to the full domain |
| `variation_count` | `5` | number of nearby values injected per numeric parameter |

`SaturationReducer` — variance-collapse detection:

| Parameter | Default | Effect |
|---|---|---|
| `cv_threshold` | `0.01` | coefficient of variation below which a value is saturated; larger declares saturation sooner |
| `window_size` | `100` | rolling tail length per value; larger is smoother and slower to react |
| `min_samples_per_value` | `20` | observations required before the CV is trusted |
| `retain_fraction` | `0.1` | share of a saturated value's combinations kept for monitoring; nearer `0` prunes harder |

## Feedback sources beyond reducers

### `intra_callback`

`intra_callback` receives `(log, msq)` and can call queue methods directly. It is supplied through the `UniversalExperimentLoop` constructor — the Python API, not YAML:

```python
def steer_toward_lbfgs(log, msq):
    msq.keep_is('solver', 'lbfgs')
    msq.inject_value('C', 7.5)
```

```python-fragment
uel = limen.UniversalExperimentLoop(
    sfd=my_sfd,
    search_strategy=strategy,
    intra_callback=steer_toward_lbfgs,
    feedback_interval=20,
)
```

The callback applies interventions directly on the `msq` and the controller records what changed. Use `intra_callback` for:

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

## Audit trail

Each feedback trigger writes one JSONL entry to `audit.jsonl` when the advanced path is using an `experiment_dir`.

A feedback-cycle audit entry records:

- the round number
- applied interventions
- suggestion interventions
- any isolated source errors
- a compact `msq_state_after`

That audit trail is the main way to explain why the search changed mid-run.

## One concrete mixed feedback cycle

In a controller run, one trigger can apply all three sources in the same cycle:

- pruning strategy: `remove_is a=3`
- `intra_callback`: `inject_value a=99`
- intervention file: `keep_is b='x'`

All three were:

- applied to the queue
- passed to `strategy.update_from_feedback(interventions)`
- written into `audit.jsonl`

On the next trigger, changing the file to inject `a=123` produced a second audit entry with the updated file-driven intervention.

## Reducer registry

All built-in reducers are available via `REDUCER_REGISTRY` for params-based or programmatic selection:

```python
from limen.experiment.reducer import REDUCER_REGISTRY
```

| Key | Class |
|-----|-------|
| `'budget'` | `BudgetReducer` |
| `'correlation'` | `CorrelationReducer` |
| `'focus'` | `FocusReducer` |
| `'sanity'` | `SanityReducer` |
| `'saturation'` | `SaturationReducer` |

## Configuring reducers in YAML

Reducers can be declared in a YAML experiment under `uel.pruning_strategies`, so a `limen run` gets the same feedback control as the Python API. Each entry names a reducer by its registry key and forwards its constructor arguments under `params`:

```yaml
uel:
  n_permutations: 200
  feedback_interval: 20
  pruning_strategies:
    - type: sanity
      params:
        metric: auc
    - type: budget
      params:
        max_permutations: 100
        check_after_pct: 0.25
```

`feedback_interval` sets how often the reducers run. Each `type` must be one of the [registry keys](#reducer-registry) above, and `params` are the reducer's constructor arguments from [Tuning guidelines](#tuning-guidelines). Unknown types or fields are rejected by `limen validate` before the run starts.

## Read next

- Continue to [Advanced Search](Advanced-Search.md) for the full artifact-backed run path for this feedback system.
- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) for where feedback triggers are scheduled during a run.
- Continue to [Trainer](Trainer.md) for downstream use of finished artifacts.
