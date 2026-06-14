# Pruning Strategies (Reducers)

## 1. Introduction

In Limen, a parameter sweep explores a potentially vast combinatorial space.
**Reducers** are pruning strategies that collapse this space during the
experiment based on evidence, removing parameter values and combinations that
are broken, redundant, or low-impact before compute is spent on them.

Every reducer implements the **Principle of Least Action**: maximize
information gain while minimizing compute expenditure. The six reducers form a
complete, orthogonal set — each operates on a distinct signal dimension:

| Reducer | Signal Dimension | Eliminates |
|---|---|---|
| Sanity | Validity | Broken combinations (NaN, OOM) |
| Correlation | Trend | Statistically low-impact parameter values |
| Saturation | Variance | Redundant sampling (converged regions) |
| Focus | Local Optimality | Everything far from a breakthrough |
| Budget | Resource | Excess combinations that exceed time/compute budget |
| Manual | External Will | Whatever the operator decides |

---

## 2. Architecture Overview

Reducers implement the `PruningStrategy` ABC
(`limen/experiment/reducer/pruning_strategy.py`) and are invoked by
`FeedbackController` (`limen/experiment/feedback_controller.py`) at each
feedback interval. The flow is:

```
FeedbackController.trigger()
  │
  ├─ For each active PruningStrategy:
  │    strategy.analyze_and_intervene(log, msq)
  │      → returns list of intervention dicts
  │
  ├─ Dispatch each intervention to MSQ
  │    (remove_is, remove_ge, keep_between, trim, inject, inject_value)
  │
  ├─ Notify SearchStrategy of changes
  │
  └─ Write audit trail entry (JSONL)
```

**Key design principle:** Reducers are pure analysis + decision components.
They read the log, decide what to prune, and return intervention dicts. The
actual MSQ mutation is handled by `FeedbackController._apply_intervention()`.

Each reducer exposes `get_state()` / `set_state()` for checkpoint support.

For the full experiment lifecycle, see
[Universal-Experiment-Loop.md](../Universal-Experiment-Loop.md).

---

## 3. Reducer Reference

### Intervention Operations

All reducers return intervention dicts with an `op` key. Available operations:

| Operation | Required Keys | Effect |
|---|---|---|
| `remove_is` | `param`, `value` | Remove a specific value from a parameter |
| `remove_ge` | `param`, `threshold` | Remove all values greater than or equal to threshold |
| `remove_le` | `param`, `threshold` | Remove all values less than or equal to threshold |
| `keep_is` | `param`, `value` | Keep only this value, remove all others |
| `keep_between` | `param`, `lower`, `upper` | Keep only values in [lower, upper] |
| `inject` | `combo`, `prioritize` (optional) | Add a specific combination to the queue |
| `inject_value` | `param`, `value` | Add a new value to a parameter's domain |
| `trim` | `target_count` | Limit remaining combinations to target_count |

---

### 3.1 Sanity Reducer

**Signal dimension:** Validity — combination violates feasibility constraints.

**Trigger:** A completed permutation produces NaN results or other failure
indicators.

#### Args

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | `str` | required | Target metric column to check for NaN |
| `nan_threshold` | `float` | `0.1` | Failure rate above which a param value is removed |
| `min_observations` | `int` | `1` | Min trials per value before evaluating |

#### YAML Config

```yaml
pruning_strategies:
  - type: sanity
    params:
      metric: auc
      nan_threshold: 0.1
      min_observations: 1
```

#### Behavior

1. Scan experiment log for rows where `metric` is NaN
2. Calculate failure rate per parameter value:
   `failure_rate = nan_count / total_trials`
3. If `failure_rate > nan_threshold` and `total_trials >= min_observations`,
   emit `remove_is` for that parameter value

---

### 3.2 Correlation Reducer

**Signal dimension:** Trend — is a parameter value consistently associated
with low metric values

**Trigger:** Sufficient observations accumulated to compute reliable
correlations.

#### Args

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | `str` | required | Target metric for correlation |
| `method` | `str` | `'spearman'` | Correlation method |
| `min_observations` | `int` | `50` | Min completed rounds before analysis |
| `prune_threshold` | `float` | `0.05` | Absolute correlation below this is low-impact |
| `sign_stability_threshold` | `float` | `0.8` | Min sign stability to act |

#### YAML Config

```yaml
pruning_strategies:
  - type: correlation
    params:
      metric: auc
      method: spearman
      min_observations: 50
      prune_threshold: 0.05
      sign_stability_threshold: 0.8
```

#### Behavior

1. Wait until `min_observations` rounds are completed
2. Compute bootstrap correlations between each parameter and the target metric
3. If `|correlation| < prune_threshold` and
   `sign_stability > sign_stability_threshold`, the parameter is low-impact —
   fix to its highest-performing value via `keep_is`

---

### 3.3 Saturation Reducer

**Signal dimension:** Variance — is continued sampling yielding new
information deficit

**Trigger:** A parameter value's result variance drops below threshold.

#### Args

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | `str` | required | Target metric for variance monitoring |
| `window_size` | `int` | `100` | Rolling window per parameter value |
| `cv_threshold` | `float` | `0.01` | CV below this indicates saturation |
| `retain_fraction` | `float` | `0.1` | Fraction of saturated combos to keep |
| `min_samples_per_value` | `int` | `20` | Min samples before evaluating |

#### YAML Config

```yaml
pruning_strategies:
  - type: saturation
    params:
      metric: auc
      window_size: 100
      cv_threshold: 0.01
      retain_fraction: 0.1
      min_samples_per_value: 20
```

#### Behavior

1. For each parameter value, collect the last `window_size` metric values
2. Compute coefficient of variation: `CV = std / mean`
3. If `CV < cv_threshold` and `samples >= min_samples_per_value`, the value
   is saturated — remove `(1 - retain_fraction)` of its pending combinations
4. `retain_fraction` keeps a bounded sample for continued monitoring

---

### 3.4 Focus Reducer

**Signal dimension:** Local optimality — metric crosses the configured breakthrough threshold.

**Trigger:** A permutation result crosses `breakthrough_threshold`, switching
from exploration to exploitation.

#### Args

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | `str` | required | Target metric for breakthrough detection |
| `breakthrough_threshold` | `float` | required | Metric value that triggers focus mode |
| `focus_range_pct` | `float` | `0.2` | Fraction of range to keep around breakthrough |
| `focus_timeout` | `int` | `200` | Observations without improvement before revert |
| `inject_variations` | `bool` | `False` | Whether to inject fine-grained combos |
| `variation_count` | `int` | `10` | Number of variations to inject |

#### YAML Config

```yaml
pruning_strategies:
  - type: focus
    params:
      metric: auc
      breakthrough_threshold: 0.95
      focus_range_pct: 0.2
      focus_timeout: 200
      inject_variations: true
      variation_count: 10
```

#### Behavior

1. Monitor results for `metric >= breakthrough_threshold`
2. On breakthrough, transition to focus mode:
   - For numeric parameters: `keep_between` using `focus_range_pct` of the
     parameter range centered on the breakthrough value
   - For categorical parameters: `keep_is` with the breakthrough value
3. If `inject_variations` is true, emit `inject` interventions with
   `variation_count` fine-grained combinations near the breakthrough point
4. If `focus_timeout` observations pass without improvement, revert to full
   space (focus mode OFF)

---

### 3.5 Budget Reducer

**Signal dimension:** Resource — projected resource use exceeds the configured budget.

**Trigger:** Elapsed time or completed permutations approach configured limits.

#### Args

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_walltime_hours` | `float \| None` | `None` | Wall-clock time limit |
| `max_permutations` | `int \| None` | `None` | Total permutation limit |
| `trim_strategy` | `str` | `'random'` | How to trim: `'random'` or `'worst_first'` |
| `check_after_pct` | `float` | `0.1` | Start checking after this fraction of budget |

#### YAML Config

```yaml
pruning_strategies:
  - type: budget
    params:
      max_walltime_hours: 24
      max_permutations: 10000
      trim_strategy: random
      check_after_pct: 0.1
```

#### Behavior

1. Skip checks until `check_after_pct` of budget is consumed
2. Project completion based on current throughput and `msq.remaining_count()`
3. If projected completion exceeds budget, emit `trim` intervention:
   - `random`: uniform downsample of the queue
   - `worst_first`: remove combinations most similar to low-performing
     completed ones

---

### 3.6 Manual Reducer

**Signal dimension:** External will — human operator override.

Already implemented via `FeedbackController._collect_from_file()`. The
`intervention_path` parameter points to a JSON file that is polled by
modification time at each feedback trigger.

#### Control File Format

```json
[
    {"op": "remove_is", "param": "optimizer", "value": "sgd"},
    {"op": "keep_between", "param": "learning_rate", "lower": 0.001, "upper": 0.01},
    {"op": "trim", "target_count": 5000}
]
```

Each dict follows the same intervention format as automated reducers. An
optional `"source"` key is auto-filled as `"intervention_file"` if absent.

#### YAML Config

```yaml
feedback_controller:
  intervention_path: ./interventions.json
```

---

## 4. Auto-Pruning Mode

Auto-pruning instantiates all reducers and executes them in a fixed order at
each feedback trigger:

1. **Sanity** — Remove broken combinations before statistical analysis
2. **Correlation** — Remove statistically low-impact parameter values
3. **Saturation** — Remove redundant exploration
4. **Focus** — Narrow to breakthrough region (if triggered)
5. **Budget** — Trim to fit resource constraints (last resort)
6. **Manual** — Apply human overrides last (highest authority)

### Minimum Remaining Space Safeguard

`min_remaining_fraction` (default `0.05`) prevents over-pruning. After each
reducer executes, if `msq.remaining_count()` drops below
`min_remaining_fraction × original_count`, remaining reducers are skipped for
that trigger cycle.

#### YAML Config

```yaml
pruning_strategies:
  - type: auto
    params:
      metric: auc
      min_remaining_fraction: 0.05
      sanity:
        nan_threshold: 0.15
      focus:
        breakthrough_threshold: 0.90
```

---

## 5. Selection Guide

### Decision Matrix

| Scenario | Recommended Reducers |
|---|---|
| Quick exploration — small space, fast SFD | Sanity |
| Production sweep — moderate space, time-boxed | Sanity + Correlation + Budget |
| Deep optimization — large space, quality-focused | Sanity + Correlation + Saturation + Focus |
| Full auto — maximum efficiency | All (auto-pruning mode) |

### Tuning Guidelines

**`min_observations`** (Correlation, Sanity)
Lower values react faster but risk false positives from small samples. Start
with defaults; increase if the experiment has high metric variance.

**`nan_threshold`** (Sanity)
At `0.1`, a parameter value is removed when >10% of its trials fail. Lower
for strict experiments (e.g. `0.05`), raise for noisy SFDs where occasional
NaN is expected.

**`breakthrough_threshold`** (Focus)
Must reflect genuine domain knowledge — set to a metric value that represents
a meaningful result. Setting too low triggers premature exploitation.

**`cv_threshold`** (Saturation)
At `0.01` (1% CV), the metric must be nearly constant to trigger. Raise to
`0.05` for faster pruning in noisy experiments.

### Interaction Effects

When multiple reducers run together:

- **Correlation + Saturation** can compound: Correlation removes low-impact
  parameters, Saturation removes converged values from the remaining ones.
  Monitor `msq.remaining_count()` in the audit log.
- **Focus overrides Correlation**: Once focus mode activates, it narrows all
  parameters regardless of their correlation. Correlation results computed
  before focus may be stale.
- **Budget is the safety net**: Budget trimming fires only if the queue is
  still too large after all other reducers. It never adds complexity.
- The `min_remaining_fraction` safeguard prevents the combined effect of
  multiple reducers from collapsing the space below a usable size.

---

## 6. Testing Strategy

### Existing Test Infrastructure

| Asset | Location | Purpose |
|---|---|---|
| `StubPruningStrategy` | `tests/stubs/stubs.py` | Configurable test double returning preset interventions |
| `StubStrategy` | `tests/stubs/stubs.py` | Stub `SearchStrategy` for MSQ construction |
| `make_msq()` | `tests/stubs/stubs.py` | Create MSQ with configurable params for testing |
| `random_binary` SFD | `limen/sfd/foundational_sfd/` | Produces real Log data with metrics |

### Needed: Synthetic Log Builder

A `make_log()` test helper producing controlled `pl.DataFrame` with:

- Configurable parameter columns and value distributions
- Controllable metric values (constant, correlated, random)
- NaN injection at specified rates
- Constant-metric regions for saturation testing
- Breakthrough values at specified rows

### Per-Reducer Test Strategy

**Sanity Reducer**
- Inject NaN rows at known rates; verify `remove_is` emitted above threshold
- Verify no pruning below threshold (boundary test)
- Verify `min_observations` gate: no action with insufficient trials

**Correlation Reducer**
- Build log with known parameter-metric correlations (one high, one near-zero)
- Verify low-correlation parameter pruned, high-correlation preserved
- Verify `sign_stability_threshold` filtering
- Verify `min_observations` gate: no action with fewer than 50 rounds

**Saturation Reducer**
- Build log with constant-metric regions for specific parameter values
- Verify partial pruning respects `retain_fraction`
- Verify window behavior (only recent `window_size` samples considered)
- Verify `min_samples_per_value` gate

**Focus Reducer**
- Build log with a breakthrough round (metric exceeds threshold)
- Verify state transition OFF → ON
- Verify parameter narrowing (`keep_between` / `keep_is` interventions)
- Verify timeout reversion after `focus_timeout` observations
- Verify variation injection when `inject_variations` is enabled

**Budget Reducer**
- Simulate time progression; verify `trim` fires when projected over budget
- Verify `check_after_pct` gate: no action before threshold
- Verify both trim strategies (`random`, `worst_first`)

**Manual Reducer**
- Already covered by existing `FeedbackController` tests
  (`tests/test_feedback_controller.py`)
