# Reducers and feedback

Reducers and feedback are optional controls inside the MSQ execution path shared by Python Research and Compiled Research. They let a run react to its own results instead of sweeping the original domain unchanged; using a search strategy does not require enabling them.

Compiled Research declares its controls under `uel.pruning_strategies` in YAML. Python Research constructs reducer objects alongside its caller-owned functions. Both retain the resulting queue changes in their audit trail and checkpoint.

The [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) each demonstrate a four-candidate search with a three-candidate budget.

## Prerequisites

Choose a search strategy, a positive `feedback_interval` and enough completed rounds for the selected reducer's observation threshold. Persist the experiment directory to retain its audit trail and checkpoint.

## Feedback cycle order

At each feedback trigger, reducers analyze the completed observations and the queue state. Their returned changes apply in declaration order. Read the audit trail before interpreting the reduced search population: later changes can overwrite earlier ones.

The controller also exposes [callback and file controls](Developer/Pruning-Strategies.md#manual-intervention) for Python and component integrations. Those extension mechanisms have their own ordering and failure behavior.

## Built-in reducers

The project budget-only runs request four trials and stop at three completed rows. The proof checks the trim intervention, completed results and checkpoint budget in both interfaces:

<!-- example: feedback-python-budget source=docs/examples/python-research/scenarios.py region=adaptive_search scenario=search interface=python role=executed -->
```python
def adaptive_search(data_dir, output):
    space = {'window': [4, 12], 'alpha': [.1, 1.]}

    def model(data, round_params):
        fitted = project.fit(data['train'], round_params)
        return {'validation_net_return': project.backtest(data['validation'], fitted.predict(data['validation'].select('momentum')))}

    sfd = SimpleNamespace(__name__=__name__, params=lambda: space, prep=recordable_prep, model=model)
    loop = UniversalExperimentLoop(data=hourly(data_dir, 1440), sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)),
                                   experiment_dir=output, checkpoint_interval=1, feedback_interval=1,
                                   pruning_strategies=[BudgetReducer(max_permutations=3)])
    loop.run('python-adaptive-search', n_permutations=4, workers=1, progress_bar=False)
    checkpoint = json.loads((output / 'checkpoint.json').read_text())['msq_state']
    return {'rows': project.completed_log(loop).select('window', 'alpha', 'validation_net_return').to_dicts(),
            'audit': [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()],
            'checkpoint_yielded': checkpoint['yielded_count'], 'checkpoint_remaining': checkpoint['trim_budget']}
```

<!-- example: feedback-compiled-budget source=docs/examples/compiled-research/adaptive.yaml region=budget_config scenario=search interface=compiled role=executed -->
```yaml
uel:
  output_path: compiled-adaptive
  n_permutations: 4
  search_strategy:
    type: grid
  output_format: csv
  checkpoint_interval: 1
  pruning_strategies:
  - type: budget
    params:
      max_permutations: 3
  feedback_interval: 1
```

These small runs exercise `BudgetReducer`. The four metric-driven alternatives below need enough observations for their respective analyses; a three-round example does not demonstrate their statistical reliability.

The [reducer reference](Developer/Pruning-Strategies.md#constructor-reference) lists each constructor and its validation rules. In YAML, its arguments belong under the reducer's `params` mapping.

## When to use which reducer

The five reducers answer different questions. Pick by the problem you have, not by wanting "more feedback".

| Reducer | Reach for it when | What it does | Reversible |
|---|---|---|---|
| `BudgetReducer` | the sweep must finish inside a walltime or permutation cap | trims the remaining queue once, when the budget is projected to overrun | no — a trim is permanent |
| `SanityReducer` | you want to stop wasting rounds on broken parameter values | hard-removes values whose metric is mostly null; flags zero-metric, timeout, and warning-heavy values as suggestions | removals no; suggestions are advisory |
| `CorrelationReducer` | you want to prune values that move the metric the wrong way or barely at all | removes wrong-direction values, suggests keeps for low-impact ones | removals no; suggestions advisory |
| `FocusReducer` | a breakthrough score has appeared and you want to exploit near the winner | narrows each parameter around the best round and injects nearby variations | yes — named filters, snaps back after a timeout |
| `SaturationReducer` | a value's metric has stopped varying and no longer earns full budget | down-samples saturated values, keeping a monitoring sample | yes — restores a value if it starts varying again |

`SanityReducer` identifies unusable observations; its removals still change the search population. `BudgetReducer` is a resource cap. The other three change *what* the search explores, so they carry more risk of steering the run — introduce them one at a time and read `audit.jsonl` to see their effect.

Reducers combine, but with no conflict resolution. Within a feedback trigger they run in the order of the list they were constructed in, each analyzing the same pre-trigger `MSQ`; their interventions are dispatched afterwards with last-write-wins at the queue. `FocusReducer` and `SaturationReducer` use namespaced filter keys, so they do not clobber each other, but two exploitation reducers can still fight (e.g. a `BudgetReducer` trim shrinking the queue that a `FocusReducer` is injecting into). See [Feedback cycle order](#feedback-cycle-order) and the [controller reference](Developer/Pruning-Strategies.md#feedback-cycle-order).

## Tuning guidelines

Metric-driven reducers take `metric`, the column they reason over. Every reducer takes `active` (default `True`). The knobs below are the ones that change behavior most; each reducer validates its ranges and raises `ValueError` on bad input.

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

## Audit trail

With an experiment directory, every feedback trigger writes one JSONL entry to `audit.jsonl`: round, applied interventions, suggestions, source errors and compact queue state. `checkpoint.json` retains the resulting domain and remaining budget. Read the completed rows as well: a domain mutation does not retroactively change observations already collected.

## Declare the control before execution

`feedback_interval` schedules the feedback cycles. Each reducer `type` is a registry key; `params` supplies its constructor arguments. `limen validate` rejects unknown types and fields before execution. Use the budget declaration above as the starting point, then introduce metric-driven reducers one at a time.

The [developer appendix](Developer/Pruning-Strategies.md) covers the reducer registry, constructor signatures, callbacks, file controls and controller verification.

## Read next

Continue to [search and recovery](Advanced-Search.md) for candidate generation and checkpointing, [the experiment loop](Universal-Experiment-Loop.md) for feedback scheduling, and [Trainer](Trainer.md) for reconstructible Compiled experiments.
