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

The separate metric-driven campaigns exercise all four alternatives. Python compares 30 real fitted candidates before inspecting correlations and parameter importance. Each feedback run starts a fresh queue. The Python sanity case marks measured execution times above its deliberately strict zero-second allowance as suggestions. The Compiled case removes a calibration mode whose chosen score is unavailable; its missing-rate threshold distinguishes that mode from the mixed population. Saturation uses a declared variance threshold; focus narrows around an observed winner; correlation distinguishes applied removals from advisory suggestions. These small populations demonstrate control behavior, not statistical reliability.

### Python metric-driven controls

<!-- example: reducers-python-metric-controls source=docs/examples/python-research/analysis_cases.py region=reducer_config scenario=search interface=python role=executed -->
```python
def metric_reducers():
    return {
        'sanity': SanityReducer(metric='validation_rmse', min_observations=2, execution_time_threshold=0., timeout_rate_threshold=0.),
        'saturation': SaturationReducer(metric='validation_rmse', cv_threshold=.1, min_samples_per_value=2, retain_fraction=.5),
        'focus': FocusReducer(metric='validation_rmse', maximize=False, breakthrough_threshold=1., min_observations=5, focus_timeout=2, variation_count=3),
        'correlation': CorrelationReducer(metric='validation_rmse', maximize=False, min_observations=10, n_boot=30,
            prune_threshold=1., sign_stability_threshold=0., random_state=42),
    }
```

### Compiled metric-driven controls

The four manifests declare those controls independently. Their native `audit.jsonl` files retain every applied intervention, suggestion and source error; the checkpoint retains the resulting queue. Run them with the CLI:

<!-- example: reducers-compiled-metric-controls source=docs/examples/compiled-research/run.sh region=reducers scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar \
  compiled-research/reducer_sanity.yaml \
  compiled-research/reducer_saturation.yaml \
  compiled-research/reducer_focus.yaml \
  compiled-research/reducer_correlation.yaml \
  compiled-research/analysis.yaml \
  compiled-research/rule_contracts.yaml
```

The [reducer reference](Developer/Pruning-Strategies.md#constructor-reference) lists each constructor and its validation rules. In YAML, its arguments belong under the reducer's `params` mapping.

## When to use which reducer

The five reducers address distinct resource and search decisions. Their intervention policy determines which remaining candidates execute.

| Reducer | Decision addressed | Intervention | Reversible |
|---|---|---|---|
| `BudgetReducer` | the sweep must finish inside a walltime or permutation cap | trims the remaining queue once, when the budget is projected to overrun | no — a trim is permanent |
| `SanityReducer` | parameter values produce unusable observations | removes values whose metric is mostly null; flags zero-metric, timeout, and warning-heavy values as suggestions | removals no; suggestions are advisory |
| `CorrelationReducer` | metric relationships guide the remaining domain | removes wrong-direction values and suggests retention of low-impact values | removals no; suggestions advisory |
| `FocusReducer` | an observed improvement supplies a temporary search center | narrows parameters around the best round and injects nearby variations | yes — named filters restore after a timeout |
| `SaturationReducer` | a value's metric has stopped varying and no longer earns full budget | down-samples saturated values, keeping a monitoring sample | yes — restores a value if it starts varying again |

`SanityReducer` identifies unusable observations; its removals still change the search population. `BudgetReducer` is a resource cap. The other three change *what* the search explores, so they carry more risk of steering the run — introduce them one at a time and read `audit.jsonl` to see their effect.

Reducers combine, but with no conflict resolution. Within a feedback trigger they run in the order of the list they were constructed in, each analyzing the same pre-trigger `MSQ`; their interventions are dispatched afterwards with last-write-wins at the queue. `FocusReducer` and `SaturationReducer` use namespaced filter keys, so they do not clobber each other, but two exploitation reducers can still fight (e.g. a `BudgetReducer` trim shrinking the queue that a `FocusReducer` is injecting into). See [Feedback cycle order](#feedback-cycle-order) and the [controller reference](Developer/Pruning-Strategies.md#feedback-cycle-order).

## Tune against a declared metric

Choose the metric and its direction before interpreting an intervention. Missingness, time budgets, variance and a breakthrough score answer different questions. The [constructor reference](Developer/Pruning-Strategies.md#constructor-reference) owns the complete defaults, domains and refusal rules for each reducer. The project cases set their thresholds explicitly and retain the actual actions.

## Forest analysis supplies a separate suggestion mechanism

`CorrelationReducer(analyzer="random_forest")` uses held-out surrogate parameter importance. Its input can contain numeric, Boolean, and categorical scalar parameters. The analysis requires at least 20 distinct successful configurations and the declared minimum observation count.

The held-out surrogate must reach `min_model_score` before its parameter suggestions become available. Repeated shuffles estimate how much each parameter contributes to prediction of the recorded metric. Low-importance parameters receive advisory keep suggestions under the configured screening rule. This mode does not infer a monotonic direction from a nonlinear relationship.

The forest is an analyzer choice within `CorrelationReducer`. The five reducer types remain Budget, Sanity, Correlation, Focus, and Saturation. [Research evidence](Research-Evidence.md) develops the separate parameter-importance result and its interpretation; the [constructor contract](Developer/Pruning-Strategies.md#constructor-reference) defines analyzer arguments.

## Feedback preserves the order of applied changes

A feedback cycle evaluates reducers, then the Python callback, then the intervention file. The audit records the source and operation for each applied change. Reducer observations describe the pre-trigger queue, while the eventual queue reflects the ordered interventions.

A priority injection can deliberately repeat a candidate or bypass ordinary filters. Parameter identity identifies the configuration; occurrence identity identifies each admitted execution. The audit and completed log therefore answer different questions about the campaign. Search changes affect future work without rewriting completed observations.

## Audit trail

With an experiment directory, every feedback trigger writes one JSONL entry to `audit.jsonl`: round, applied interventions, suggestions, source errors and compact queue state. `checkpoint.json` retains the resulting domain and remaining budget. Read the completed rows as well: a domain mutation does not retroactively change observations already collected.

## Declare the control before execution

`feedback_interval` schedules the feedback cycles. Each reducer `type` is a registry key; `params` supplies its constructor arguments. `limen validate` rejects unknown types and fields before execution. Use the budget declaration above as the starting point, then introduce metric-driven reducers one at a time.

The [developer appendix](Developer/Pruning-Strategies.md) covers the reducer registry, constructor signatures, callbacks, file controls and controller verification.

## Read next

Continue to [retention and recovery](Run-And-Recover.md) to keep the result and unfinished queue together.
