# Search, queue and recovery reference

The [Python project](examples/python-research/README.md) constructs its search around caller-owned functions. The [Compiled project](examples/compiled-research/README.md) declares search and persistence in YAML and uses the CLI. This appendix defines the queue, saved state and component extension contracts behind the [search chapter](Advanced-Search.md).

## Optional persistence and feedback

Both project searches include a separate budget-controlled run. It requests four trials and stops after three completed rounds. The same controller writes the trim into its audit and checkpoint; the assertions check the actual queue state.

<!-- example: search-python-budget source=docs/examples/python-research/scenarios.py region=adaptive_search scenario=search interface=python role=executed -->
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

<!-- example: search-compiled-budget source=docs/examples/compiled-research/adaptive.yaml region=budget_config scenario=search interface=compiled role=executed -->
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

Use feedback when changing the remaining search belongs to the research design. A reduced budget changes which observations of model performance are collected. It does not establish that the resulting selection is better.

## What `experiment_dir` stores

Python Research persists these files when `experiment_dir` is set. The Compiled CLI writes them beneath its resolved YAML output path:

| File | Meaning |
|---|---|
| `results.csv` | streaming round log |
| `round_data.jsonl.gz` | round params and payloads selected by the [recording policy](Universal-Experiment-Loop.md#recording-policy) |
| `checkpoint.json` | search state for resume |
| `audit.jsonl` | feedback-cycle audit trail |
| `metadata.json` | experiment metadata used by `Trainer` |

The [controller extension reference](Developer/Pruning-Strategies.md#manual-intervention) describes caller-owned callback and file controls. Compiled Research declares its adaptive controls in YAML.

## Resume flow

Python/MSQ can resume from a saved checkpoint when the applicable [recording groups](Universal-Experiment-Loop.md#recording-policy) were enabled on the initial run, and the original strategy and configuration are reconstructed:

Recovery tests interrupt after completed rounds, resume the saved queue and compare it with uninterrupted execution. The Compiled entry point is `limen run --resume RESULTS_DIR`; its YAML records the initial recording and checkpoint policy.

<!-- example: search-python-recovery source=docs/examples/python-research/scenarios.py region=recovery scenario=recovery interface=python role=executed -->
```python
def recovery(data_dir, output):
    bars = hourly(data_dir, 1440)
    space = {'window': [4, 12], 'alpha': [0.1, 1.0]}

    def model(data, round_params):
        fitted = project.fit(data['train'], round_params)
        forecasts = fitted.predict(data['validation'].select('momentum'))
        return {'validation_net_return': project.backtest(data['validation'], forecasts), '_preds': forecasts}

    sfd = SimpleNamespace(__name__=__name__, params=lambda: space, prep=recordable_prep, model=model)

    def stop_after_two(log, msq):
        if log.height == 2:
            signal.raise_signal(signal.SIGINT)

    def loop(callback=None, directory='experiment'):
        return UniversalExperimentLoop(data=bars, sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)), experiment_dir=output / directory, checkpoint_interval=1, feedback_interval=1, intra_callback=callback)

    recording: dict[str, bool] = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)
    interrupted = loop(stop_after_two)
    interrupted.run('python-recovery', n_permutations=4, workers=1, progress_bar=False, recording=recording)
    completed_before_resume = project.completed_log(interrupted).height
    resumed = loop()
    resumed.run('python-recovery', n_permutations=4, workers=1, progress_bar=False, recording=recording, resume=True)
    with gzip.open(output / 'experiment/round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    full = loop(directory='uninterrupted')
    full.run('python-recovery', n_permutations=4, workers=1, progress_bar=False, recording=recording)
    if not project.completed_log(resumed).drop('execution_time').equals(project.completed_log(full).drop('execution_time')):
        raise ValueError('Resumed Python results differ from uninterrupted research')
    return {'interrupted_rounds': completed_before_resume, 'rounds': project.completed_log(resumed).height, 'round_indices': [row['_round_index'] for row in records], 'round_ids': [row['round_id'] for row in records], 'fitted_state_restored': False, 'uninterrupted_equivalence': True,
            'captured': captured_state(bars, output / 'captured')}
```

<!-- example: search-compiled-recovery source=docs/examples/compiled-research/run.sh region=recovery scenario=recovery interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/recovery.yaml
limen run --no-progress-bar --resume results/dev/compiled-recovery
```

In the shutdown-and-resume example:

- the first phase stopped after `2` completed rounds
- `round_data.jsonl.gz` and `results.csv` each contained `2` entries
- the resumed phase finished the remaining work
- the final files contained `4` rows with `_round_index` values `0, 1, 2, 3`; CSV `id` and JSONL `round_id` contain the corresponding 64-character SHA-256 parameter hashes

For resume to work cleanly, keep these stable:

- the same `experiment_dir`
- the same `SearchStrategy` type
- the same parameter content
- the same configured reducer stack
- the initial recording policy, with predictions/alignment and any applicable event ledger or walk-forward folds retained; see the [recording policy](Universal-Experiment-Loop.md#recording-policy)

Ordinary serial recovery restores successful parameter values from their recorded round occurrences, preserving string categories such as `"001"` and recorded empty strings or nulls. Failed rounds have no typed parameter record: their CSV cells use the same supplied scalar `context_params` first, then the restored checkpoint domain, as type hints. Supply the historical context unchanged when resuming a failed prefix. Omitted or changed context, ambiguous failed empty-string/null cells, and unrecorded heterogeneous parameter history cannot be reconstructed exactly. Committed CSV fields retain their original text during recovery.

An unfinished UEL recovery refuses custom filters and named filters without a
supported, valid declarative descriptor, before executing a round or changing run
artifacts. Use supported `set_filter(..., filter_type=..., filter_params=...)`
descriptors for constraints that must survive recovery; otherwise start a new
experiment directory. Sampling descriptors must retain their saved hash rule.

Recovery can still load a checkpoint that cannot draw another candidate: its saved
cap or trim budget is exhausted, or its grid and injection queue are exhausted.
The saved cap applies even if the resume call supplies a larger one. A completed
serial run can still require continuation when its last checkpoint precedes its
final results. Standalone MSQ restoration keeps its warning and gives the caller
the opportunity to re-register filters before iterating.

Shutdown checkpoints preserve filter metadata at the committed frontier. Recovery
also checks existing intervention history for filters erased by older rewind
behavior. Checkpoints with neither filter metadata nor intervention evidence cannot
reveal lost constraints. This does not repair earlier draws made without filters
or reconstruct non-deterministic callbacks after the checkpoint.

## Capabilities shared with CLI

The CLI compiles YAML and constructs the search around MSQ. Python Research supplies its own functions and strategy objects. Both use the queue's candidate identities, feedback and checkpoint semantics.

Python Research owns its arbitrary SFD functions and fitted models. Recovery restores the search state and recorded results, rather than arbitrary fitted Python objects. Saving a custom run does not give it Trainer reconstruction: Trainer and CLI resume require a reconstructible YAML definition saved as `yaml_reference`. [Trainer](Trainer.md) describes that boundary.

## Optional custom `SearchStrategy`

The [identity compatibility design](Developer/Search-Identity-Compatibility.md)
records the boundary for a future protocol change. It does not change the
current search or recovery format.

Implement a strategy only when the built-ins do not meet the experiment's search requirements. A strategy must:

- hold a reference to a shared `ParamDomain`
- yield dictionaries of round parameters
- expose `get_state()` and `set_state()` for checkpointing
- react to domain changes if its internal state depends on the current domain

Use the shipped [GridStrategy implementation](https://github.com/Vaquum/Limen/blob/main/limen/experiment/param_search/grid_strategy.py) as the exhaustive-search reference. It checks the inherited seen set before yielding candidates, rebuilds its cursor after domain changes, and saves/restores its enumeration state. Custom strategies must preserve this deduplication behavior across feedback and resume; resetting a cursor alone can repeat completed combinations.

## Seeded finite traversal

The Python search function constructs its shuffled grid with `shuffle=True, seed=42`. The Compiled declarations record every traversal choice: the random and Sobol definitions use seed 7, while the factorized grid declares `factorize: true`, `shuffle: true` and seed 13. These choices are executed directly from the displayed files.

Fresh seeded shuffled grids use version 2: eight balanced Feistel rounds with SHAKE-256, cycle-walked into the finite index domain. Candidate generation never materializes the Cartesian product or a permutation list. Cycle-walk cost varies. Sorted parameter keys and declared level order define the mixed-radix mapping. The seed changes the permutation; the round budget does not, so increasing it extends the same prefix. Exhaustion stops normally.

Each parameter's choices must have distinct execution identities. YAML, `ParamDomain`, legacy `ParamSpace` and restored domain states reject collisions before accepting the domain. Identity uses the existing sorted JSON encoding with `default=str` and SHA-256; mapping key order, tuple/list pairs or an object and its identical string representation can collide. Values such as `1`, `1.0` and `True` retain distinct identities. Inputs are never silently deduplicated, and existing hashes and seeded traversal remain unchanged. A checkpoint with colliding choices requires a corrected declaration and a new run. Exact-choice removal, retention, membership and injection use that same identity. Numeric reducer decisions are rebound to the unique original domain choice when log concatenation widens integers to floats; ambiguous numeric groups fail explicitly. Reducer checkpoint pairs retain their existing format and distinguish exact identities. Numeric range operations retain their comparison semantics. Injecting an existing domain identity returns `False` without notification; priority `MSQ.inject` retains its explicit queue semantics.

This is joint candidate traversal, rather than shuffling levels before nested enumeration. Tests measure marginal and pair-projection prefix coverage over pinned unequal domains/seeds against affine and nested-enumeration controls. They establish those cases, not a universal prefix-coverage guarantee or uniform random sampling. A search seed does not seed models.

Version and ordered-domain identity are saved in strategy state; existing metadata independently pins version 2. Factorized resume rejects seed/version/ordered-domain changes before rewriting artifacts and restarts reuse caches empty. Versionless old integer-seeded shuffled checkpoints retain their affine version-1 order on the ordinary path. Seedless native shuffle remains legacy version 1, without a reproducible resume guarantee, and is not admitted by factorization. Existing unshuffled behavior is unchanged.

Factorized traversal retains its [snapshot admission and cache limits](Universal-Experiment-Loop.md#factorized-snapshot-grids). Live pruning, callbacks and interventions remain excluded; use a new immutable manifest for a follow-up search.

## Seeded Sobol sampling

Install the existing statistics extra with `pip install 'vaquum-limen[stats]'` for SciPy `>=1.15,<2`.

Each traversal comparison uses four candidate combinations. The Python variants use seed `42`; Compiled random and Sobol use `7`, and its factorized grid uses `13`; the assertions compare completed candidate identities and retain the resulting scores. The Compiled scenario also runs its separate model-family diagnostics.

Python Research uses `SobolStrategy(domain, seed=42)`, imported from `limen.experiment.param_search`. The seed is a required literal, non-boolean, nonnegative integer. Scrambling is always enabled; `scramble`, `shuffle` and other unsupported Sobol fields are rejected. This selects a candidate generator for the existing MSQ path, without changing model execution or its randomness. `factorize=True` still requires GridStrategy and rejects Sobol.

The sampler uses [SciPy's Sobol engine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html) with LMS+shift scrambling and 30 bits. Sorted parameter keys define coordinates; declared value-list order defines equal-width bins. Coordinate `u` selects `values[floor(u * len(values))]`, preserving the declared value and its existing parameter hash. No Cartesian product is materialized. Domain value changes update these bins while retaining the raw sequence position and accepted hashes. Existing MSQ budgets, filters, feedback and explicit repeat injections keep their semantics.

Different points can select the same discrete combination. Accepted combinations are deduplicated; failure to find a novel combination within 1,000 proposals raises an error, without switching samplers. The engine also raises when its `2**30` points are exhausted. Use grid search when exhaustive traversal is required. Discrete projection, arbitrary prefixes, filters and deduplication remove Sobol balance guarantees; no convergence gain, speedup or universal finite-domain coverage is promised.

Checkpoint state pins the seed, stream format, raw point position, generated count, ordered current domain and SciPy version. Resume reconstructs the engine and fast-forwards to that position, using the existing domain and accepted-hash restoration. Changed or unsupported state fails before persisted artifacts are rewritten; keep the same SciPy version when resuming. The existing [recording requirements](Search-Reference.md#resume-flow) still apply.

The project proof uses the pinned hourly observations. [Sobol protocol tests](../tests/test_sobol_search.py) retain additional state, mutation and refusal cases; the [scenario assertions](examples/tests/scenario_oracles.py) check the project runs.
