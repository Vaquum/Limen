# Choose which trials run

A search chooses the next parameter combination to evaluate. Fix the admissible values, scoring population and trial budget before interpreting its winner. Python Research owns its SFD functions and strategy objects; Compiled Research obtains a declared pipeline and can construct its strategy through the CLI or Python.

The [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) each provide a `search` scenario. Their examples below are exact functions from those projects. Use the complete project entry points, matching wheel and recorded input bundle; their preparation helpers and declarations remain in the complete source.

## Prerequisites

- the matching package and recorded input bundle described in each project
- a bounded parameter domain and an explicit research objective
- the statistics extra for Sobol sampling
- an experiment directory and the required recording groups when continuing an interrupted run

## The core pieces

| Piece | What it owns |
|---|---|
| `ParamDomain` | the mutable legal values for each parameter |
| `SearchStrategy` | how new combinations are generated from the domain |
| `MSQ` | the mutable search queue that applies interventions, filters, injections, and trims |
| `FeedbackController` | reducer, callback, and file-driven feedback cycles |
| `CheckpointManager` | durable save and restore of persisted MSQ run state |

A candidate passes through four stages:

1. `SearchStrategy` proposes combinations
2. `MSQ` decides what is still legal
3. feedback mutates `MSQ` and sometimes the underlying `ParamDomain`
4. checkpoints and full recording preserve enough state to continue later with `resume=True`

## Choose a built-in strategy

| Strategy | Behavior | CLI declaration |
|---|---|---|
| `GridStrategy` | Enumerate combinations deterministically; execute up to the permutation budget. | `uel.search_strategy.type: grid` |
| `RandomStrategy` | Lazily sample unseen combinations; accepts a candidate-generation seed. | `uel.search_strategy.type: random`, optionally with `seed` |
| `SobolStrategy` | Project seeded, scrambled Sobol points onto declared parameter values. | `uel.search_strategy.type: sobol` with a required `seed` |

CLI defaults to random search. Grid search is unshuffled by default. From 5.31.0, YAML and Python accept a seeded finite shuffle; YAML `shuffle: true` requires an integer seed. A shuffle declaration on other strategies is rejected. The two project functions make those choices explicit:


<!-- example: search-python-traversal source=docs/examples/python-research/scenarios.py region=search scenario=search interface=python role=executed -->
```python
def search(data_dir, output):
    bars = hourly(data_dir, 1440)
    space = {'window': [4, 12], 'alpha': [0.1, 1.0]}
    runs = {}
    for name in ('grid', 'random', 'sobol', 'shuffled'):
        domain = ParamDomain(space)
        strategy = (GridStrategy(domain, shuffle=True, seed=42) if name == 'shuffled'
                    else {'grid': GridStrategy, 'random': RandomStrategy, 'sobol': SobolStrategy}[name](domain, **({} if name == 'grid' else {'seed': 42})))

        def model(data, round_params):
            fitted = project.fit(data['train'], round_params)
            return {'validation_net_return': project.backtest(data['validation'], fitted.predict(data['validation'].select('momentum')))}

        loop = UniversalExperimentLoop(data=bars, sfd=SimpleNamespace(params=lambda: space, prep=project.prep, model=model), search_strategy=strategy)
        loop.run(str(output / name), n_permutations=4, workers=1, progress_bar=False)
        runs[name] = project.completed_log(loop).select('window', 'alpha', 'validation_net_return').to_dicts()
    return {'runs': runs, 'common_baselines': common_baselines(data_dir, output / 'common-baselines'), 'preparation': preparation(data_dir, output / 'preparation'), 'seed': 42, 'adaptive': adaptive_search(data_dir, output / 'adaptive'),
            'feedback': mixed_feedback(data_dir, output / 'feedback')}
```

<!-- example: search-compiled-traversal source=docs/examples/compiled-research/scenarios.py region=search scenario=search interface=compiled role=executed -->
```python
def search(data_dir, output):
    results = {}
    for kind in ('grid', 'random', 'sobol', 'shuffled', 'factorized'):
        config = configuration('factorized' if kind == 'factorized' else 'search', data_dir)
        config['uel']['search_strategy'] = {'type': kind if kind in ('random', 'sobol') else 'grid'}
        if kind in ('random', 'sobol'):
            config['uel']['search_strategy']['seed'] = 42
        if kind in ('shuffled', 'factorized'):
            config['uel']['search_strategy'].update(shuffle=True, seed=42)
        loop = execute(config, output / kind, factorize=kind == 'factorized')
        results[kind] = research_project.completed_log(loop).select('id', 'accuracy', *(['n_estimators', 'fee'] if kind == 'factorized' else ['C'])).to_dicts()
    return {'runs': results, 'common_baselines': common_baselines(data_dir, output / 'common-baselines'), 'preparation': preparation(data_dir, output / 'preparation'), 'seed': 42, 'families': model_comparisons(data_dir, output / 'families'),
            'adaptive': adaptive_search(data_dir, output / 'adaptive'),
            'feedback': mixed_feedback(data_dir, output / 'feedback')}
```

The Python case compares grid, random, Sobol and shuffled traversal over two windows and two Ridge penalties. The Compiled case compares the same traversal choices, a separately admitted factorized grid and declared model families. Their budgets finish the bounded comparison used in this proof; they do not establish universal finite-domain coverage for a sampler. A search seed does not control model randomness.

## Seeded finite traversal

The Python search function constructs its shuffled grid with `shuffle=True, seed=42`. The Compiled search function declares the same strategy settings and enables factorization only for its separate factorized case. The unmodified driver records these choices alongside the base configuration; running that YAML file alone does not reproduce the driver's traversal choices.

Fresh seeded shuffled grids use version 2: eight balanced Feistel rounds with SHAKE-256, cycle-walked into the finite index domain. Candidate generation never materializes the Cartesian product or a permutation list. Cycle-walk cost varies. Sorted parameter keys and declared level order define the mixed-radix mapping. The seed changes the permutation; the round budget does not, so increasing it extends the same prefix. Exhaustion stops normally; existing full-parameter hash deduplication still collapses duplicate-valued levels.

This is joint candidate traversal, rather than shuffling levels before nested enumeration. Tests measure marginal and pair-projection prefix coverage over pinned unequal domains/seeds against affine and nested-enumeration controls. They establish those cases, not a universal prefix-coverage guarantee or uniform random sampling. A search seed does not seed models.

Version and ordered-domain identity are saved in strategy state; existing metadata independently pins version 2. Factorized resume rejects seed/version/ordered-domain changes before rewriting artifacts and restarts reuse caches empty. Versionless old integer-seeded shuffled checkpoints retain their affine version-1 order on the ordinary path. Seedless native shuffle remains legacy version 1, without a reproducible resume guarantee, and is not admitted by factorization. Existing unshuffled behavior is unchanged.

Factorized traversal retains its [snapshot admission and cache limits](Universal-Experiment-Loop.md#factorized-snapshot-grids). Live pruning, callbacks and interventions remain excluded; use a new immutable manifest for a follow-up search.

## Seeded Sobol sampling

Install the existing statistics extra with `pip install 'vaquum-limen[stats]'` for SciPy `>=1.15,<2`.

Each traversal comparison uses four candidate combinations. Both seeded strategies use a literal `42`; the assertions compare completed candidate identities and retain the resulting scores. The Compiled scenario also runs its separate model-family diagnostics.

Native equivalent: `SobolStrategy(domain, seed=42)`, imported from `limen.experiment.param_search`. The seed is a required literal, non-boolean, nonnegative integer. Scrambling is always enabled; `scramble`, `shuffle` and other unsupported Sobol fields are rejected. This selects a candidate generator for the existing MSQ path, without changing model execution or its randomness. `factorize=True` still requires GridStrategy and rejects Sobol.

The sampler uses [SciPy's Sobol engine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html) with LMS+shift scrambling and 30 bits. Sorted parameter keys define coordinates; declared value-list order defines equal-width bins. Coordinate `u` selects `values[floor(u * len(values))]`, preserving the declared value and its existing parameter hash. No Cartesian product is materialized. Domain value changes update these bins while retaining the raw sequence position and accepted hashes. Existing MSQ budgets, filters, feedback and explicit repeat injections keep their semantics.

Different points can select the same discrete combination. Accepted combinations are deduplicated; failure to find a novel combination within 1,000 proposals raises an error, without switching samplers. The engine also raises when its `2**30` points are exhausted. Use grid search when exhaustive traversal is required. Discrete projection, arbitrary prefixes, filters and deduplication remove Sobol balance guarantees; no convergence gain, speedup or universal finite-domain coverage is promised.

Checkpoint state pins the seed, stream format, raw point position, generated count, ordered current domain and SciPy version. Resume reconstructs the engine and fast-forwards to that position, using the existing domain and accepted-hash restoration. Changed or unsupported state fails before persisted artifacts are rewritten; keep the same SciPy version when resuming. The existing [recording requirements](#resume-flow) still apply.

The project proof uses the pinned hourly observations. [Sobol protocol tests](../tests/test_sobol_search.py) retain additional state, mutation and refusal cases; the [scenario assertions](examples/tests/scenario_oracles.py) check the project runs.

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

<!-- example: search-compiled-budget source=docs/examples/compiled-research/scenarios.py region=adaptive_search scenario=search interface=compiled role=executed -->
```python
def adaptive_search(data_dir, output):
    config = configuration('adaptive', data_dir)
    loop = execute(config, output)
    checkpoint = json.loads((output / 'checkpoint.json').read_text())['msq_state']
    return {'rows': research_project.completed_log(loop).select('id', 'C', 'accuracy').to_dicts(),
            'audit': [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()],
            'checkpoint_yielded': checkpoint['yielded_count'], 'checkpoint_remaining': checkpoint['trim_budget']}
```

Use feedback when changing the remaining search belongs to the research design. A reduced budget changes which observations of model performance are collected. It does not establish that the resulting selection is better.

## What `experiment_dir` stores

When `experiment_dir` is set, Python/MSQ writes:

| File | Meaning |
|---|---|
| `results.csv` | streaming round log |
| `round_data.jsonl.gz` | round params and payloads selected by the [recording policy](Universal-Experiment-Loop.md#recording-policy) |
| `checkpoint.json` | search state for resume |
| `audit.jsonl` | feedback-cycle audit trail |
| `metadata.json` | experiment metadata used by `Trainer` |
| `interventions.json` | optional input file polled by the feedback controller when the file exists |

Unlike the other files above, `interventions.json` is not produced by the run. It is an optional external control file that the feedback controller watches when `experiment_dir` is present.

## Resume flow

Python/MSQ can resume from a saved checkpoint when all four [recording groups](Universal-Experiment-Loop.md#recording-policy) were enabled on the initial run, and the original strategy and configuration are reconstructed:

The `recovery` scenarios interrupt after two completed rounds, reconstruct their original configuration and resume. The Python scenario also compares its completed results with an uninterrupted run:

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
    return {'interrupted_rounds': completed_before_resume, 'rounds': project.completed_log(resumed).height, 'round_indices': [row['_round_index'] for row in records], 'round_ids': [row['round_id'] for row in records], 'fitted_state_restored': False, 'uninterrupted_equivalence': True}
```

<!-- example: search-compiled-recovery source=docs/examples/compiled-research/scenarios.py region=recovery scenario=recovery interface=compiled role=executed -->
```python
def recovery(data_dir, output):
    config = configuration('recovery', data_dir)
    config['uel']['recording'] = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)

    def stop_after_two(log, msq):
        if log.height == 2:
            signal.raise_signal(signal.SIGINT)

    first = execute(config, output / 'experiment', callback=stop_after_two)
    resumed = execute(config, output / 'experiment', resume=True)
    with gzip.open(output / 'experiment/round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    selected = research_project.completed_log(resumed)['id'][0]
    sensors = Trainer(output / 'experiment').train([selected])
    full = execute(config, output / 'uninterrupted')
    if not research_project.completed_log(resumed).drop('execution_time').equals(research_project.completed_log(full).drop('execution_time')):
        raise ValueError('Resumed Compiled results differ from uninterrupted research')
    return {'interrupted_rounds': research_project.completed_log(first).height, 'rounds': research_project.completed_log(resumed).height, 'round_indices': [row['_round_index'] for row in records], 'round_ids': [row['round_id'] for row in records], 'reconstructed_sensors': len(sensors), 'uninterrupted_equivalence': True}
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

## Capabilities shared with CLI

The CLI constructs a compiled SFD and strategy around the same MSQ engine. Both projects use that engine's candidate identities, feedback controller and checkpoint format. Direct Python invocation can construct the shared objects explicitly; adopting a Manifest still makes the experiment Compiled Research.

Python Research owns its arbitrary SFD functions and fitted models. Recovery restores the search state and recorded results, rather than arbitrary fitted Python objects. Saving a custom run does not give it Trainer reconstruction: Trainer and CLI resume require a reconstructible YAML definition saved as `yaml_reference`. [Trainer](Trainer.md) describes that boundary.

## Optional custom `SearchStrategy`

Implement a strategy only when the built-ins do not meet the experiment's search requirements. A strategy must:

- hold a reference to a shared `ParamDomain`
- yield dictionaries of round parameters
- expose `get_state()` and `set_state()` for checkpointing
- react to domain changes if its internal state depends on the current domain

Use the shipped [GridStrategy implementation](https://github.com/Vaquum/Limen/blob/main/limen/experiment/param_search/grid_strategy.py) as the exhaustive-search reference. It checks the inherited seen set before yielding candidates, rebuilds its cursor after domain changes, and saves/restores its enumeration state. Custom strategies must preserve this deduplication behavior across feedback and resume; resetting a cursor alone can repeat completed combinations.

## Read next

- Continue to [Reducers And Feedback](Reducers-And-Feedback.md) for the intervention system that acts on the mutable search queue.
- Continue to [Command-Line Interface](Command-Line-Interface.md) for YAML execution through the same engine.
- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) for Python/MSQ execution and legacy compatibility reference.
- Continue to [Trainer](Trainer.md) for promotion of saved YAML experiments into reusable sensors.
