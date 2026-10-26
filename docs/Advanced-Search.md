# Choose which trials run

A search chooses the next parameter combination to evaluate. Fix the admissible values, scoring population and trial budget before interpreting its winner. Python Research owns its SFD functions and strategy objects; Compiled Research obtains a declared pipeline and declares its search strategy in YAML and runs it through the CLI.

The [Python project](examples/python-research/README.md) supplies its search functions. The [Compiled project](examples/compiled-research/README.md) supplies complete YAML declarations and CLI commands. Each example retains its own population and execution assumptions.

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
            'feedback': mixed_feedback(data_dir, output / 'feedback'), 'helper_diagnostics': preparation_helpers(bars, data_dir, output / 'helpers'), 'target_diagnostics': label_diagnostics(bars, data_dir, output / 'targets'), 'analysis': analyze_candidates(bars, output / 'analysis'), 'input_formats': read_source_formats(data_dir), 'multiclass': evaluate_three_classes(bars), 'rules': evaluate_rules(bars)}
```

<!-- example: search-compiled-traversal source=docs/examples/compiled-research/run.sh region=search scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar \
  compiled-research/search.yaml \
  compiled-research/search_random.yaml \
  compiled-research/search_sobol.yaml \
  compiled-research/factorized.yaml \
  compiled-research/adaptive.yaml
```

The Python case compares grid, random, Sobol and shuffled traversal over two windows and two Ridge penalties. The Compiled case compares the same traversal choices, a separately admitted factorized grid and declared model families. Their budgets finish the bounded comparison used in this proof; they do not establish universal finite-domain coverage for a sampler. A search seed does not control model randomness.

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

## Keep the completed search identifiable

The candidate domain defines which alternatives were eligible. The completed log shows which actually ran; reducers can change that population. Keep those distinctions when reading the best result. [Queue and recovery reference](Search-Reference.md) defines persisted state, stopping, resumed execution and optional search-strategy extension.

## Read the completed campaign

The Python campaign reports its explicit seed and the helper families actually exercised. The Compiled campaign records traversal, the reduced budget and its individual YAML helper campaigns. Full per-candidate rows remain in the native records; these are population and execution counts, not a significance test.

<!-- result: search-python-coverage interface=python scenario=search fields=seed,helper_diagnostics.counts -->

<!-- result: search-compiled-coverage interface=compiled scenario=search fields=traversals,budget_rounds,helper_diagnostics -->

## Continue the search decision

Continue to [reducers and feedback](Reducers-And-Feedback.md) to see an intervention change the remaining work. Then [retain and recover](Run-And-Recover.md) the experiment and its unfinished queue. The [CLI](Command-Line-Interface.md) and [UEL](Universal-Experiment-Loop.md) appendices provide the complete command and Python contracts.
