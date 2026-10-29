# Choose which trials run

A search chooses the next parameter combination to evaluate. Fix the admissible values, scoring population and trial budget before interpreting its winner. Python Research owns its SFD functions and strategy objects; Compiled Research obtains a declared pipeline and declares its search strategy in YAML and runs it through the CLI.

The [Python project](examples/python-research/README.md) supplies its search functions. The [Compiled project](examples/compiled-research/README.md) supplies complete YAML declarations and CLI commands. Each example retains its own population and execution assumptions.

## Prerequisites

- the matching package and recorded input bundle described in each project
- a bounded parameter domain and an explicit research objective
- the statistics extra for Sobol sampling
- an experiment directory and the required recording groups when continuing an interrupted run

## The search components retain distinct state

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
            'feedback': mixed_feedback(data_dir, output / 'feedback'), 'helper_diagnostics': preparation_helpers(bars, data_dir, output / 'helpers'), 'target_diagnostics': label_diagnostics(bars, data_dir, output / 'targets'), 'analysis': analyze_candidates(bars, output / 'analysis'), 'input_formats': read_source_formats(data_dir), 'multiclass': evaluate_three_classes(bars), 'rules': evaluate_rules(bars), 'linked': residual_research(bars, output / 'linked')}
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

The Python case compares grid, random, Sobol and shuffled traversal over two windows and two Ridge penalties. The Compiled case declares its traversal campaigns, a separately admitted factorized grid, and model-family experiments. Their budgets finish the bounded comparison used in this proof; they do not establish universal finite-domain coverage for a sampler. A search seed does not control model randomness.

<a id="seeded-finite-traversal" data-reader-heading="plain"></a>

## Strategy choice follows the research budget

Grid search enumerates the finite domain. A seeded shuffle changes joint candidate order while preserving a reproducible prefix. Increasing the permutation budget extends that prefix when the ordered domain and traversal identity remain fixed. Grid is the appropriate choice when completion must establish exhaustive coverage.

Random and Sobol strategies sample unseen discrete combinations under their respective contracts. Their seeds control candidate generation. Models, source snapshots, and fitted preprocessing retain separate sources of variation. Sampling, projection, deduplication, and feedback also affect which alternatives actually execute.

The four-candidate Python comparison completes grid, random, Sobol, and shuffled traversal over its bounded domain. Compiled YAML separately records grid/random/Sobol campaigns, the shuffled composable experiment, and a factorized shuffled grid. Their native artifacts identify the actual candidate identities and scores.

<a id="seeded-sobol-sampling" data-reader-heading="plain"></a>

### Sobol projects a seeded sequence onto declared values

Sobol sampling uses scrambled sequence coordinates to select from each parameter's ordered values. Multiple sequence points can map to one discrete candidate. Deduplication and bounded retries preserve the sampler's declared behavior, while grid traversal provides exhaustive finite-domain completion. The [Sobol implementation contract](Search-Reference.md#seeded-sobol-sampling) states the seed, dependency, projection, and recovery requirements.

## Candidate identity preserves declared alternatives

Parameter identity includes the sorted parameter keys and each declared choice. Distinct execution identities are required before search begins. Exact-choice mutations and priority injections use that identity, while occurrence identity distinguishes repeated executions of a configuration.

A restored strategy validates its saved domain and traversal state before accepting work. Changing the ordered choices or seed changes the continuation contract. The [finite traversal reference](Search-Reference.md#seeded-finite-traversal) owns exact identity rules, traversal versions, recovery compatibility, and prefix measurements. The [Sobol reference](Search-Reference.md#seeded-sobol-sampling) defines projection, bounded retries, sequence limits, and dependency compatibility.

## Factorization reuses independent preparation and signals

The Compiled factorized case crosses two LightGBM tree counts with two fee assumptions. Four economic results require only the admitted fitting work for each model choice. Replaying fees can reuse predictions because the fee axis is independent of the configured preparation and model signal:

<details>
<summary>Read the complete factorized experiment</summary>

<!-- example: search-compiled-factorized source=docs/examples/compiled-research/factorized.yaml region=declaration scenario=search interface=compiled role=executed -->
```yaml
schema_version: '1.0'
metadata:
  name: book_compiled_factorized
  mode: development
sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20241231.parquet
    split_dates:
      train_start: '2024-01-01'
      train_end: '2024-01-15'
      val_start: '2024-01-15'
      val_end: '2024-01-18'
      test_start: '2024-01-18'
      test_end: '2024-01-22'
    indicators:
    - func: limen.indicators.window_return
      params:
        period: 1
    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: ret_1
        quantile: 0.5
      transform_params:
        shift: -1
    backtest:
      fee_bps: '{fee}'
      slip_bps: 3.0
      notional_rate: 0.5
    scaler:
      class: limen.scalers.RobustScaler
    reference_architecture: limen.sfd.reference_architecture.lightgbm_binary
  params:
    fee:
    - 7.0
    - 9.0
    n_estimators:
    - 4
    - 8
    n_jobs:
    - 1
    random_state:
    - 42
    deterministic:
    - true
    force_row_wise:
    - true
    early_stopping_rounds:
    - 0
    subsample_freq:
    - 0
    subsample:
    - 1.0
    colsample_bytree:
    - 1.0
uel:
  output_path: compiled-factorized
  factorize: true
  n_permutations: 4
  search_strategy:
    type: grid
    shuffle: true
    seed: 13
  output_format: csv
  checkpoint_interval: 1
```

</details>

Admission requires a deterministic reuse declaration for every relevant stage and an immutable compatible grid. Preparation and signal caches are bounded separately at 128 entries. Supported training preparation reuse has a narrower target, scaler, and feature contract than signal reuse. Cache eviction affects repeated work while preserving the required research result.

Factorization uses snapshot execution. Walk-forward, multiple native workers, active feedback, allocation, explicit exits, and validation objectives have their own execution paths. A resumed factorized run validates its prefix and starts reuse caches empty. The [factorization contract](Universal-Experiment-Loop.md#factorized-snapshot-grids) gives exact admission and comparison requirements.

## Workers parallelize admitted candidate fits

Native workers reconstruct supported components from unchanged compiled YAML and frozen hashed input frames. The parent selects candidates, applies feedback, commits completed results in order, and writes artifacts. Interrupted execution preserves the committed contiguous prefix and its matching checkpoint.

Arbitrary caller-owned Python functions execute with one worker. Native worker admission includes per-round preparation, thread controls, source identity, and recording. The [workflow table](Experiment-Manifest.md#supported-research-workflows) connects these requirements to folds, recovery, and protected research. The [complete workflow](End-to-End-Workflow.md#choose-parallel-execution-deliberately) demonstrates the installed CLI worker and resume commands.

## Keep the completed search identifiable

The candidate domain defines which alternatives were eligible. The completed log shows which actually ran; reducers can change that population. Keep those distinctions when reading the best result. [Queue and recovery reference](Search-Reference.md) defines persisted state, stopping, resumed execution and optional search-strategy extension.

## Read the completed campaign

The Python campaign reports its explicit seed and the helper families actually exercised. The Compiled campaign records traversal, the reduced budget and its individual YAML helper campaigns. Full per-candidate rows remain in the native records; these are population and execution counts, not a significance test.

<!-- result: search-python-coverage interface=python scenario=search fields=seed,helper_diagnostics.counts -->

<!-- result: search-compiled-coverage interface=compiled scenario=search fields=traversals,budget_rounds,helper_diagnostics -->

## Continue the search decision

Continue to [reducers and feedback](Reducers-And-Feedback.md) to see an intervention change the remaining work. Then [retain and recover](Run-And-Recover.md) the experiment and its unfinished queue. The [CLI](Command-Line-Interface.md) and [UEL](Universal-Experiment-Loop.md) appendices provide the complete command and Python contracts.
