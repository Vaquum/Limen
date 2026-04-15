# Universal Experiment Loop

The Universal Experiment Loop (UEL) is Limen's experiment runner. It takes an SFD, executes parameter combinations, streams a CSV log, and then exposes post-run analysis surfaces such as confusion metrics and backtest results.

This is the page to read when you want to answer four practical questions:

- how to run a first experiment
- what `UniversalExperimentLoop` actually stores after a run
- when to use the simple run path versus the advanced artifact-rich path
- which run-time rules matter for manifest-driven and custom SFDs

## Two Execution Modes

UEL currently has two execution modes.

| Mode | How you enter it | Best for | What you get |
|---|---|---|---|
| standard run path | instantiate with `sfd=` and optionally `data=`, then call `run()` without a `search_strategy` | most public examples and straightforward sweeps | in-memory UEL artifacts plus a streaming CSV at `<experiment_name>.csv` |
| MSQ / artifact-rich path | instantiate with a concrete `search_strategy`, optionally `experiment_dir`, then call `run()` | advanced search flows, checkpointing, resumability, trainer workflows | `results.csv`, `round_data.jsonl`, checkpoints, audit trail, metadata, and in-memory UEL artifacts |

The standard run path is the right starting point for most readers. The advanced path is real and supported, but it is an extension-oriented surface built around `SearchStrategy`.

## First Real Run

This is the most reliable local example because it uses the repo test CSV through the new file-backed ingestion path.

```python
import limen
from limen.data import HistoricalData

historical = HistoricalData()
historical.get_any_file(str(HistoricalData.DEFAULT_TEST_FILE_PATH), n_rows=2000)

uel = limen.UniversalExperimentLoop(
    data=historical.data,
    sfd=limen.sfd.logreg_binary,
)

uel.run(
    experiment_name='logreg-first',
    n_permutations=4,
    prep_each_round=True,
    random_search=False,
)
```

After the run you can inspect:

```python
uel.experiment_log
uel.experiment_confusion_metrics
uel.experiment_backtest_results
```

On a live local run over the file-backed test data, that produced:

- `uel.experiment_log` with one row per round
- `uel.experiment_confusion_metrics` with one row per round
- `uel.experiment_backtest_results` with one row per round
- `uel.preds`, `uel.round_params`, and `uel._alignment` for round-level reconstruction

## Constructor Contract

```python
uel = limen.UniversalExperimentLoop(
    data=None,
    sfd=my_sfd,
    search_strategy=None,
    experiment_dir=None,
)
```

### Core constructor arguments

| Argument | Meaning |
|---|---|
| `sfd` | required SFD module |
| `data` | optional input dataframe; required for custom SFDs, optional for manifest-driven SFDs |
| `search_strategy` | advanced search hook; enables the MSQ execution path |
| `experiment_dir` | directory for stored experiment artifacts; meaningful with the advanced path |
| `pruning_strategies`, `feedback_interval`, `checkpoint_interval`, `intra_callback` | advanced MSQ controls |

### Data behavior

- If the SFD exposes `manifest()` and you do not pass `data=`, UEL fetches data from the manifest.
- If the SFD is custom and has no manifest, `data=` is required.
- For manifest-driven SFDs, the data source used by auto-fetch depends on `LOOP_ENV`.

## `run()` Contract

```python
uel.run(
    experiment_name='my_experiment',
    n_permutations=100,
    prep_each_round=True,
)
```

### Core run arguments

| Argument | Meaning |
|---|---|
| `experiment_name` | run name and CSV path stem; `my_experiment` writes `my_experiment.csv` |
| `n_permutations` | number of rounds to execute |
| `prep_each_round` | whether prep runs every round; required for manifest-driven SFDs |
| `random_search` | random versus deterministic parameter generation on the standard path |
| `context_params` | extra static keys injected into every round |
| `params`, `prep`, `model` | optional overrides for the standard path |
| `resume` | resume from checkpoint in the advanced path |

### Manifest-driven rules

If the SFD uses `manifest()`:

- `prep_each_round=True` is required
- `prep=` and `model=` overrides are not allowed
- `params=` override is allowed

### Custom-SFD rules

If the SFD uses custom `prep()` and `model()`:

- `data=` must be provided when you instantiate UEL
- `prep_each_round` can be `True` or `False`, depending on whether prep depends on round params
- `params=`, `prep=`, and `model=` overrides are available on the standard path

## What UEL Stores After A Run

The most important attributes are:

| Attribute | Meaning |
|---|---|
| `uel.data` | dataframe used by the run |
| `uel.params` | parameter space in use |
| `uel.round_params` | actual parameter values used for each round |
| `uel.experiment_log` | main round-by-round experiment log |
| `uel.experiment_confusion_metrics` | confusion-style analysis derived from predictions |
| `uel.experiment_backtest_results` | backtest-style analysis derived from predictions |
| `uel.preds` | stored test predictions per round |
| `uel.scalers` | fitted scalers captured from prep or manifest scaling |
| `uel._alignment` | alignment metadata per round |
| `uel._log` | internal `Log` object for deeper analysis |

### Alignment metadata

Each entry in `uel._alignment` includes:

- `missing_datetimes`
- `first_test_datetime`
- `last_test_datetime`

This is what lets downstream analysis stay aligned with the actual test window seen by a round.

### Deeper post-run analysis

UEL constructs a `Log` instance automatically at the end of a successful run. That gives you access to methods such as:

- `uel._log.permutation_prediction_performance(round_id=0)`
- `uel._log.permutation_confusion_metrics('price_change', round_id=0)`
- `uel.experiment_parameter_correlation('auc')`

## Standard Path Versus Artifact-Rich Path

### Standard Path

The standard path writes a streaming CSV at:

```text
<experiment_name>.csv
```

and keeps the full run state in memory on the `uel` object.

This is the path to use for:

- first experiments
- docs examples
- quick local research loops
- simple parameter sweeps

### Artifact-Rich Path

If you instantiate UEL with a concrete `search_strategy` and an `experiment_dir`, Limen stores structured artifacts there.

| File | Meaning |
|---|---|
| `results.csv` | streaming round log |
| `round_data.jsonl` | round params, predictions, and alignment metadata |
| `checkpoint.json` | checkpoint state for resumption |
| `audit.jsonl` | feedback-controller audit trail |
| `interventions.json` | optional external intervention file polled by the feedback controller if you create it |
| `metadata.json` | experiment metadata used by `Trainer` |

This path is what powers checkpointing, resumability, and the [Trainer](Trainer.md) workflow.

### Important scope note

Limen ships built-in strategies (`GridStrategy`, `RandomStrategy`) and the `SearchStrategy` abstraction for writing your own. The advanced path is available now with the built-in strategies or a custom implementation from your codebase.

## One Real Advanced Run

The UEL-facing part of an advanced run looks like this once you already have a concrete `SearchStrategy`:

```python
import limen

from limen.experiment.param_domain import ParamDomain
from limen.experiment.reducer import BudgetReducer

domain = ParamDomain(limen.sfd.random_binary.params())
strategy = MiniGrid(domain)  # see Advanced Search for a complete minimal implementation

uel = limen.UniversalExperimentLoop(
    sfd=limen.sfd.random_binary,
    search_strategy=strategy,
    pruning_strategies=[
        BudgetReducer(max_permutations=4, check_after_pct=0.25),
    ],
    feedback_interval=2,
    checkpoint_interval=3,
    experiment_dir='advanced-budget',
)

uel.run(
    experiment_name='advanced-budget',
    n_permutations=6,
)
```

On a live local run in this repo, that advanced run:

- requested `6` permutations
- finished with `4` rows in `results.csv`
- wrote `4` entries to `round_data.jsonl`
- wrote `1` entry to `audit.jsonl`
- saved a checkpoint after round `3`

That behavior came from a reducer-triggered trim during the feedback cycle, not from a simple early stop.

## Resume In Practice

Resumption belongs only to the advanced path:

```python
uel.run(
    experiment_name='advanced-budget',
    n_permutations=6,
    resume=True,
)
```

In a live shutdown-and-resume run in this repo:

- the first phase stopped after `2` completed rounds
- `results.csv` and `round_data.jsonl` each contained `2` entries
- the resumed phase finished the remaining rounds
- the final stored round ids were `0, 1, 2, 3`

Use the same `experiment_dir`, strategy type, and reducer configuration when resuming.

For the full advanced-search contract, continue to [Advanced Search](Advanced-Search.md) and [Reducers And Feedback](Reducers-And-Feedback.md).

## Common Errors

### Manifest-driven runs require `prep_each_round=True`

If you run a manifest-driven SFD with `prep_each_round=False`, UEL currently raises the exact runtime string `prep_each_round must be True for manifest-driven SFMs`. The `SFM` wording there is legacy runtime text, not current docs terminology. Set `prep_each_round=True`.

### Manifest-driven runs cannot override `prep` or `model`

If you pass `prep=` or `model=` to `run()` for a manifest-driven SFD, UEL currently raises `Cannot override prep/model when SFM has manifest`. That `SFM` wording is also legacy runtime text. Put the logic into the manifest instead, or switch to the custom SFD path.

### Custom SFDs require explicit `data=`

If you instantiate UEL with a custom SFD and omit `data=`, UEL raises `data parameter required for custom SFDs using custom functions approach`.

### Resuming requires a search strategy

Resumption belongs to the advanced path. If you call `run(..., resume=True)` without a search strategy, UEL raises `resume=True is only supported with a search_strategy`.

## Read Next

- Continue to [Log](Log.md) to understand the analysis surfaces built on top of UEL results.
- Continue to [Experiment Manifest](Experiment-Manifest.md) if you are building a manifest-driven SFD.
- Continue to [Trainer](Trainer.md) if you are using the artifact-rich path and want to retrain finished rounds into sensors.
