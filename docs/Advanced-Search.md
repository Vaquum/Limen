# Advanced Search

New runs use compressed round artifacts; existing JSONL runs remain readable. See the [round storage format](Universal-Experiment-Loop.md#round-storage-format).
Search strategies control candidate generation for both CLI and Python/MSQ experiments. The CLI builds the strategy from YAML. Python callers construct a strategy over a `ParamDomain` and pass it to UEL; UEL creates MSQ around that strategy.

This page covers:

- choosing built-in grid or random search
- optional persistence and resumability
- optional feedback that mutates the remaining search space
- implementing a custom Python search strategy

Start with the [CLI workflow](Command-Line-Interface.md) for YAML or the [Python/MSQ run](Universal-Experiment-Loop.md#direct-python-run) for an SFD. Neither route requires adaptive feedback or a custom strategy implementation.

## Prerequisites

- a working SFD and bounded data source
- a `ParamDomain` and `SearchStrategy` for direct Python use
- an `experiment_dir` only when saved artifacts or checkpoints are needed

## The core pieces

| Piece | What it owns |
|---|---|
| `ParamDomain` | the mutable legal values for each parameter |
| `SearchStrategy` | how new combinations are generated from the domain |
| `MSQ` | the mutable search queue that applies interventions, filters, injections, and trims |
| `FeedbackController` | reducer, callback, and file-driven feedback cycles |
| `CheckpointManager` | durable save and restore of persisted MSQ run state |

The important mental model is:

1. `SearchStrategy` proposes combinations
2. `MSQ` decides what is still legal
3. feedback mutates `MSQ` and sometimes the underlying `ParamDomain`
4. checkpoints and full recording preserve enough state to continue later with `resume=True`

## Choose a built-in strategy

| Strategy | Behavior | CLI declaration |
|---|---|---|
| `GridStrategy` | Enumerate combinations deterministically; execute up to the permutation budget. | `uel.search_strategy.type: grid` |
| `RandomStrategy` | Lazily sample unseen combinations; accepts a candidate-generation seed. | `uel.search_strategy.type: random`, optionally with `seed` |

CLI defaults to random search. Grid search is unshuffled by default. From 5.31.0, YAML and Python accept a seeded finite shuffle; YAML `shuffle: true` requires an integer seed. A shuffle declaration on random search is rejected. In Python, select either strategy over the SFD's parameter space:

```python
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy

domain = ParamDomain(sfd.params())
strategy = GridStrategy(domain)
```

Use the [Python/MSQ example](Universal-Experiment-Loop.md#direct-python-run) for the complete data-loading and execution flow. Random search deduplicates repeated draws and can fail near domain exhaustion; use grid search when exhaustive coverage is required. A search seed does not control model or preparation randomness.

## Seeded finite traversal

```yaml
uel:
  factorize: true
  search_strategy:
    type: grid
    shuffle: true
    seed: 42
```

Native equivalent: `GridStrategy(domain, shuffle=True, seed=42)` with the existing `run(..., factorize=True)` switch. The [recorded example](examples/factorized_shuffle.yaml) runs from the repository root with `limen run docs/examples/factorized_shuffle.yaml`.

Fresh seeded shuffled grids use version 2: eight balanced Feistel rounds with SHAKE-256, cycle-walked into the finite index domain. Candidate generation never materializes the Cartesian product or a permutation list. Cycle-walk cost varies. Sorted parameter keys and declared level order define the mixed-radix mapping. The seed changes the permutation; the round budget does not, so increasing it extends the same prefix. Exhaustion stops normally; existing full-parameter hash deduplication still collapses duplicate-valued levels.

This is joint candidate traversal, rather than shuffling levels before nested enumeration. Tests measure marginal and pair-projection prefix coverage over pinned unequal domains/seeds against affine and nested-enumeration controls. They establish those cases, not a universal prefix-coverage guarantee or uniform random sampling. A search seed does not seed models.

Version and ordered-domain identity are saved in strategy state; existing metadata independently pins version 2. Factorized resume rejects seed/version/ordered-domain changes before rewriting artifacts and restarts reuse caches empty. Versionless old integer-seeded shuffled checkpoints retain their affine version-1 order on the ordinary path. Seedless native shuffle remains legacy version 1, without a reproducible resume guarantee, and is not admitted by factorization. Existing unshuffled behavior is unchanged.

Factorized traversal retains its [snapshot admission and cache limits](Universal-Experiment-Loop.md#factorized-snapshot-grids). Live pruning, callbacks and interventions remain excluded; use a new immutable manifest for a follow-up search.

## Optional persistence and feedback

Add `experiment_dir` to save round artifacts and checkpoints. Add reducers only when adapting the search fits the research design. This example adds a budget reducer to a built-in grid:

```python
import limen

from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.experiment.reducer import BudgetReducer

domain = ParamDomain(limen.sfd.random_binary.params())
strategy = GridStrategy(domain)

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

In the bundled budget-reducer example, this requested `6` permutations but finished with:

- `4` rows in `results.csv`
- `4` entries in `round_data.jsonl.gz`
- `1` feedback-cycle entry in `audit.jsonl`
- a checkpoint saved after round `3`

The reducer intervention that caused the trim was:

```python
{'op': 'trim', 'target_count': 4, 'reason': 'budget trim to 4 total permutations'}
```

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

```python
uel.run(
    experiment_name='advanced-budget',
    n_permutations=6,
    resume=True,
)
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

CLI runs already use MSQ and provide the same search, persisted round predictions, checkpoint/resume, and YAML-configured reducers. These are capabilities of the shared engine, not a separate execution route.

Python additionally permits custom SFD functions, custom `SearchStrategy` implementations, and in-process callbacks. An `experiment_dir` alone does not make those custom runs compatible with [Trainer](Trainer.md) or CLI resume: those tools require a reconstructible YAML definition saved as `yaml_reference`.

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
