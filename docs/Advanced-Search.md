# Advanced Search

Advanced search is Limen's artifact-backed experiment path. It adds a mutable parameter domain, a search strategy abstraction, checkpointing, resumability, and mid-run feedback on top of the normal Universal Experiment Loop.

This page covers:

- a search strategy that is not the legacy `ParamSpace` sweep
- a durable `experiment_dir` with resumable artifacts
- mid-run interventions that can mutate the remaining search space
- promotion-ready runs for [Trainer](Trainer.md)

For a first experiment, stay on the standard UEL path. Advanced search is real and supported, but it is the extension-oriented layer of Limen.

## The core pieces

| Piece | What it owns |
|---|---|
| `ParamDomain` | the mutable legal values for each parameter |
| `SearchStrategy` | how new combinations are generated from the domain |
| `MSQ` | the mutable search queue that applies interventions, filters, injections, and trims |
| `FeedbackController` | reducer, callback, and file-driven feedback cycles |
| `CheckpointManager` | durable save and restore of advanced-run state |

The important mental model is:

1. `SearchStrategy` proposes combinations
2. `MSQ` decides what is still legal
3. feedback mutates `MSQ` and sometimes the underlying `ParamDomain`
4. checkpoints preserve enough state to continue later with `resume=True`

## Minimal custom `SearchStrategy`

Limen ships two built-in strategies (`GridStrategy` for exhaustive search, `RandomStrategy` for lazy sampling) and the `SearchStrategy` abstraction for custom strategies. A strategy must at minimum:

- hold a reference to a shared `ParamDomain`
- yield dictionaries of round parameters
- expose `get_state()` and `set_state()` for checkpointing
- react to domain changes if its internal state depends on the current domain

This is the smallest practical exhaustive strategy shape:

```python
import itertools

from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import SearchStrategy


class MiniGrid(SearchStrategy):
    @property
    def is_finite(self):
        return True

    def __init__(self, domain: ParamDomain, *, seed: int | None = None):
        super().__init__(domain, seed=seed)
        self._combos = self._build_combos()
        self._index = 0

    def _build_combos(self):
        params = self._domain.params
        keys = sorted(params.keys())
        return [
            dict(zip(keys, vals, strict=True))
            for vals in itertools.product(*(params[k] for k in keys))
        ]

    def __next__(self):
        if self._index >= len(self._combos):
            raise StopIteration
        combo = self._combos[self._index]
        self._index += 1
        self._generated_count += 1
        return combo

    def on_domain_changed(self, _domain, _changed_params):
        self._combos = self._build_combos()
        self._index = 0

    def get_state(self):
        return {'index': self._index, 'generated_count': self._generated_count}

    def set_state(self, state):
        self._index = state['index']
        self._generated_count = state['generated_count']
```

## First artifact-backed run

```python
import limen

from limen.experiment.param_domain import ParamDomain
from limen.experiment.reducer import BudgetReducer

domain = ParamDomain(limen.sfd.random_binary.params())
strategy = MiniGrid(domain)

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

On a live local run over the bundled test data, this requested `6` permutations but finished with:

- `4` rows in `results.csv`
- `4` entries in `round_data.jsonl`
- `1` feedback-cycle entry in `audit.jsonl`
- a checkpoint saved after round `3`

The reducer intervention that caused the trim was:

```python
{'op': 'trim', 'target_count': 4, 'reason': 'budget trim to 4 total permutations'}
```

## What `experiment_dir` stores

When `experiment_dir` is set, advanced search writes:

| File | Meaning |
|---|---|
| `results.csv` | streaming round log |
| `round_data.jsonl` | round params, stored predictions, and alignment metadata |
| `checkpoint.json` | search state for resume |
| `audit.jsonl` | feedback-cycle audit trail |
| `metadata.json` | experiment metadata used by `Trainer` |
| `interventions.json` | optional input file polled by the feedback controller when the file exists |

Unlike the other files above, `interventions.json` is not produced by the run. It is an optional external control file that the feedback controller watches when `experiment_dir` is present.

## Resume flow

Resumption belongs to the advanced path only.

```python
uel.run(
    experiment_name='advanced-budget',
    n_permutations=6,
    resume=True,
)
```

In a live local shutdown-and-resume run in this repo:

- the first phase stopped after `2` completed rounds
- `round_data.jsonl` and `results.csv` each contained `2` entries
- the resumed phase finished the remaining work
- the final files contained `4` rows with round ids `0, 1, 2, 3`

For resume to work cleanly, keep these stable:

- the same `experiment_dir`
- the same `SearchStrategy` type
- the same parameter content
- the same configured reducer stack

## What advanced search adds beyond standard UEL

Compared to the standard path, advanced search adds:

- mutable pruning and focus during a run
- stored round-level predictions and alignment data
- auditability of feedback cycles
- resumability after interruption
- the artifact contract that [Trainer](Trainer.md) expects

What it does not change:

- SFDs still define the research unit
- manifests still govern split-first prep
- post-run analysis still flows through `Log`

## Read next

- Continue to [Reducers And Feedback](Reducers-And-Feedback.md) for the intervention system that acts on the mutable search queue.
- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) for the broader run contract around standard and advanced execution.
- Continue to [Trainer](Trainer.md) for promotion of finished advanced runs into reusable sensors.
