# `limen.sfd.loop` — Temporary Loop Web UI Integration

**Status: temporary scaffolding.** This subpackage will be deleted when
RFC-1005 (YAML compiler) lands in Limen. Do not import from this subpackage
in any other Limen module.

## What this is

A compiler that turns a Loop web UI experiment design payload (JSON) into a
Limen-compatible SFD object. The SFD can then run through `UniversalExperimentLoop`
like any other foundational SFD.

## Quick start

```python
import json
from limen.sfd.loop import LoopSFD
from limen import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy

payload = json.loads(open('loop_logreg_binary.json').read())
sfd = LoopSFD(payload)

uel = UniversalExperimentLoop(
    sfd=sfd,
    search_strategy=RandomStrategy(ParamDomain(sfd.params()), seed=42),
    experiment_dir='/tmp/exp',
    feedback_interval=1,
)
uel.run(experiment_name='/tmp/exp/run', n_permutations=10)
```

Or via the CLI:

```bash
python -m limen.sfd.loop.run path/to/payload.json --out /tmp/exp --n 10
```

## Module layout

| File | Purpose |
|---|---|
| `loop_sfd.py` | `LoopSFD` class — the compiler |
| `registry.py` | Name → callable registries (auto-built from `__all__`) |
| `meta.py` | `LABEL_TARGET_COLUMNS` and `SCALER_NAME_MAP` (hand-maintained) |
| `reference_defaults.py` | Default model hyperparams from foundational SFDs |
| `progress.py` | `make_progress_callback` for live progress reporting |
| `run.py` | CLI entry point |

## Design notes

- **Source of truth split**: Loop payload provides the *data pipeline*
  (indicators, features, transforms, labels, scaler, split, reference
  architecture name). The reference architecture's *model hyperparams* come
  from the matching foundational SFD or a constructor override — never from
  the payload, which has case issues (`input_logreg_binary_c` vs `C`) and
  may not include them at all.
- **Drops `selectedItems`**: the payload's `selectedItems` and
  `*_selected_items` keys are UI form metadata and are filtered out.
- **Single scaler**: uses `scaler.scalingMethod` (e.g. `"LinearScaler"`),
  mapped to the lowercase `SCALER_REGISTRY` key via `SCALER_NAME_MAP`.
- **Progress via `intra_callback`**: writes `{experiment_dir}/progress.json`
  every round (when `feedback_interval=1`). A backend can poll this file.

## Quarantine rules (so removal is trivial)

This subpackage is designed to be deleted in a single PR with no collateral
damage. To preserve that:

1. **All code lives under `limen/sfd/loop/`**. No files outside this dir.
2. **Single test file**: `tests/test_loop_sfd.py`. No `conftest.py` changes.
3. **Sample payload lives at** `tests/fixtures/loop_logreg_binary.json` so it
   is part of the removable set.
4. **Zero modifications to existing Limen code** other than a single
   self-marking `CHANGELOG.md` entry.
5. **No new dependencies**.
6. **No CLI entry points in `pyproject.toml`** — invoke via `python -m`.
7. **One-way dependency**: this subpackage imports from other Limen modules,
   but nothing in Limen imports from `limen.sfd.loop`.
8. **Not exported from `limen/__init__.py`** — only reachable via
   `from limen.sfd.loop import LoopSFD`.

## Removal procedure

When RFC-1005 lands and this is no longer needed:

```bash
git rm -rf limen/sfd/loop/
git rm tests/test_loop_sfd.py
git rm tests/fixtures/loop_logreg_binary.json
# Optional: remove the temporary CHANGELOG entry
```

Then verify clean removal:

```bash
grep -r "limen.sfd.loop" limen/ tests/   # should return nothing
python tests/run.py                        # should still pass
```

## Known limitation: payload `transforms[]` are ignored

In this iteration, the payload's `transforms[]` array is **intentionally
ignored**. `LoopSFD` logs an `INFO` message naming the skipped transforms.

Reasoning: transforms like `mad_transform` need to run in the target
context (after labels, on eager DataFrames) and the exact wiring + scaler
compatibility (`mad_transform` + `LinearScaler` produces NaN that survives
polars' `drop_nulls`) needs more validation against representative payloads
before it can be enabled by default.

`TRANSFORM_REGISTRY` is still built and exported from `registry.py` so the
next iteration can wire transforms in without further plumbing changes.
