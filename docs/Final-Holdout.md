# Reserve final observations

Reserve a project's final time window before running research. Supported experiments, Trainer refits and Log reconstruction then admit only native bars ending at or before the earliest active reservation. The exclusion applies before selectors, splitting and fitted preparation, including caller-supplied or reassigned frames.

This records local workflow protection. It cannot establish that a person, another project or an external process has never read these observations. Physical source materialization can exceed the admitted research prefix; provenance records both quantities.

## Prerequisites

Use a Git project containing `limen.toml`, supported native OHLC data and serial built-in research. Preserve its local history as described below before cloning or restoring the project.

## Reservation lifecycle

Run from a Git project containing `limen.toml`:

```python
from pathlib import Path
from limen.experiment.holdout import reserve_holdout, release_holdout, withdraw_holdout

root = Path.cwd()
reservation_id = reserve_holdout(
    root, '2024-12-01', '2025-01-01', interval_seconds=3600,
)
```

Dates must be valid `YYYY-MM-DD` UTC midnights. Windows are half-open and immutable. The interval must be an integer dividing one UTC day; the source must supply that native interval without local aggregation. Active windows are disjoint and share one interval. Their earliest start excludes the entire subsequent suffix from research, including gaps between later reservations.

`withdraw_holdout(root, reservation_id)` retires an unused reservation. Protected preparation makes withdrawal unavailable; use `release_holdout(root, reservation_id)` to lift exclusion explicitly. Release preserves the interval as historical exposure, and the same interval cannot be reserved again. An unused withdrawn interval may be reserved again with a new ID. Neither operation deletes history.

Each change advances the project revision. Reconstruct existing loops, Trainers and compiled manifests after a change; stale objects refuse continued work. A running research operation holds an admission lease, so a concurrent reservation change fails after five seconds rather than taking effect midway through research. Concurrent research can still write its history. A process crash releases the lease but leaves any started record with an unknown outcome.

## Supported research

Use ordinary serial MLManifest or RuleBasedManifest preparation. Native callers may pass `project_root=root` to UniversalExperimentLoop or Trainer; CompiledSFD also accepts it. CLI run/resume/profile discovers the root from the YAML or experiment location even when invoked from elsewhere. Conflicting explicit, experiment, compiled and current-directory roots are errors. Binding is private metadata and does not change canonical YAML identity.

While a reservation is active, the workflow refuses:

- parallel, factorized, resumed or walk-forward execution and profiling;
- custom SFD functions, replaced prep/model methods or callbacks;
- alternate test sources, formed/dollar/event bars, event/funding execution and alternate execution sources;
- source row limits, source/interval search parameters, unsupported file formats and resampling;
- split dates extending beyond the research cutoff.

Without active reservations, ordinary numerical paths and supported modes remain available. A discovered Git project uses an admission lease even before first reservation, preventing first-reservation creation from racing an already-running operation. No protected provenance is added to ordinary zero-active artifacts.

Supported source declarations are local Parquet via HistoricalData.get_any_file or unchanged default hosted get_spot_klines native datasets at 60, 900, 1800, 3600, 7200 or 14400 seconds. Hosted defaults have a provider-native contract. Local files and supplied frames have caller-declared native support; an evenly spaced grid alone cannot prove their venue origin or intrabar completeness.

Research preserves naturally missing whole native intervals and records their gap bounds. It does not pad, resample or join separated reserved windows. Duplicate, reversed, off-grid or invalid OHLC rows fail. An hourly declaration over recorded 15-minute rows fails. Local/hosted source scans can materialize a broader snapshot; only admitted rows reach selectors and preparation. Direct HistoricalData calls and arbitrary caller I/O remain outside this workflow.

## Provenance

Guarded metadata and round records retain project UUID/revision, record IDs, source identity/digest, materialized/admitted/selected bounds and counts, selector parameters and actual split geometry. ML preparation records ordered feature names/types before scaler/PCA and at model input. Trainer records the actual refit separately and carries that record in private fitted metadata on its Sensor. Rule-based records identify the absence of a fitted model schema. Legacy artifacts and file-only Log views do not acquire protected-history claims.

Log replay verifies the bound revision and admitted geometry; it refuses a changed population rather than silently shortening retained returns. Ordinary Sensor inference is not clipped by research reservations.

## Preserve history

`manifests/holdout-project.json` identifies the project UUID. Commit and preserve this marker. The local Git exclude file excludes both SQLite databases and their sidecars, plus `manifests/holdout/` evidence. A tracked history/evidence path is refused. Git worktrees use Git's resolved exclude path.

The durable history is `manifests/holdout.sqlite3`. Coordination uses a separate `manifests/holdout-lease.sqlite3`; holding its read lease does not block history writes. A marker without history fails closed, including in a clone whose local history has not been restored. Preserve a quiescent copy of marker, history and any evidence directory separately from `limen backup`; restore the same UUID together. The lease database has no exposure records.

This is local coordination, not tamper-proof storage or shared-filesystem coordination. Deleting both marker and history defeats local identity-loss detection. Unknown earlier or external exposure remains unknown.

## Final assessment

This release supplies reservation, research exclusion and preparation provenance. Freezing fitted selections and allocating final assessments are delivered separately by [S980B](https://github.com/Vaquum/Limen/issues/995); those APIs are not available yet. Existing [acceptance reports](Benchmark.md) remain research diagnostics. Keep their walk-forward limitations separate from a final holdout claim.
