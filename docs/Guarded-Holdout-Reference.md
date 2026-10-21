# Guarded holdout reference

The [Compiled project](examples/compiled-research/README.md) reserves and assesses its final window through YAML and CLI. The [Python project](examples/python-research/README.md) owns a separate final-observation exclusion in ordinary Python. The [book chapter](Final-Holdout.md) develops those research decisions.

This appendix documents native reservation, exposure, freezing and assessment contracts. Its Python API names are optional integration and component reference; they do not add a Python step to Compiled Research.

## Reservation lifecycle

The guarded scenario runs `limen reserve compiled-research/final/holdout.yaml` before research. The returned identity is supplied to `limen assess`. The API names below document the underlying reservation lifecycle for component authors.

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

Compiled profiling holds the lease through loading, every sample evaluation and cleanup, including failed or interrupted calls. Active reservations still refuse profiling before data work and remain unused and withdrawable.

Supported source declarations are local Parquet or CSV via HistoricalData.get_any_file or unchanged default hosted get_spot_klines native datasets at 60, 900, 1800, 3600, 7200 or 14400 seconds. CSV datetimes follow the ordinary loader: naive ISO stamps are taken as UTC, recorded offsets are converted, and unparseable stamps are rejected with the other native-bar checks. Hosted defaults have a provider-native contract. Local files and supplied frames have caller-declared native support; an evenly spaced grid alone cannot prove their venue origin or intrabar completeness.

Research preserves naturally missing whole native intervals and records their gap bounds. It does not pad, resample or join separated reserved windows. Duplicate, reversed, off-grid or invalid OHLC rows fail. An hourly declaration over recorded 15-minute rows fails. Local/hosted source scans can materialize a broader snapshot; only admitted rows reach selectors and preparation. Direct HistoricalData calls and arbitrary caller I/O remain outside this workflow.

## Provenance

Guarded metadata and round records retain project UUID/revision, record IDs, source identity/digest, materialized/admitted/selected bounds and counts, selector parameters and actual split geometry. ML preparation records ordered feature names/types before scaler/PCA and at model input. Trainer records the actual refit separately and carries that record in private fitted metadata on its Sensor. Rule-based records identify the absence of a fitted model schema. Legacy artifacts and file-only Log views do not acquire protected-history claims.

Log replay verifies the bound revision and admitted geometry; it refuses a changed population rather than silently shortening retained returns. Ordinary Sensor inference is not clipped by research reservations.

## Preserve history

`manifests/holdout-project.json` identifies the project UUID. Commit and preserve this marker. The local Git exclude file excludes both SQLite databases and their sidecars, plus `manifests/holdout/` evidence. A tracked history/evidence path is refused. Git worktrees use Git's resolved exclude path.

The durable history is `manifests/holdout.sqlite3`. Coordination uses a separate `manifests/holdout-lease.sqlite3`; holding its read lease does not block history writes. A marker without history fails closed, including in a clone whose local history has not been restored. Preserve a quiescent copy of marker, history and any evidence directory separately from `limen backup`; restore the same UUID together. The lease database has no exposure records.

This is local coordination, not tamper-proof storage or shared-filesystem coordination. Deleting both marker and history defeats local identity-loss detection. Unknown earlier or external exposure remains unknown.

## Final assessment

The Compiled scenario freezes the validated Trainer selection with `freeze_holdout_selection()`, computes `final_holdout_report()`, and verifies the retained report through `read_holdout_report()`.

Context is an explicit UTC date strictly before the final start. Optional `start`/`end` narrow the reservation, retaining half-open bounds. Final start must follow every member's test end. The entire context and final interval must have complete native support. Each final bar uses the same context anchor and only observations through that bar. Each member predicts once; scenarios share that tape. No fitting, calibration, threshold selection or model promotion occurs.

Supported selections are exact ordinary YAML-backed binary Sensor/Cohort objects with fitted LogRegBinary or LightGBMBinary, ordered Trainer schemas and matching completed project research records. Cached pipelines must still match their YAML declaration. Configured calibration must already be fitted; Python-backed Sensors, legacy provenance, custom predictors, nonordinary splits, formed bars, trade contracts and decoder lookback above one are refused before allocation. Original selection mutation cannot change the frozen copy. One handle permits sequential reuse; concurrent use is refused.

Each YAML scenario requires `fee_bps`, `slip_bps`, `notional_rate`, `execution_lag_bars`, `take_profit_bps` and `stop_loss_bps`. Fees and slippage are finite nonnegative basis points; notional is in `(0,1]`; execution lag is a positive integer; TP/SL are null or finite positive basis points. Boolean numbers are refused. Order and duplicates are retained. Execution starts flat, excludes context positions, applies lag and forces terminal exit using existing snapshot economics. Raw positions are 0/1 and raw returns are unit-notional fractions. Deployed positions and gross/net/cost returns multiply raw values by notional once; summary returns already use that deployed fraction.

Optional `evaluation_target` accepts an initialized exact NextBarUpTarget, NextBarDownTarget, or ThresholdBinaryTarget with a fixed threshold over open, high, low, or close and horizon one. Labels are computed after predictions from final rows only; the terminal label is unavailable. Accuracy, precision, recall, FPR and AUC disclose denominators/support. Missing labels, zero denominators, one-class AUC, absent probabilities and undefined return summaries are null with an explicit reason. Majority-vote probabilities remain null.

## Attempts and evidence

Allocation commits a serialized project assessment index and planned support before reading context or final data. Every subsequent error carries its attempt ID and retains failed exposure; a killed process may leave `started` with unknown outcome. Warm-up failure consumes an attempt. A new frozen selection with an earlier context or narrower window creates another attempt; it does not erase previous access. Any attempt prevents withdrawal; explicit release remains available.

History migrates atomically from v1 to v2 under the reservation lease. Older reservation-only installations refuse v2. The shared admission lease covers evaluation and finalization, preventing a reservation change mid-assessment while allowing separately serialized allocations.

Reports retain allocation-time prior attempts/reservations and overlap classes. Failed and started attempts count equally with completed attempts. `initial-recorded` is conditional on completed hosted-native local research evidence, an active unreleased reservation and no previous planned support intersecting the final window. Overlap makes later assessments `subsequent`; release makes them `post-release`; local/caller training data is `unknown`. Shared unreserved context alone does not defeat initial-recorded status. External exposure is always unknown.

Each completed attempt stores `manifests/holdout/<attempt_id>/rows.parquet` and `report.json`. Rows retain context and final OHLC, original source columns/types, row hashes, member and aggregate predictions, availability reasons, optional labels and raw/deployed scenario ledgers. Other source columns use a `source_` prefix, repeated until unique when a name would collide; the report retains the original-to-retained name map. Strict JSON metrics contain no NaN/Infinity. History stores both artifact hashes; retrieval verifies completed status, identities, hashes and schemas without loading market data or running models. Partial files from failed writes remain unadvertised evidence. Preserve the evidence directory with the marker and history during backup.

Specification and fitted-state digests pin the selected pipeline, fitted object and environment. They are witnesses for this installed environment, not portable independently refitted model identities. Repeat assessments can share science and row hashes while having different attempt IDs, exposure classes and history.

Existing [acceptance reports](Benchmark.md) remain research diagnostics. Their walk-forward limitations remain separate from a final holdout claim.
