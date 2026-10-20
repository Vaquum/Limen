# Reserve and assess final observations

A final window is assessed after preparation, model fitting and selection have stopped changing. The [Python project](examples/python-research/README.md) owns that exclusion in its research code. The [Compiled project](examples/compiled-research/README.md) uses a guarded local reservation and a frozen selected Cohort.

Reserve a guarded project's final time window before running research. Supported experiments, Trainer refits and Log reconstruction then admit only native bars ending at or before the earliest active reservation. The exclusion applies before selectors, splitting and fitted preparation, including caller-supplied or reassigned frames.

This records local workflow protection. It cannot establish that a person, another project or an external process has never read these observations. Physical source materialization can exceed the admitted research prefix; provenance records both quantities.

## Prerequisites

Use the matching wheel and recorded input bundle for both projects. The guarded Compiled path additionally requires a Git project containing `limen.toml`, supported native OHLC data and serial built-in research. Preserve its local history as described below before cloning or restoring the project.

## Assess caller-owned final observations

The Python scenario selects on the first 1,440 quarter-hour rows, then assesses 479 next-bar decisions from the remaining observations. It reuses the evaluated fitted state. Its exclusion is a property of this code and its recorded input; it does not create reservation or exposure-history evidence.

<!-- example: holdout-python-assessment source=docs/examples/python-research/scenarios.py region=holdout scenario=holdout interface=python role=executed -->
```python
def holdout(data_dir, output):
    bars = quarter_hour(data_dir)
    cohort, _, selected, retained = project.run_research(bars, str(output / 'selection'))
    tape = cohort.predict_all(bars[1428:1920])[12:]
    # A decision at the current completed close trades the next open-to-close.
    predictions = []
    for point in tape[:-1]:
        if point.prediction is None:
            raise ValueError('A final assessment decision is unavailable: ' + str(point.reason))
        predictions.append(point.prediction)
    positions = np.sign(np.asarray(predictions, dtype=np.float64))
    next_returns = bars['close'].to_numpy()[1441:1920] / bars['open'].to_numpy()[1441:1920] - 1
    net = positions * next_returns - abs(positions) * .0002
    pl.DataFrame({'decision': [point.datetime for point in tape[:-1]], 'position': positions, 'net_return': net}).write_parquet(output / 'assessment.parquet')
    return {'selected': selected, 'retained_models': len(retained), 'assessment_rows': len(net), 'mean_net_return': float(net.mean()), 'fitting_cutoff': bars['datetime'][960].isoformat(), 'selection_cutoff': bars['datetime'][1440].isoformat(), 'protection': 'caller-owned exclusion; no project reservation or exposure-history guarantee'}
```

## Reserve and freeze a Compiled selection

The Compiled scenario creates its project before reservation, runs research ending before January 5, 2024, reconstructs the selected round, and freezes its Cohort before assessing January 5–6. It verifies that the retained report matches the completed assessment. The local input keeps exposure classification `unknown`; a reservation cannot establish prior access outside the local project.

<!-- example: holdout-compiled-assessment source=docs/examples/compiled-research/scenarios.py region=holdout scenario=holdout interface=compiled role=executed -->
```python
def holdout(data_dir, output):
    project = output / 'project'
    project.mkdir(parents=True)
    (project / 'limen.toml').write_text('[project]\nname="book-final-observations"\n')
    subprocess.run(['git', 'init', '--quiet', str(project)], check=True, timeout=10)
    reservation = reserve_holdout(project, '2024-01-05', '2024-01-06', interval_seconds=3600)
    config = configuration('holdout', data_dir)
    directory = project / 'experiment'
    loop = execute(config, directory, project_root=project)
    selected_ids = research_project.completed_log(loop)['id'].head(1).to_list()
    selected = Trainer(directory, project_root=project).train(selected_ids)
    cohort = Cohort(experiment_log_path=str(directory), permutation_ids=selected_ids)
    cohort.set_members(selected)
    frozen = freeze_holdout_selection(cohort, context_start='2024-01-04', scenarios=[dict(fee_bps=5.0, slip_bps=2.0, notional_rate=.25, execution_lag_bars=1, take_profit_bps=None, stop_loss_bps=None)], evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up'))
    report = final_holdout_report(frozen, project_root=project, reservation_id=reservation)
    attempt_id = report['attempt_id']
    if not isinstance(attempt_id, str):
        raise ValueError('The final report must have an attempt identity')
    if read_holdout_report(project, attempt_id) != report:
        raise ValueError('Retained final report differs')
    (output / 'retained-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return {'rounds': research_project.completed_log(loop).height, 'planned': report['planned'], 'classification': report['classification'], 'scenarios': report['scenarios'], 'exposure': report['exposure'], 'rows_sha256': report['rows_sha256'], 'source': report['source']}
```

## Reservation lifecycle

The guarded scenario calls `reserve_holdout()` before starting research in its Git project.

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

Supported selections are exact ordinary YAML-backed binary Sensor/Cohort objects with fitted LogRegBinary or LightGBMBinary, ordered Trainer schemas and matching completed project research records. Cached pipelines must still match their YAML declaration. Configured calibration must already be fitted; Python-backed Sensors, positive native LightGBM best iteration, legacy provenance, custom predictors, nonordinary splits, formed bars, trade contracts and decoder lookback above one are refused before allocation. Original selection mutation cannot change the frozen copy. One handle permits sequential reuse; concurrent use is refused.

Each scenario requires exactly the six keys shown. Fees and slippage are finite nonnegative basis points; notional is in `(0,1]`; execution lag is a positive integer; TP/SL are null or finite positive basis points. Boolean numbers are refused. Order and duplicates are retained. Execution starts flat, excludes context positions, applies lag and forces terminal exit using existing snapshot economics. Raw positions are 0/1 and raw returns are unit-notional fractions. Deployed positions and gross/net/cost returns multiply raw values by notional once; summary returns already use that deployed fraction.

Optional `evaluation_target` accepts an initialized exact NextBarUpTarget or NextBarDownTarget. Labels are computed after predictions from final rows only; the terminal label is unavailable. Accuracy, precision, recall, FPR and AUC disclose denominators/support. Missing labels, zero denominators, one-class AUC, absent probabilities and undefined return summaries are null with an explicit reason. Majority-vote probabilities remain null.

## Attempts and evidence

Allocation commits a serialized project assessment index and planned support before reading context or final data. Every subsequent error carries its attempt ID and retains failed exposure; a killed process may leave `started` with unknown outcome. Warm-up failure consumes an attempt. A new frozen selection with an earlier context or narrower window creates another attempt; it does not erase previous access. Any attempt prevents withdrawal; explicit release remains available.

History migrates atomically from v1 to v2 under the reservation lease. Older reservation-only installations refuse v2. The shared admission lease covers evaluation and finalization, preventing a reservation change mid-assessment while allowing separately serialized allocations.

Reports retain allocation-time prior attempts/reservations and overlap classes. Failed and started attempts count equally with completed attempts. `initial-recorded` is conditional on completed hosted-native local research evidence, an active unreleased reservation and no previous planned support intersecting the final window. Overlap makes later assessments `subsequent`; release makes them `post-release`; local/caller training data is `unknown`. Shared unreserved context alone does not defeat initial-recorded status. External exposure is always unknown.

Each completed attempt stores `manifests/holdout/<attempt_id>/rows.parquet` and `report.json`. Rows retain context and final OHLC, original source columns/types, row hashes, member and aggregate predictions, availability reasons, optional labels and raw/deployed scenario ledgers. Other source columns use a `source_` prefix, repeated until unique when a name would collide; the report retains the original-to-retained name map. Strict JSON metrics contain no NaN/Infinity. History stores both artifact hashes; retrieval verifies completed status, identities, hashes and schemas without loading market data or running models. Partial files from failed writes remain unadvertised evidence. Preserve the evidence directory with the marker and history during backup.

Specification and fitted-state digests pin the selected pipeline, fitted object and environment. They are witnesses for this installed environment, not portable independently refitted model identities. Repeat assessments can share science and row hashes while having different attempt IDs, exposure classes and history.

Existing [acceptance reports](Benchmark.md) remain research diagnostics. Their walk-forward limitations remain separate from a final holdout claim.
