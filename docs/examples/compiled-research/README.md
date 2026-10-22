# Compiled Research project

Write the research definition in YAML and run it with `limen`. This project carries recorded observations through preparation, search, selection, Sensor reconstruction, Cohort inference and guarded final assessment.

The [research chapter](../../Research-Book.md) introduces both interfaces. The [definition chapter](../../Compiled-Research.md) develops this project's full manifest. The [Python project](../python-research/README.md) is the independent caller-owned alternative.

## Acquire the matching project

Download [acquire.sh](https://docs.vaquum.fi/limen/source/docs/examples/acquire.sh). It reads one public publication manifest, verifies the exact project-bundle and wheel digests, extracts the recorded inputs and installs the matching wheel with `data`, `stats` and `boosting` extras into a new environment. It requires Python 3.10–3.13, `curl`, `jq`, `tar` and OpenSSL. Package installation is setup; Compiled research itself uses YAML and the Limen CLI.

<!-- example: acquire-compiled-project source=docs/examples/commands.json region=acquire-project scenario=selection interface=compiled role=executed -->
```bash
sh acquire.sh limen-research
```

Enter `limen-research/` and activate its environment with `. .venv/bin/activate`. Run every following command from that bundle root. Keep `publication.json`; it pins the selected release and artifact digests even after the public site advances. The examples and inputs are separate from the installed wheel. To repeat a past run, retain that manifest, bundle, wheel and environment record.

## Prerequisites

Use the recorded example bundle and its matching Limen wheel, Python 3.10–3.13, Git, and `vaquum-limen[data,stats,boosting]`. Run every command from the bundle root. The project uses local recorded inputs and performs no live data download.

The bundle pins raw observations and two derived files by SHA-256. `spot_1h_20240101_20240116.parquet` is exactly the first 384 recorded hourly rows. `recorded-funding.parquet` normalizes the pinned funding records, including realized rates and mark prices. Tests compare both derivations with their recorded sources. No Python preprocessing step is required to run the project.

Use a fresh working directory for a fresh campaign. The YAML's `uel.output_path` determines its results directory under `results/dev/`; recovery uses that saved directory explicitly.

## Begin with selection

<!-- example: compiled-selection-command source=docs/examples/commands.json region=compiled-selection scenario=selection interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/selection.yaml
```

The small `selection.yaml` declares two logistic candidates, chronological dates, a training-fitted target/scaler and a validation-return objective. `limen assess` applies its selection declaration and writes its Sensor/Cohort handoff.

<!-- example: compiled-selection-lifecycle source=docs/examples/compiled-research/run.sh region=selection scenario=selection interface=compiled role=executed -->
```bash
limen validate compiled-research/selection.yaml
limen profile compiled-research/selection.yaml
limen run --dry-run compiled-research/selection.yaml
limen run --no-progress-bar compiled-research/selection.yaml
limen assess compiled-research/selection.yaml --results results/dev/compiled-cli
```

## Run the comprehensive declaration

<!-- example: compiled-comprehensive-command source=docs/examples/commands.json region=compiled-research scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/research.yaml
```
<!-- example: compiled-comprehensive-assess source=docs/examples/commands.json region=compiled-assess scenario=search interface=compiled role=executed -->
```bash
limen assess compiled-research/research.yaml --results results/dev/compiled-research
```

`research.yaml` declares indicators, lagged and fractional features, train-fitted robust scaling, ablation, PCA, calibration, threshold tuning, the execution product and costs, the objective, eight candidates, recording, checkpoints and validation selection. `assessment.json` records two selected Sensor identities; `predictions.parquet` records their Cohort's predictions and unavailable reasons. Reconstruction checks replayed metrics before inference.

<details>
<summary>Read the complete canonical manifest</summary>

<!-- example: compiled-comprehensive-definition source=docs/examples/compiled-research/research.yaml region=research scenario=search interface=compiled role=executed -->
```yaml
schema_version: "1.0"
metadata:
  name: recorded_compiled_research
  mode: development
  description: Compare preparation and regularization on recorded Bitcoin bars.
sfd:
  manifest:
    type: ml
# region source
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_15m_20250101_20250531.parquet
# endregion source
# region partitions
    split_dates:
      train_start: "2025-01-01"
      train_end: "2025-01-15"
      val_start: "2025-01-15"
      val_end: "2025-01-18"
      test_start: "2025-01-18"
      test_end: "2025-01-22"
      val_predict_guard: true
      test_predict_guard: true
# endregion partitions
# region indicators
    indicators:
      - func: limen.indicators.roc
        params:
          period: 1
          group: momentum
      - func: limen.indicators.roc
        params:
          period: 3
          group: momentum
      - func: limen.indicators.roc
        params:
          period: 12
          group: momentum
# endregion indicators
# region features
    features:
      - func: limen.features.lag_range
        params:
          col: roc_1
          start: 1
          end: 3
          group: momentum
      - func: limen.features.fractional_diff
        params:
          d: 0.4
          cols: [close]
          threshold: 0.05
          group: price
# endregion features
# region target
    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: roc_1
        quantile: 0.5
      transform_params:
        shift: -1
# endregion target
# region preparation
    scaler:
      class: limen.scalers.RobustScaler
    feature_ablation:
      drop_count_key: feature_drop_count
      seed_key: feature_drop_seed
    pca_compression:
      enabled_param: auto_pca
      n_components_param: pca_k
# endregion preparation
# region calibration
    calibration:
      probability_calibration:
        func: limen.calibration.sklearn_probability_calibrator
        params:
          method: sigmoid
      threshold_function:
        func: limen.calibration.grid_threshold_optimizer
        params: {}
# endregion calibration
# region objective
    objective:
      metric: backtest_total_return
      direction: maximize
# endregion objective
# region execution
    backtest:
      source_interval_seconds: 900
      fee_bps: 7.0
      slip_bps: 3.0
      notional_rate: 0.5
      execution_lag_seconds: 0.0
      product:
        kind: cash_spot
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1.0e-9
        min_notional: 0.0
# endregion execution
    reference_architecture: limen.sfd.reference_architecture.logreg_binary
# region parameters
  params:
    C: [0.5, 1.0]
    random_state: [42]
    feature_groups: [momentum|price]
    feature_drop_count: [0, 1]
    feature_drop_seed: [7]
    auto_pca: [false, true]
    pca_k: [2]
    use_calibration: [true]
    use_threshold: [true]
# endregion parameters
# region run
uel:
  output_path: compiled-research
  n_permutations: 8
  workers: 1
  search_strategy:
    type: grid
    shuffle: true
    seed: 17
  prep_each_round: true
  checkpoint_interval: 1
  feedback_interval: 1
  output_format: parquet
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
# endregion run
# region assessment
assessment:
  metric: val_backtest_total_return
  direction: maximize
  top_k: 2
  predict: true
# endregion assessment
```

</details>

## Run the complete project

<!-- example: compiled-all-command source=docs/examples/commands.json region=compiled-all scenario=search interface=compiled role=executed -->
```bash
sh compiled-research/run.sh
```

`run.sh` contains only shell and CLI commands. Every research choice is in the YAML it names. Tests execute those files unchanged against the installed wheel; test code reads their artifacts but cannot supply additional research settings.

| Declaration | Question and artifacts |
|---|---|
| `research.yaml`, `preparation.yaml` | Compare preparation and regularization; retain predictions, metrics and the selected handoff. The preparation variant uses a triple-barrier target. |
| `search.yaml`, `search_random.yaml`, `search_sobol.yaml` | Traverse the same four-candidate domain with explicit strategy seeds. |
| `factorized.yaml` | Compare deterministic LightGBM tree-count and fee axes with factorization and seeded shuffle declared in YAML. |
| `adaptive.yaml` | Limit a four-candidate search to three completed rounds; retain its budget intervention and checkpoint. |
| `recovery.yaml` | Pin the Parquet source by SHA-256, record all payload groups and checkpoint every round; continue with `limen run --resume`. Tests also interrupt an actual CLI process and compare resumed and uninterrupted results. |
| `walk_forward.yaml` | Fit and calibrate two qualified folds on the explicit 384-row input; reconstruct their Sensors with `limen assess`. |
| `acceptance.yaml` | Retain two rule trials, 96-return tracks and independent statistical acceptance reports. |
| `calibration.yaml` | Compare all four probability-calibration/threshold modes. |
| `execution.yaml`, `funded.yaml` | Retain dynamic-barrier ledgers, clock bindings, costs and recorded funding. |
| `bars.yaml` | Form volume, trade and liquidity bars separately after each split; omit incomplete final bars. |
| `allocation.yaml`, `allocation_ml.yaml`, `allocation_funded.yaml` | Compare rule and learned primaries with completed-opportunity allocation and recorded funding. |
| `common_strategies.yaml`, `family_*.yaml` | Compare eight fixed baselines and the admitted model families on their own declared populations. |
| `final/holdout.yaml` | Reserve the final window before research; reconstruct, freeze and assess the selected Cohort through the CLI. Preserve its project history and final report together. |
| `final-event/holdout.yaml` | Reserve a separate project and assess one selected Sensor under its frozen event contract, with schema-2 ledger evidence. |
| `project.sh` | Validate, profile, scaffold, commit, fork, index, execute, back up and restore a campaign using installed CLI commands and a local Git remote. |

## Read the evidence

Native `results.csv` and compressed round records are the research outputs. `assessment.json` records selection and membership; it does not serialize arbitrary fitted model weights. Continuous-family YAML uses the CLI's ordinary execution path; optional legacy binary summaries are outside those declarations.

Search seeds govern candidate order. The random-binary diagnostic is stochastic and makes no deterministic replay claim. Real TabPFN execution remains unproved under [#628](https://github.com/Vaquum/Limen/issues/628).

The guarded project reserves January 5–6, 2024, before research. Earlier exposure remains unknown for local recorded files. Reservation and freezing protect this local workflow; they cannot prove that another person or process never read those observations.

The [documentation contract](../../Developer/Documentation-System.md) requires exact excerpt equality, execution of every declared scenario, independent assertions, and source-changing commit/merged-PR provenance. Candidate checks are distinct from released-wheel publication. Historical observations retain their original revision and cannot be relabelled as current CLI output.

The project lifecycle uses the static production declaration in `project.yaml`. Its four candidates share the selection pipeline; production mode permits content-addressed commits and lineage. The shell script copies that declaration unchanged.

`limen fork` leaves an editable development declaration. The lifecycle commits the separately reviewed `project_child.yaml` unchanged, with the original identity supplied through `--parent`; its content and lineage are explicit.
