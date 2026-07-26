# End-to-End Workflow

This guide takes a contributor from a fresh checkout to a benchmarked classifier using the current YAML-first surface.

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Create the experiment YAML

Start from the packaged logistic-regression SFD template:

```bash
limen init logreg-first.yaml --template logreg_binary
```

That file is the editable Design of Experiment surface. Its pipeline flows through:

1. `data_source`: fetch OHLCV bars with `limen.data.HistoricalData.get_spot_klines`
2. `indicators`: add `roc`, `atr`, `ppo`, and `wilder_rsi`
3. `features`: add fractional differencing, `vwap`, and `kline_imbalance`
4. `target`: fit a train-only quantile target and shift it forward
5. `scaler`: resolve scaler choice from `params.scaler_type`
6. `params`: define target, feature, calibration, and classifier hyperparameter grids
7. `uel`: define permutation count, search strategy, prep cadence, and output format

The minimal runnable shape is:

```yaml
schema_version: "1.0"

metadata:
  name: logreg_first
  limen_version: "5.9.4"
  mode: development
  description: First logistic-regression binary classifier

sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.HistoricalData.get_spot_klines
      params:
        kline_size: 3600
    split_dates:
      train_start: "2024-01-01"
      train_end: "2024-09-30"
      val_start: "2024-10-01"
      val_end: "2024-11-30"
      test_start: "2024-12-01"
      test_end: "2024-12-31"
    indicators:
      - func: limen.indicators.roc
        params:
          period: "{roc_period}"
          group: momentum
      - func: limen.indicators.atr
        params:
          period: 14
          group: volatility
    features:
      - func: limen.features.vwap
        params:
          group: volume
      - func: limen.features.kline_imbalance
        params:
          group: volume
    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: "roc_{roc_period}"
        quantile: "{q}"
      transform_params:
        shift: "{shift}"
    scaler:
      from_params: scaler_type
    strict_mode: true
    reference_architecture: limen.sfd.reference_architecture.logreg_binary.logreg_binary
  params:
    shift: [-1]
    q: [0.50]
    roc_period: [12]
    scaler_type: [robust]
    solver: [lbfgs]
    penalty: [l2]
    dual: [false]
    tol: [0.01]
    C: [1.0]
    fit_intercept: [true]
    intercept_scaling: [1.0]
    class_weight: [0.55]
    random_state: [42]
    max_iter: [120]
    multi_class: [deprecated]
    verbose: [0]
    warm_start: [false]
    n_jobs: [null]
    l1_ratio: [null]

uel:
  n_permutations: 1
  search_strategy:
    type: grid
  prep_each_round: true
  output_format: csv
```

## Validate, profile, and run

```bash
limen validate logreg-first.yaml
limen profile logreg-first.yaml
limen run --dry-run logreg-first.yaml
limen run logreg-first.yaml
```

`validate` checks schema and reference resolution. `profile` samples the real data path and estimates runtime. `--dry-run` compiles without executing permutations. `run` trains the classifier and writes the result directory.

## Review outputs

The run writes a timestamped directory under `results/dev/` because `metadata.mode` is `development`.

Key artifacts:

| File | Use |
|---|---|
| copied YAML manifest | Exact run definition. |
| `metadata.json` | Manifest reference and run metadata used by downstream tools. |
| `results.csv` | One row per completed permutation. |
| `round_data.jsonl` | Round params, predictions, and alignment metadata. |
| `checkpoint.json` | Resume state when checkpointing has run. |
| `audit.jsonl` | Feedback-cycle trail when feedback has run. |

## Benchmark the classifier

For a CLI-created run, use the generated `results.csv` for first inspection:

```python
import pandas as pd

results = pd.read_csv("results/dev/logreg_first_<timestamp>/results.csv")
print(results.sort_values("auc", ascending=False).head())
```

For full benchmark tables, run the same SFD through a UEL-backed Python object with `post_processing=True`, then inspect `uel.experiment_confusion_metrics` and `uel.experiment_backtest_results`.

## Read next

- [Command-Line Interface](Command-Line-Interface.md)
- [Experiment Manifest](Experiment-Manifest.md)
- [Log](Log.md)
- [Benchmark](Benchmark.md)
- [Backtest](Backtest.md)
