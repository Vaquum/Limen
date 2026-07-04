# `limen.data.utils`

> Public helpers that form bars and split raw frames into the model-ready `data_dict` schema.

## Canonical docs

- [Historical Data](../../../docs/Historical-Data.md)
- [Data Bars](../../../docs/Data-Bars.md)

## What this package owns

Owns the manifest-facing data helpers: the `compute_data_bars()` bar-formation entry point, the train/validation/test split functions, and random window slicing.
Does **not** own raw data access (owned by `limen.data.HistoricalData`) or the low-level bar builders (owned by `limen.data.bars`).

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `compute_data_bars` | Public bar-formation entry point | Dispatches to `limen.data.bars`; used by `set_bar_formation()` |
| `split_sequential` | Ordered train/validation/test windows | The standard manifest split |
| `split_random` | Randomised split | Non-sequential alternative |
| `split_by_dates` | Explicit date-boundary split | For fixed calendar windows |
| `split_data_to_prep_output` | Build the `data_dict` (`x_train`, `y_test`, …) | Does not mutate the passed split list or column list |
| `split_data_to_rule_based_prep_output` | `data_dict` variant for rule-based SFDs | Preserves columns rule strategies read |
| `random_slice` | Random contiguous window | Sampling helper |

## Adjacent modules

- `limen.data.HistoricalData` produces the raw frames these helpers split.
- `limen.data.bars` provides the bar builders `compute_data_bars()` dispatches to.
- `limen.experiment` consumes the resulting `data_dict` through manifests and the Universal Experiment Loop.

## Quick orientation

```text
utils/
├── compute_data_bars.py   # Public bar-formation entry point
├── splits.py              # Train/validation/test split helpers
└── random_slice.py        # Random window slicing helper
```

## Things to know

- `split_data_to_prep_output()` is non-mutating: it copies the split list and column list before building `data_dict`.
- The `data_dict` keys (`x_train`, `y_train`, `x_val`, `x_test`, `y_test`, …) are the schema the rest of Limen's model code expects.
- Rule-based SFDs need `split_data_to_rule_based_prep_output()` because they read raw columns the standard prep drops.

## Read next

- [Historical Data](../../../docs/Historical-Data.md)
- [Data Bars](../../../docs/Data-Bars.md)
- [Experiment Manifest](../../../docs/Experiment-Manifest.md)
