# Changelog

## 25th of May, 2025

- Add `klines_size` as input argument to [`get_klines_data`](utils/get_klines_data.py) to define size of window in seconds
- Rename `n_rows` input parameter to `n_latest` in [`get_trades_data`](utils/get_trades_data.py) for getting latest rows
- Add `n_sample` input parameter to [`get_trades_data`](utils/get_trades_data.py) for random sampling
- Add ability to pass `params`, `prep`, or `model` as input parameters to `loop.UniversalExperimentLoop.run` to allow quickly iterating through development and research workflows without having to compromise `uel` as a base

## 28th of May, 2025

- Add `utils.metrics` with classification metrics an experimental regression metrics
- Add `uel.extras` for storing any arbitrary artefacts in `round_results` in `uel.run`
- Add `uel.models` for storing model as part of each permutation
- Add `models.xgboost` as a placeholder for further XGBoost explorations
- Modularize testing suite in `tests.py`
