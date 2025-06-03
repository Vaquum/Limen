# Changelog

## 25th of May, 2025

- Add `klines_size` as input argument to [`get_klines_data`](utils/get_klines_data.py) to define size of window in seconds
- Rename `n_rows` input parameter to `n_latest` in [`get_trades_data`](utils/get_trades_data.py) for getting latest rows
- Add `n_sample` input parameter to [`get_trades_data`](utils/get_trades_data.py) for random sampling
- Add ability to pass `params`, `prep`, or `model` as input parameters to `loop.UniversalExperimentLoop.run` to allow quickly iterating through development and research workflows without having to compromise `uel` as a base

## 28th of May, 2025

- Move database connection init to inside `uel.run`
- Add `utils.metrics` with classification metrics an experimental regression metrics
- Add `uel.extras` for storing any arbitrary artefacts in `round_results` in `uel.run`
- Add `uel.models` for storing model as part of each permutation
- Add `models.xgboost` as a placeholder for further XGBoost explorations
- Modularize testing suite in `tests.py`

## 30th of May, 2025
- Add `prep_each_round` to `uel.run` for executing `sfm.prep` for each round individually
- Add `random_search` to `uel.run` for turning random search on or off (with off being grid search)
- Separated parameter space handling into pure polars based utility in `utils.param_space`
- Move data splitters out from `loop.HistoricalData` and into `utils.splits`
- Add `splits.split_data_to_prep_output` for a clean way to get the classic 6-fold split data dictionary
- Add reference model `models.random`

## 31st of May, 2025
- Improve `n_permutations` handling in `uel.run` so that if `None` the whole space is searched
- Add `reports.quantiles` for getting quantile ranges for a column
- Add `log_to_optuna_study` for unlocking Optuna reporting for `uel.log_df`
- Add several indicators including RSI
- Add `generators.generate_parameter_range` for convenient params management
- Add `maintain_details_in_params` as input argument to `uel.run` for keeping experiment info in `sfm.params`
- Add `quantile_flag`, `atr`, `rsi`, `ema_breakout`, `ppo`, `wilder_rsi`, `kline_imbalance`, `vwap`, and `macd`

## 3rd of June, 2025
- Remove data splitting corner cases from `utils.splits.split_sequential`
- Generalize `reports.log_df.corr_df` and add several improvements to it
- Fix the issue in `utils.get_klines_data` which caused open and close being always same
