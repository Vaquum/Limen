# Changelog

## 25th of May, 2025
- Add `klines_size` as input argument to [`get_klines_data`](utils/get_klines_data.py) to define size of window in seconds
- Rename `n_rows` input parameter to `n_latest` in [`get_trades_data`](utils/get_trades_data.py) for getting latest rows
- Add `n_sample` input parameter to [`get_trades_data`](utils/get_trades_data.py) for random sampling
- Add ability to pass `params`, `prep`, or `model` as input parameters to `loop.UniversalExperimentLoop.run` to allow quickly iterating through development and research workflows without having to compromise `uel` as a base

## v0.8.0 28th of May, 2025
- Move database connection init to inside `uel.run`
- Add `utils.metrics` with classification metrics an experimental regression metrics
- Add `uel.extras` for storing any arbitrary artefacts in `round_results` in `uel.run`
- Add `uel.models` for storing model as part of each permutation
- Add `models.xgboost` as a placeholder for further XGBoost explorations
- Modularize testing suite in `tests.py`

## v0.8.1 on 30th of May, 2025
- Add `prep_each_round` to `uel.run` for executing `sfm.prep` for each round individually
- Add `random_search` to `uel.run` for turning random search on or off (with off being grid search)
- Separated parameter space handling into pure polars based utility in `utils.param_space`
- Move data splitters out from `loop.HistoricalData` and into `utils.splits`
- Add `splits.split_data_to_prep_output` for a clean way to get the classic 6-fold split data dictionary
- Add reference model `models.random`

## v0.8.2 on 31st of May, 2025
- Improve `n_permutations` handling in `uel.run` so that if `None` the whole space is searched
- Add `reports.quantiles` for getting quantile ranges for a column
- Add `log_to_optuna_study` for unlocking Optuna reporting for `uel.log_df`
- Add several indicators including RSI
- Add `generators.generate_parameter_range` for convenient params management
- Add `maintain_details_in_params` as input argument to `uel.run` for keeping experiment info in `sfm.params`
- Add `quantile_flag`, `atr`, `rsi`, `ema_breakout`, `ppo`, `wilder_rsi`, `kline_imbalance`, `vwap`, and `macd`

## v0.8.3 on 3rd of June, 2025
- Remove data splitting corner cases from `utils.splits.split_sequential`
- Generalize `reports.log_df.corr_df` and add several improvements to it
- Fix the issue in `utils.get_klines_data` which caused open and close being always same

## v0.8.4 on 5th of June, 2025 
- Make `uel` move `round_params['_scaler']` to `uel.scaler` for post run descaling
- Added `fpr`, `positive_rate`, and `negative_rate` to `utils.metrics.metrics_for_classification`
- Removed `f1score` from `utils.metrics.metrics_for_classification`
- Add `models.logreg` as a base model for logistic regression modelling
- Add `transforms.logreg_transform` for bespoke data scaling for `models.logreg`
- Add `reports.confusion_matrix_plus` for post-experiment benchmarking
- Add `reports.results_df`for post-experiment analysis (e.g. input for `reports.confusion_matrix_plus`)
- Add `reports.experiment_benchmarking` as an experimental post-experiment model comparison loop

## v0.8.5 on 5th of June, 2025
- Fix `transforms.logreg_transform.inverse_transform` use in `reports.experiment_benchmarking`
- Eliminate tensorflow import warnings from `import loop`
- Allow `uel` uel to move `round_results[_preds]` from `sfm.model` to `uel.preds` for post run use
