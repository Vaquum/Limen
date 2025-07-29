# Changelog

## v0.7.9 on 25th of May, 2025
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

## v0.8.6 on 6th of June, 2025
- Generalize `reports.results_df` to work with any experiment using `get_historical_klines`
- Use `x` for axis label in `reports.confusion_matrix_plus`
- Add liquidity-based signals to `get_historical_klines`
- Add `start_date_limit` to `get_historical_klines` for limiting the start date of the data

## v0.8.7 on 6th of June, 2025
- Add `start_date_limit` as an input parameter to `historical.get_historical_klines`

## v0.8.8 on 7th of June, 2025
- Add `futures=False` input argument to `HistoricalData.get_klines_historical` for getting futures data klines

# v0.8.9 on 7th of June, 2025
- Add `context_params` dictionary input in `uel.run` for passing context parameters through `round_params` for logging

# v0.9.0 on 12th of June, 2025
- Add statistical metrics (`mean`, `std`, `median`, and `iqr`) to `data.HistoricalData.get_historical_klines` endpoint for richer data analysis

# v0.9.1 on 17th of June, 2025
- Add simple lag based indicators `lag_column`, `lag_columns` and `lag_range`

# v0.9.2 on 22nd of June, 2025
- Add `breakout_features` indicators for comprehensive breakout signal generation.

# v0.9.3 on 25th of June, 2025
- Fix datetime bucketing logic in `get_klines_data` to use epoch-based intervals instead of minute-boundary resets, ensuring continuous kline intervals across time boundaries
- Add several updates to `models.logreg`
- Reduce pulled datasizes in tests
- Disabled test for `reports.experiment_benchmarking` due to the requirement to manually close the opening plot window

# v0.9.4 on 26th of June, 2025
- Added `utils` for breakout labeling.

# v0.9.5 on 28th of June, 2025
- Added `MegaModelDataSampler` class in lightgbm `utils` for megamodels via data sampling

# v0.9.6 on 30th of June, 2025
- Added lightgbm based `regime_multiclass` SFM for breakout regime classification

# v0.9.7 on 1st of July, 2025
- Added confidence filter utils for all models; megamodel preds for lightgbm base model

# v0.9.8 on 2nd of July, 2025
- Added quantile model and moving average correction for lightgbm base model

# v0.9.9 on 3rd of July, 2025
- Added lightgbm based `breakout_regressor` SFM for predicting breakout magnitude

# v1.0.0 on 5th of July, 2025
- Added lightgbm based `regime stability` SFM - better regime prediction via regime stability

# v1.1.0 on 5th of July, 2025
- Add a plotting function for visualizing decile means in datasets.
- Introduce detailed performance metrics and trade simulation in backtesting
- Add standard backtesting library `Bactest.py`
- Add standard book keeping library `Account.py`
- Improve input validation and overflow protection in `Account`
- Add conviction tests for `Account` and `Backtest`
- Updated `loop.reports` namespace

# v1.1.1 on 21st of July, 2025
- Fix data leakage in quantile flag calculation by adding cutoff parameter
- Rename logreg.py to logreg_example.py for clarity
- Reorganize logreg models to match lightgbm structure with dedicated folder
- Add breakout_regressor_ridge.py model for ridge regression breakout prediction
- Add regime_multiclass.py and breakout_regressor_ridge.py in logreg folder

# v1.2.0 on 23rd of July, 2025

## Metrics
- Add `loop.metrics` as a standard metrics sub-module
- Move `utils.safe_ovr_auc.py` to `loop.metrics`
- Move `utils.metrics` to `loop.metrics.metrics`
- Refactor `loop.metrics.metrics` functions to separate files in `loop.metrics`

# SFM
- Rename `loop.models` to `loop.sfm`
- Refactor `loop.sfm`

# Tests
- Refactor the test suite
- Make tests fail hard
- Remove all printouts (except PASSED/FAILED)

# v1.3.0 on 25th of July, 2025
- Update `sfm.reference.lightgbm` to data scaling and output standard
- Update `sfm.lightgbm.breakout_regressor` to data scaling and output standard
- Update `sfm.lightgbm.regime_multiclass` to data scaling and output standard
- Update `sfm.logreg.breakout_regressor_ridge` to data scaling and output standard
- Update `sfm.logreg.regime_multiclass` to data scaling and output standard
- Refactor test suite to fully modular
- Add all of the above into `tests.test_sfm`
- Add guideline/template comments in `logreg.regime_multiclass`