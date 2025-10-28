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

# v1.3.1 on 25th of July, 2025
- Fix CSV reading issue where string values had leading spaces in `reports.log_df.read_from_file`

# v1.4.0 on 31st of July, 2025
- Add tests to CI
- Set `n_permutations` to `10000` in `uel.run` by default, and require an int as input
- Add `n_permutations` as input argument to `utils.ParamSpace`
- Make writing to sqlite optional in `uel.run` with `save_to_sqlite` having `False` as default
- Disable `test_regime_stability`

# v1.4.1 on 31st of July, 2025
- Update `params` in `sfm.reference.logreg`
- Remove redundant code from `model` in `sfm.reference.logreg`
- Remove redundant logic/code from `prep` in `sfm.reference.logreg`
- Refactor `sfm.reference.logreg` to work with standard data handling

# v1.4.2 on 1st of August, 2025
- Fix dataframe column orders in `sfm.logreg.breakout_regressor_ridge` and `sfm.lightgbm.breakout_regressor`
- Update  `sfm.lightgbm.breakout_regressor` and `sfm.lightgbm.regime_multiclass` to data scaling
- Fix reference models verbosity configuration
- Refactor `test_mega_model` to use CSV data instead of live data fetching.

# v.1.4.3 on 2nd of August, 2025
- Update `sfm.lightgbm.regime_stability` to output standard
- Add `sfm.lightgbm.regime_stability` to test suite

# v1.5.0 on 4th of August, 2025
- Create `sfm.lightgbm.tradeable_regressor` decoder
- Add `sfm.lightgbm.tradeable_regressor` to test suite

# v1.5.1 on 5th of August, 2025
- Fix Polars schema error in regime_multiclass SFM by handling None values in params

# v1.6.0 on 5th of August, 2025
- Rename `loop.data` to `loop.historical_data`
- Split `get_historical_klines` into `get_spot_klines` and `get_futures_klines`
- Rename `get_historical_trades` to `get_spot_trades`
- Rename `get_historical_agg_trades` to `get_spot_agg_trades`
- Rename `get_historical_futures_trades` to `get_futures_trades`
- Make `get_futures_trades` create `self.data` instead of returning `pl.DataFrame`

# v1.7.0 on 6th of August, 2025
- Add `loop.features` sub-module
- Add `features.conserved_flux_renormalization`
- Add `transform.mad_transform` for Median Absolute Deviation scaling
- Add `utils.check_if_has_header` to check Binance Market Data files for header

# v1.8.0 on 7th of August, 2025
- Add `lightgbm.tradeline_multiclass` sfm
- Add test for the above SFM

# v1.9.0 on 7th of August, 2025
- Refactor `loop.indicators` sub-module (no code changes except imports)
- Refactor `loop.features` sub-module (no code changes except imports)
- Make docstrings cohesive and coherent across `loop.indicators` and `loop.features`
- Add comprehensive docs for `loop.indicators` and `loop.features`

# v1.10.0 on 8th of August 2025
- Add `rules_based.momentum_volatility` sfm
- Add test for the above SFM

# v1.12.0 on 13th of August 2025
- Add `maker_volume` and `maker_liquidity` columns to `get_klines_data` output
- Add/Update test and doc for the above

# v1.13.0 on 14th of August 2025
- Rename `uel.log_df` to `uel.experiment_log`
- Improve data plumbing in `uel.run`
- Add `loop.backtest` sub-module
- Move `loop.backtest` to `loop.backtest.backtest_sequential`
- Add `loop.backtest.backtest_snapshot`
- Add `loop.log` sub-module
- Add `loop.log.experiment_backtest_results`
- Add `loop.log.experiment_confusion_metrics`
- Add `loop.log.experiment_parameter_correlation`
- Add `loop.log.permutation_confusion_metrics`
- Add `loop.log.permutation_prediction_performance`
- At end of `uel.run` add the above `loop.log` functions as properties
- Add test for the above SFM
- Improve `utils.split_data_to_prep_output` to support latest end-to-end features
- Update all the SFMs to support latest end-to-end features
- Add the latest features to tests
- **NOTE**: Temporarily disables tests `reference.lightgbm` and`lightgbm.test_regime_stability`

# v1.13.1 on 16th of August 2025
- Remove lightgbm regime_multiclass and regime_stability models
- Remove tests for the above
- Cleanup all lightgbm utils/sfms (except megamodel code) docstrings, comments, prints, imports

# v1.13.2 on 16th of August 2025
- Port sfm.reference.lightgbm to Loop standards
- Update `tests.test_sfm` to enable the above
- Removed util function not in use anymore

# v1.13.3 on 16th of August 2025
- Standardize `loop.indicators` inputs and outputs
- Fix `loop.indicators.price_change_pct` calculation
- Format code style for `loop.indicators`
- Update docs for `loop.indicators`

# v1.14.0 on 17th of August 2025
- Refactor data sampler code to generic form in loop.utils.data_sampler
- Remove data sampler class from lightgbm.utils
- Refactor mega model code to generic form in loop.utils
- Remove mega model code from lightgbm.utils, tests
- Add docs for data sampler, mega model

# v1.15.0 on 18th of August 2025
- Follow the column naming pattern from `experiment_backtest_results` in `experiment_confusion_metrics`
- Organize `experiment_confusion_metrics` columns based on actual use pattern
- Pre-compute `experiment_confusion_metrics` and `experiment_backtest_results` (**NOTE**: `experiment_parameter_correlation` remains callable)
- Clean `uel` object namespace

# v1.15.1 on 19th of August 2025
- Add Ichimoku Cloud feature to `loop.features`

# v1.16.0 on 24th of August 2025
- Simplify `lightgbm.tradeable_regressor` - remove deadwood, genericize, etc

# v1.16.1 on 4th of September 2025
- Use deterministic and stable SQL fuctions in `get_klines_data()` to minimize
data mismatch
- Update datasets for test data

# v1.16.2 on 31st of August, 2025
- Fix Streamlit explorer launch path by injecting project root into `PYTHONPATH` for the subprocess in `loop.explorer.loop_explorer`, ensuring `loop` is importable when started via tools/Playwright

# v1.17.0 on 9th of September 2025
- Add `linear_transform.py` under `loop.transforms`
- Add `ridge_classifier.py` under `loop.sfm.ridge`
- Fix `loop.features.ichimoku_cloud` not added into `loop.features.__init__`
- Add test for the above SFM

# v1.18.0 on 13th of September, 2025
- Add `loop.explorer` data visualization toolkit
- Add `features.breakout_percentile_regime`
- Add `features.hh_hl_structure_regime`
- Add `features.ma_slope_regime`
- Add `features.price_vs_band_regime`
- Add `features.window_return_regime`
- Add `indicators.sma_deviation_std`
- Add `indicators.window_return`
- Add `transforms.quantile_trim_transform`
- Add `transforms.winsorize_return`
- Add `transforms.zscore_transform`
- Add new module `snippets` as a home for various dev workflow snippets specific to Loop
- Add `snippets.get_uel_run_object`
- Add `snippets.test_explorer_locally`
- Update project CLAUDE.md and Project.md

# V1.19.0 on 19th of September, 2025
- Add `loop.manifest` for experiment configuration
- Add method chaining API for manifest configuration
- Fix parameter space explosion in `utils.param_space` with mixed radix sampling
- Added a test for sampling from large param space
- Update `sfm.reference.logreg` to use new manifest API
- Add comprehensive docs for `loop.manifest` including integration examples
- Update `Single-File-Model.md` and `Universal-Experiment-Loop.md` to include manifest support 

# V1.19.1 on 22nd of September 2025
- Add `loop.data` sub-module for computing time and information-based bars from base klines data.
- Add fixed threshold trade, volume and liquidity bars to `loop.data.bars`
- Add test cases for the above
- Add documentation `docs/Data-Bars.md`

# V1.19.2 on 24th of September 2025
- Refactor datetime alignment to work with manifest based bar data.
- Fix `snippets/test_explorer_locally.py` to work with manifest based logreg.

# V1.19.3 on 30th of September 2025
- Refactor `sfm.reference.random` to use manifest system
- Refactor `sfm.reference.lightgbm` to use manifest system
- Add `features.lagged_features` to consolidate all lagged features with vectorized Polars implementations
- Remove pandas dependency from `utils.log_to_optuna_study` and `utils.confidence_filtering_system`
- Refactor `utils.add_breakout_ema` to pure Polars implementation
- Update `docs/Features.md` with consolidated lagged features documentation under single section

# V1.19.4 on 10th of October 2025
- Enhance color consistency, typography & spacing in Explorer

# V1.20.0 on 11th of October 2025
- Refactor `loop.manifest`, `loop.universal_experiment_loop`, `loop.log` to remove prep() and model() functions
- Add `loop.sfm.model` that contains sfm model files
- Use latest Manifest on `loop.sfm.ridge.ridge_classifier` and `loop.sfm.reference.logreg`
- Update `loop.universal_experiment_loop` to support fully Manifest, partial Manifest and legacy modes.

# V1.20.1 on 18th of October 2025
- Modify `requirement.txt` with newer package dependencies for `pandas>=2.3.1`, `scikit-learn>=1.6.1`, and `numpy>=2.2.6`
- Fix padkage dependencies versioning for `numpy`, `scikit-learn` and `pandas` in JupyterLab.

# V1.20.2 on 23rd of October 2025
- Refactor `loop.sfm.logreg.regime_multiclass` and `loop.sfm.logreg.breakout_regressor_ridge` to use manifest.
- Fix a data alignment bug in `loop/log/log.py` when there are no missing datetime values.
- Refactor manifest for `loop.sfm.reference.lightgbm.py` to include model assignment. 

# V1.20.3 on 27th of October 2025
- Add image-based deployment with Docker containerization
- Add automated deployment workflow for GitHub Container Registry
- Add Dockerfile for containerized Loop Streamlit application deployment
