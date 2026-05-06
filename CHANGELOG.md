# Changelog

Note: add all new changelog entries to the bottom of this file.

## v0.7.9 on 25th of May, 2025

- Add `klines_size` as input argument to `get_klines_data` to define size of window in seconds
- Rename `n_rows` input parameter to `n_latest` in `get_trades_data` for getting latest rows
- Add `n_sample` input parameter to `get_trades_data` for random sampling
- Add ability to pass `params`, `prep`, or `model` as input parameters to `limen.UniversalExperimentLoop.run` to allow quickly iterating through development and research workflows without having to compromise `uel` as a base

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
- Move data splitters out from `limen.HistoricalData` and into `utils.splits`
- Add `splits.split_data_to_prep_output` for a clean way to get the classic 6-fold split data dictionary
- Add reference model `models.random`

## v0.8.2 on 31st of May, 2025

- Improve `n_permutations` handling in `uel.run` so that if `None` the whole space is searched
- Add `reports.quantiles` for getting quantile ranges for a column
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
- Eliminate tensorflow import warnings from `import limen`
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

## v0.8.9 on 7th of June, 2025

- Add `context_params` dictionary input in `uel.run` for passing context parameters through `round_params` for logging

## v0.9.0 on 12th of June, 2025

- Add statistical metrics (`mean`, `std`, `median`, and `iqr`) to `data.HistoricalData.get_historical_klines` endpoint for richer data analysis

## v0.9.1 on 17th of June, 2025

- Add simple lag based indicators `lag_column`, `lag_columns` and `lag_range`

## v0.9.2 on 22nd of June, 2025

- Add `breakout_features` indicators for comprehensive breakout signal generation.

## v0.9.3 on 25th of June, 2025

- Fix datetime bucketing logic in `get_klines_data` to use epoch-based intervals instead of minute-boundary resets, ensuring continuous kline intervals across time boundaries
- Add several updates to `models.logreg`
- Reduce pulled datasizes in tests
- Disabled test for `reports.experiment_benchmarking` due to the requirement to manually close the opening plot window

## v0.9.4 on 26th of June, 2025

- Added `utils` for breakout labeling.

## v0.9.5 on 28th of June, 2025

- Added `MegaModelDataSampler` class in lightgbm `utils` for megamodels via data sampling

## v0.9.6 on 30th of June, 2025

- Added lightgbm based `regime_multiclass` SFM for breakout regime classification

## v0.9.7 on 1st of July, 2025

- Added confidence filter utils for all models; megamodel preds for lightgbm base model

## v0.9.8 on 2nd of July, 2025

- Added quantile model and moving average correction for lightgbm base model

## v0.9.9 on 3rd of July, 2025

- Added lightgbm based `breakout_regressor` SFM for predicting breakout magnitude

## v1.0.0 on 5th of July, 2025

- Added lightgbm based `regime stability` SFM - better regime prediction via regime stability

## v1.1.0 on 5th of July, 2025

- Add a plotting function for visualizing decile means in datasets.
- Introduce detailed performance metrics and trade simulation in backtesting
- Add standard backtesting library `Bactest.py`
- Add standard book keeping library `Account.py`
- Improve input validation and overflow protection in `Account`
- Add conviction tests for `Account` and `Backtest`
- Updated `limen.reports` namespace

## v1.1.1 on 21st of July, 2025

- Fix data leakage in quantile flag calculation by adding cutoff parameter
- Rename logreg.py to logreg_example.py for clarity
- Reorganize logreg models to match lightgbm structure with dedicated folder
- Add breakout_regressor_ridge.py model for ridge regression breakout prediction
- Add regime_multiclass.py and breakout_regressor_ridge.py in logreg folder

## v1.2.0 on 23rd of July, 2025

- Add `limen.metrics` as a standard metrics sub-module
- Move `utils.safe_ovr_auc.py` to `limen.metrics`
- Move `utils.metrics` to `limen.metrics.metrics`
- Refactor `limen.metrics.metrics` functions to separate files in `limen.metrics`
- Rename `limen.models` to `limen.sfm`
- Refactor `limen.sfm`
- Refactor the test suite
- Make tests fail hard
- Remove all printouts (except PASSED/FAILED)

## v1.3.0 on 25th of July, 2025

- Update `sfm.reference.lightgbm` to data scaling and output standard
- Update `sfm.lightgbm.breakout_regressor` to data scaling and output standard
- Update `sfm.lightgbm.regime_multiclass` to data scaling and output standard
- Update `sfm.logreg.breakout_regressor_ridge` to data scaling and output standard
- Update `sfm.logreg.regime_multiclass` to data scaling and output standard
- Refactor test suite to fully modular
- Add all of the above into `tests.test_sfm`
- Add guideline/template comments in `logreg.regime_multiclass`

## v1.3.1 on 25th of July, 2025

- Fix CSV reading issue where string values had leading spaces in `reports.log_df.read_from_file`

## v1.4.0 on 31st of July, 2025

- Add tests to CI
- Set `n_permutations` to `10000` in `uel.run` by default, and require an int as input
- Add `n_permutations` as input argument to `utils.ParamSpace`
- Make writing to sqlite optional in `uel.run` with `save_to_sqlite` having `False` as default
- Disable `test_regime_stability`

## v1.4.1 on 31st of July, 2025

- Update `params` in `sfm.reference.logreg`
- Remove redundant code from `model` in `sfm.reference.logreg`
- Remove redundant logic/code from `prep` in `sfm.reference.logreg`
- Refactor `sfm.reference.logreg` to work with standard data handling

## v1.4.2 on 1st of August, 2025

- Fix dataframe column orders in `sfm.logreg.breakout_regressor_ridge` and `sfm.lightgbm.breakout_regressor`
- Update `sfm.lightgbm.breakout_regressor` and `sfm.lightgbm.regime_multiclass` to data scaling
- Fix reference models verbosity configuration
- Refactor `test_mega_model` to use CSV data instead of live data fetching.

## v.1.4.3 on 2nd of August, 2025

- Update `sfm.lightgbm.regime_stability` to output standard
- Add `sfm.lightgbm.regime_stability` to test suite

## v1.5.0 on 4th of August, 2025

- Create `sfm.lightgbm.tradeable_regressor` decoder
- Add `sfm.lightgbm.tradeable_regressor` to test suite

## v1.5.1 on 5th of August, 2025

- Fix Polars schema error in regime_multiclass SFM by handling None values in params

## v1.6.0 on 5th of August, 2025

- Rename `limen.data` to `limen.historical_data`
- Split `get_historical_klines` into `get_spot_klines` and `get_futures_klines`
- Rename `get_historical_trades` to `get_spot_trades`
- Rename `get_historical_agg_trades` to `get_spot_agg_trades`
- Rename `get_historical_futures_trades` to `get_futures_trades`
- Make `get_futures_trades` create `self.data` instead of returning `pl.DataFrame`

## v1.7.0 on 6th of August, 2025

- Add `limen.features` sub-module
- Add `features.conserved_flux_renormalization`
- Add `transform.mad_transform` for Median Absolute Deviation scaling
- Add `utils.check_if_has_header` to check Binance Market Data files for header

## v1.8.0 on 7th of August, 2025

- Add `lightgbm.tradeline_multiclass` sfm
- Add test for the above SFM

## v1.9.0 on 7th of August, 2025

- Refactor `limen.indicators` sub-module (no code changes except imports)
- Refactor `limen.features` sub-module (no code changes except imports)
- Make docstrings cohesive and coherent across `limen.indicators` and `limen.features`
- Add comprehensive docs for `limen.indicators` and `limen.features`

## v1.10.0 on 8th of August 2025

- Add `rules_based.momentum_volatility` sfm
- Add test for the above SFM

## v1.12.0 on 13th of August 2025

- Add `maker_volume` and `maker_liquidity` columns to `get_klines_data` output
- Add/Update test and doc for the above

## v1.13.0 on 14th of August 2025

- Rename `uel.log_df` to `uel.experiment_log`
- Improve data plumbing in `uel.run`
- Add `limen.backtest` sub-module
- Move `limen.backtest` to `limen.backtest.backtest_sequential`
- Add `limen.backtest.backtest_snapshot`
- Add `limen.log` sub-module
- Add `limen.log.experiment_backtest_results`
- Add `limen.log.experiment_confusion_metrics`
- Add `limen.log.experiment_parameter_correlation`
- Add `limen.log.permutation_confusion_metrics`
- Add `limen.log.permutation_prediction_performance`
- At end of `uel.run` add the above `limen.log` functions as properties
- Add test for the above SFM
- Improve `utils.split_data_to_prep_output` to support latest end-to-end features
- Update all the SFMs to support latest end-to-end features
- Add the latest features to tests
- **NOTE**: Temporarily disables tests `reference.lightgbm` and`lightgbm.test_regime_stability`

## v1.13.1 on 16th of August 2025

- Remove lightgbm regime_multiclass and regime_stability models
- Remove tests for the above
- Cleanup all lightgbm utils/sfms (except megamodel code) docstrings, comments, prints, imports

## v1.13.2 on 16th of August 2025

- Port sfm.reference.lightgbm to Loop standards
- Update `tests.test_sfm` to enable the above
- Removed util function not in use anymore

## v1.13.3 on 16th of August 2025

- Standardize `limen.indicators` inputs and outputs
- Fix `limen.indicators.price_change_pct` calculation
- Format code style for `limen.indicators`
- Update docs for `limen.indicators`

## v1.14.0 on 17th of August 2025

- Refactor data sampler code to generic form in `limen.utils.data_sampler`
- Remove data sampler class from lightgbm.utils
- Refactor mega model code to generic form in `limen.utils`
- Remove mega model code from lightgbm.utils, tests
- Add docs for data sampler, mega model

## v1.15.0 on 18th of August 2025

- Follow the column naming pattern from `experiment_backtest_results` in `experiment_confusion_metrics`
- Organize `experiment_confusion_metrics` columns based on actual use pattern
- Pre-compute `experiment_confusion_metrics` and `experiment_backtest_results` (**NOTE**: `experiment_parameter_correlation` remains callable)
- Clean `uel` object namespace

## v1.15.1 on 19th of August 2025

- Add Ichimoku Cloud feature to `limen.features`

## v1.16.0 on 24th of August 2025

- Simplify `lightgbm.tradeable_regressor` - remove deadwood, genericize, etc

## v1.16.1 on 4th of September 2025

- Use deterministic and stable SQL fuctions in `get_klines_data()` to minimize
  data mismatch
- Update datasets for test data

## v1.16.2 on 31st of August, 2025

- Fix Streamlit explorer launch path by injecting project root into `PYTHONPATH` for the subprocess in `limen.explorer.limen_explorer`, ensuring `limen` is importable when started via tools/Playwright

## v1.17.0 on 9th of September 2025

- Add `linear_transform.py` under `limen.transforms`
- Add `ridge_classifier.py` under `limen.sfm.ridge`
- Fix `limen.features.ichimoku_cloud` not added into `limen.features.__init__`
- Add test for the above SFM

## v1.18.0 on 13th of September, 2025

- Add `limen.explorer` data visualization toolkit
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

## v1.19.0 on 19th of September, 2025

- Add `limen.manifest` for experiment configuration
- Add method chaining API for manifest configuration
- Fix parameter space explosion in `utils.param_space` with mixed radix sampling
- Added a test for sampling from large param space
- Update `sfm.reference.logreg` to use new manifest API
- Add comprehensive docs for `limen.manifest` including integration examples
- Update `Single-File-Model.md` and `Universal-Experiment-Loop.md` to include manifest support

## v1.19.1 on 22nd of September 2025

- Add `limen.data` sub-module for computing time and information-based bars from base klines data.
- Add fixed threshold trade, volume and liquidity bars to `limen.data.bars`
- Add test cases for the above
- Add documentation `docs/Data-Bars.md`

## v1.19.2 on 24th of September 2025

- Refactor datetime alignment to work with manifest based bar data.
- Fix `snippets/test_explorer_locally.py` to work with manifest based logreg.

## v1.19.3 on 30th of September 2025

- Refactor `sfm.reference.random` to use manifest system
- Refactor `sfm.reference.lightgbm` to use manifest system
- Add `features.lagged_features` to consolidate all lagged features with vectorized Polars implementations
- Remove pandas dependency from `utils.log_to_optuna_study` and `utils.confidence_filtering_system`
- Refactor `utils.add_breakout_ema` to pure Polars implementation
- Update `docs/Features.md` with consolidated lagged features documentation under single section

## v1.19.4 on 10th of October 2025

- Enhance color consistency, typography & spacing in Explorer

## v1.20.0 on 11th of October 2025

- Refactor `limen.manifest`, `limen.universal_experiment_loop`, `limen.log` to remove prep() and model() functions
- Add `limen.sfm.model` that contains sfm model files
- Use latest Manifest on `limen.sfm.ridge.ridge_classifier` and `limen.sfm.reference.logreg`
- Update `limen.universal_experiment_loop` to support fully Manifest, partial Manifest and legacy modes.

## v1.20.1 on 18th of October 2025

- Modify `requirement.txt` with newer package dependencies for `pandas>=2.3.1`, `scikit-learn>=1.6.1`, and `numpy>=2.2.6`
- Fix package dependencies versioning for `numpy`, `scikit-learn` and `pandas` in JupyterLab.

## v1.20.2 on 23rd of October 2025

- Refactor `limen.sfm.logreg.regime_multiclass` and `limen.sfm.logreg.breakout_regressor_ridge` to use manifest.
- Fix a data alignment bug in `limen/log/log.py` when there are no missing datetime values.
- Refactor manifest for `limen.sfm.reference.lightgbm.py` to include model assignment.

## v1.21.0 on 30th of October 2025

- Add strategy logic to `lightgbm.tradeline_multiclass`
- Simplify `lightgbm.tradeline_multiclass` - remove deadwood, genericize, etc
- Add `lightgbm.tradeline_long_binary` based on the multiclass SFM

## v1.21.1 on 31st of October 2025

- Add `indicator.bollinger_bands`, `indicator.cci`, and `indicator.stochastic_oscillator`
- Add `features.sma_crossover`
- Refactor `sfm.ridge.ridge_classifier` with richer parameter ranges
- Fix `limen.universal_experiment_loop` to provide manifest support for `sfm.reference.empty`
- Add scaling for all klines data in `transform.linear_transform`
- Fix padkage dependencies versioning for `numpy`, `scikit-learn` and `pandas` in JupyterLab.

## v1.22.0 on 26th of November, 2025

- Implement Regime Diversified Opinion Pools (RDOP) system in `limen.regime_diversified_opinion_pools.py`
- Fix `limen.sfm.reference.xgboost` parameter issue with missing round_params in UniversalExperimentLoop
- Fix `limen.universal_experimental_loop` issue with support for `prep_each_round=False`
- Move `get_klines_data_fast()`, `get_klines_data_medium()`, `get_klines_data_large()`, and `get_klines_data_small_fast()` from `limen.tests.test_sfm` to `limen.tests.utils.get_data`
- Add comprehensive RDOP testing suite in `tests.test_regime_diversified_opinion_pools.py`
- Create documentation `docs/Regime-Diversified-Opinion-Pools.md`

## v1.23.0 on 22nd of November 2025

- Add `lightgbm.tradeline_directional_conditional` SFM
- Fix type compatibility in tradeline_multiclass trading metrics

## v1.24.0 on 8th of December 2025

- Refactor xgboost reference sfm to use manifest
- Refactor rule based sfms to use manifest
- Refactor `lightgbm/tradeable_regressor` sfm to use manifest

## v1.24.1 on 9th of December, 2025

- Modify `limen.sfm.model.ridge_binary` to add combination of frozenestimator and cv folds params
- Add use_frozen_estimator parameter to mimic prefitted calibration behavior using sklearn's FrozenEstimator
- Add ensemble parameter to control ensemble calibration in CalibratedClassifierCV

## v1.25.0 on 19th of December, 2025

- Add declarative data fetching to manifest
- Add data source configuration to all manifest-based SFMs

## v1.26.0 on 25th of December, 2025

- Remove all but foundational SFDs
- Prune unused code
- Update tests and docs to incorporate the standard glossary
- Add local file-based endpoint to `historical_data.py`

## v1.27.0 on 26th of December, 2025

- Organize files in root into respective modules
- Move all data related code from `limen/utils` to `limen/data`

## v1.28.0 on 28th of December, 2025

- Move scalers from `limen.transforms` to new sub-module `limen.scalers`
- Update documentations for the above
- Refactor `limen.utils` and `limen.transforms`

## v1.28.1 on 29th of December, 2025

- Add missing indicator exports in `limen.indicator.__init__`
- Add missing feature exports in `limen.feature.__init__`
- Update `Indicators.md` and `Features.md` to the latest

## v1.29.0 on 29th of December, 2025

- Remove `limen.reports` sub-module entirely

## v1.30.0 on 29th of December, 2025

- Add `tabpfn_binary` foundational SFD with validation-based dynamic threshold tuning
- Add `bollinger_position` indicator
- Add `forward_breakout_target` feature
- Add `balanced_metric` for threshold optimization

## v1.31.0 on 31st of December, 2025

- Refactor generic endpoints for querying Clickhouse data
- Update docs for the above

## v1.32.0 on 9th of January, 2026

- Move data endpoint creds to input argument

## v1.32.1 on 22nd of January, 2026

- Configure strict Ruff linting rules in `pyproject.toml` to enforce code style and architectural guidelines
- Replace all `print()` statements with `logging` in library code
- Replace usage of `os.path` and `open()` with `pathlib.Path` across the codebase
- Extract magic numbers into named constants in `limen/` modules
- Add comprehensive type hints to `experiment_core`, `account`, and `backtest` modules
- Clean up unused arguments and imports
- Update GitHub Actions workflow to enforce linting checks

## v1.32.2 on 5th of February, 2026

- Made naming of all PR check workflows consistent

## v1.33.0 on 24th of February, 2026

- Add `ParamDomain` mutable parameter space with observer pattern
- Add `SearchStrategy` abstract base class for search strategy implementations
- Add `MSQ` (Mutable Search Queue) intervention orchestrator with priority queue, custom filters, trim budget, and checkpoint support
- Add ruff per-file ignores for test files

## v1.34.0 on 24th February, 2026

- Add `PruningStrategy` abstract base class for experiment reducers

## v1.35.0 on 25th of February, 2026

- Add `FeedbackController` orchestrator for feedback loop interventions from pruning strategies and user defined custom rules
- Add `_create_temp_log` and `_trigger_feedback` methods to `UniversalExperimentLoop` for future integration with feedback loop
- Refactor test stubs for feedback loop into shared `tests/stubs/` module

## v1.36.0 on 26th of February, 2026

- Add `CheckpointManager` for saving, loading, and validating experiment state across MSQ, domain, and metadata
- Add `get_state` / `set_state` to `ParamDomain` for checkpoint serialisation and restore
- Add `_initialize_fresh`, `_checkpoint`, `_resume_from_checkpoint`, and `_register_shutdown_handler` methods to `UniversalExperimentLoop`
- Add `_shutdown_requested` and `_pause_requested` flags to `UniversalExperimentLoop` for graceful signal handling

## v1.37.0 on 1st of March, 2026

- Add `indicators.midpoint` implementing TA-Lib MIDPOINT (Midpoint Over Period)

## v1.38.0 on 1st of March, 2026

- Add `indicators.avgprice` (TA-Lib AVGPRICE): average of open, high, low, close
- Add `indicators.medprice` (TA-Lib MEDPRICE): (high + low) / 2
- Add `indicators.midprice` (TA-Lib MIDPRICE): rolling midpoint of high/low over period
- Add `indicators.typprice` (TA-Lib TYPPRICE): (high + low + close) / 3
- Add `indicators.wclprice` (TA-Lib WCLPRICE): (high + low + 2 * close) / 4
- Add `indicators.var` (TA-Lib VAR): rolling sample variance
- Add `indicators.linearreg` (TA-Lib LINEARREG): rolling linear regression end value
- Add `indicators.linearreg_slope` (TA-Lib LINEARREG_SLOPE): rolling OLS slope
- Add `indicators.linearreg_intercept` (TA-Lib LINEARREG_INTERCEPT): rolling OLS intercept
- Add `indicators.linearreg_angle` (TA-Lib LINEARREG_ANGLE): rolling OLS slope angle in degrees

## v1.39.0 on 4th of March, 2026

- Add `_run_with_msq` method to `UniversalExperimentLoop` for MSQ-based experiment execution with feedback and checkpoint integration
- Add `_finalize` method for post-experiment Log creation and metrics computation
- Refactor `UniversalExperimentLoop.__init__` to accept search strategy, pruning strategies, feedback, and checkpoint configuration
- Refactor `run()` to dispatch to MSQ-based flow when search strategy is configured
- Extend `CheckpointManager` to persist `FeedbackController` and `PruningStrategy` states alongside MSQ and domain
- Add round data persistence via `round_data.jsonl` for full experiment integrity across shutdown and resume

## v1.40.0 on 9th of March, 2026

- Add `SanityReducer` pruning strategy with NaN detection and suggestion system (zero-metric, execution timeout, warning detectors)
- Add advisory suggestion flow to `FeedbackController` — suggestions are logged in audit trail but not dispatched to MSQ

## v1.41.0 on 10th of March, 2026

- Implement 106 TA-Lib indicators in `limen.indicators` and align behavior with TA-Lib references.
- Add Volume Indicators: `indicators.ad`, `indicators.adosc`, `indicators.mfi`, `indicators.obv`
- Add Volatility Indicators: `indicators.atr`, `indicators.bbands`, `indicators.natr`, `indicators.trange`
- Add Momentum Indicators: `indicators.apo`, `indicators.bop`, `indicators.cci`, `indicators.cmo`, `indicators.macd`, `indicators.macdext`, `indicators.macdfix`, `indicators.mom`, `indicators.ppo`, `indicators.roc`, `indicators.rocp`, `indicators.rocr`, `indicators.rocr100`, `indicators.rsi`, `indicators.stoch`, `indicators.stochf`, `indicators.stochrsi`, `indicators.trix`, `indicators.ultosc`, `indicators.willr`
- Add Overlap Studies: `indicators.dema`, `indicators.ema`, `indicators.ht_trendline`, `indicators.kama`, `indicators.ma`, `indicators.mama`, `indicators.sar`, `indicators.sarext`, `indicators.sma`, `indicators.t3`, `indicators.tema`, `indicators.trima`, `indicators.wma`
- Add Cycle Indicators: `indicators.ht_dcperiod`, `indicators.ht_dcphase`, `indicators.ht_phasor`, `indicators.ht_sine`, `indicators.ht_trendmode`
- Add Statistic Functions: `indicators.linearreg`, `indicators.linearreg_angle`, `indicators.linearreg_intercept`, `indicators.linearreg_slope`, `indicators.stddev`, `indicators.tsf`, `indicators.var`
- Add Price Transform: `indicators.avgprice`, `indicators.medprice`, `indicators.midpoint`, `indicators.midprice`, `indicators.typprice`, `indicators.wclprice`
- Add Pattern Recognition (Candlesticks): `indicators.cdl2crows`, `indicators.cdl3blackcrows`, `indicators.cdl3inside`, `indicators.cdl3linestrike`, `indicators.cdl3starsinsouth`, `indicators.cdl3whitesoldiers`, `indicators.cdlabandonedbaby`, `indicators.cdladvancedblock`, `indicators.cdlbelthold`, `indicators.cdlclosingmarubozu`, `indicators.cdlconcealbabyswall`, `indicators.cdlcounterattack`, `indicators.cdldarkcloudcover`, `indicators.cdldoji`, `indicators.cdldragonflydoji`, `indicators.cdlengulfing`, `indicators.cdlgravestonedoji`, `indicators.cdlhammer`, `indicators.cdlhangingman`, `indicators.cdlharami`, `indicators.cdlharamicross`, `indicators.cdlhighwave`, `indicators.cdlhikkake`, `indicators.cdlhikkakemod`, `indicators.cdlhomingpigeon`, `indicators.cdlidentical3crows`, `indicators.cdlinvertedhammer`, `indicators.cdlladderbottom`, `indicators.cdllongleggeddoji`, `indicators.cdllongline`, `indicators.cdlmarubozu`, `indicators.cdlmatchinglow`, `indicators.cdlmathold`, `indicators.cdlonneck`, `indicators.cdlpiercing`, `indicators.cdlrickshawman`, `indicators.cdlrisefall3methods`, `indicators.cdlseparatinglines`, `indicators.cdlshootingstar`, `indicators.cdlshortline`, `indicators.cdlspinningtop`, `indicators.cdlstalledpattern`, `indicators.cdlsticksandwich`, `indicators.cdltakuri`, `indicators.cdlthrusting`, `indicators.cdltristar`, `indicators.cdlunique3river`
- Add TA-Lib parity test file `tests/test_indicators_vs_talib.py`.

## v1.41.1 on 11th of March, 2026

- Add `CorrelationReducer` pruning strategy with wrong-direction removal and low-impact suggestions
- Add named filter infrastructure to `MSQ` with `set_filter`/`clear_filter` for reversible domain restrictions
- Add declarative filter specs (`FILTER_EXCLUDE_VALUE`, `FILTER_KEEP_VALUES`, `FILTER_KEEP_BETWEEN`, `FILTER_SAMPLE`) with builder registry in `FeedbackController`
- Add `SaturationReducer` pruning strategy with CV-based saturation detection and partial pruning via sample filters
- Add `FocusReducer` pruning strategy with breakthrough detection, parameter space narrowing, variation injection, and timeout snap-back
- Add `BudgetReducer` pruning strategy with walltime projection, permutation counting, and random/worst_first trim strategies
- Add ta-lib to `pyproject.toml` dependencies for testing purpose

## v1.42.0 on 12th of March, 2026

- Add `ReferenceModel` abstract base class for class-based reference architecture with shared `_compute_confusion()` and `_compute_backtest()` helpers
- Refactor `XGBoostRegressor`, `LogRegBinary`, `RandomBinary`, `TabPFNBinary` from standalone functions to classes with `.train()` / `.evaluate()` interface
- Legacy function wrappers preserved for backward compatibility

## v1.43.0 on 13th of March, 2026

- Add `price_data_for_backtest` to `prepare_data()` output with raw OHLC from test split after bar formation
- Add `with_params_override()` method to `Manifest` for creating deep copies with overridden split_config or data source parameters

## v1.44.0 on 16th of March, 2026

- Enable inline metrics (`confusion_*`, `backtest_*` columns) in all reference architecture function wrappers by default

## v1.45.0 on 17th of March, 2026

- Add `metadata.json` to experiment directory, written on experiment start with SFD module path and version
- Add `Trainer` class for retraining selected permutations from a completed experiment
- Add `Sensor` class as callable wrapper around trained models

## v1.46.0 on 20th of March, 2026

- Add Pass 2 full-data retraining to Trainer with `split_config=(1,0,0)`
- Add `ReconstructionError` exception raised when Pass 1 metrics deviate beyond tolerance
- Add `deterministic` class attribute to `ReferenceModel` for per-model tolerance selection
- Sensor now wraps trained `ReferenceModel` and is callable for inference

## v1.47.0 on 23rd of March, 2026

- Replace `FeatureEntry` tuple with `TransformEntry` dataclass supporting `group` and `include_if` metadata
- Add feature group filtering via `feature_groups` round_params key
- Add conditional feature inclusion via `include_if` round_params key
- Add `set_feature_ablation()` for random Drop-N feature column ablation with deterministic seeding
- Rename `FeatureEntry` to `PipelineStep` for pre-split and bar formation pipeline steps

## v1.48.0 on 24th of March, 2026

- Add `RobustScaler` with median and IQR scaling
- Add `RankGaussScaler` with rank-to-normal Gaussian transformation
- Add `SCALER_REGISTRY` and `set_scaler_from_params()` for params-based scaler selection

## v1.49.0 on 26th of March, 2026

- Add `fractional_diff` feature using Fixed-Width Fractional Differentiation (FFD) method
- Add `adf_test` wrapper for Augmented Dickey-Fuller stationarity testing
- Add `find_min_d` utility for finding minimum differentiation order achieving stationarity
- Add split validation in `set_split_config` (train must be positive, val/test non-negative)
- Add column consistency check across splits in `prepare_data`
- Add `statsmodels` as project dependency

## v1.50.0 on 31st of March, 2026

- Add `REDUCER_REGISTRY` for params-based reducer selection
- Migrate `logreg_binary` SFD to params-based scaler and feature groups

## v1.51.0 on 3rd of April, 2026

- Add `RandomStrategy` for lazy random parameter sampling with dedup
- Add `GridStrategy` with index-based modular arithmetic and optional shuffle
- Add `STRATEGY_REGISTRY` for params-based strategy selection
- Add `_param_hash` dedup infrastructure to `SearchStrategy` base class
- Add strategy metadata (`_param_hash`, `_generation_index`, `_search_strategy`) to MSQ yield
- Add `_seen` set rebuild from experiment log on resume
- Migrate foundational SFD tests from legacy path to MSQ path

## v1.52.0 on 7th of April, 2026

- Remove `log_to_optuna_study` from `limen.utils` and drop `optuna` from the project dependencies
- Remove the legacy `save_to_sqlite` path from `UniversalExperimentLoop.run`
- Remove the unused top-level `reports` export from `limen`
- Remove legacy non-core dependencies (`matplotlib`, `seaborn`, `streamlit`, `plotly`, `ipython`, `mcp`, and `playwright`) from `pyproject.toml`
- Replace temp-file based `Log.read_from_file()` loading with in-memory CSV parsing while preserving duplicate-header cleanup
- Add regression coverage for file-backed log loading and whitespace trimming in `tests.run`

## v1.53.0 on 13th of April, 2026

- Switch Limen spot raw-trade and kline queries from `tdw.binance_trades` to `tdw.binance_trades_complete` so `HistoricalData` sees the daily overlay view while preserving finalized monthly history

## v2.0.0 on 15th of April, 2026

- Replace `HistoricalData`'s ClickHouse-backed surface with a file-backed surface
- Reduce `HistoricalData` to `get_spot_klines()`, `get_binance_file()`, and `get_any_file()`
- Make `HistoricalData` methods return `polars.DataFrame` directly while still updating `self.data`
- Source `get_spot_klines()` from the daily Hugging Face BTCUSDT 1-minute dataset by default
- Remove `_get_data_for_test()` and update manifests, tests, and docs to use file-based ingestion
- Remove the old `_internal/generic_endpoints.py` ClickHouse helper
- Drop the `clickhouse_connect` dependency and declare `requests` explicitly
- Remove `median` and `iqr` from `get_spot_klines()` output because the canonical dataset no longer carries them

## v2.1.0 on 18th of April, 2026

- Split time-based feature extraction into `calendar_time_features` and `cyclical_time_features`
- Remove `time_features`, so `from limen.features.time_features import time_features` no longer works; migrate to `calendar_time_features`, `cyclical_time_features`, or call both if you previously relied on the combined output
- Add discrete calendar fields including `hour`, `weekday`, `day_of_month`, `day_of_year`, `week_of_year`, `month`, `quarter`, and `is_weekend`
- Add sine/cosine encodings for cyclical calendar context including hour, minute, weekday, month, quarter, and related date fields

## v2.1.1 on 18th of April, 2026

- Remove `wrangle` dependency by replacing `wrangle.col_to_multilabel` with inline `pd.get_dummies` in `limen/log/log.py`

## v2.2.0 on 18th of April, 2026

- Add range-based volatility features: `parkinson_volatility`, `garman_klass_volatility`, `rogers_satchell_volatility`, and `yang_zhang_volatility`
- Add OHLCV liquidity and impact features: `dollar_volume`, `amihud_illiquidity`, `return_per_dollar_volume`, `range_per_dollar_volume`, and `illiquidity_shock`
- Add realized risk and tail features: `realized_semivariance`, `realized_skewness`, `realized_kurtosis`, `jump_variation_proxy`, `tail_event_intensity`, and `volatility_of_volatility`
- Add hour-of-week normalized OHLCV features: `relative_volume_seasonality`, `relative_range_seasonality`, and `relative_volatility_seasonality`
- Add candle-structure and auction-style features: `body_to_range`, `wick_imbalance`, `range_overlap`, `rejection_intensity`, and `absorption_intensity`
- Add cross-timescale state features: `trend_coherence` and `volatility_term_structure`
- Add documentation and full behavioral test coverage for the new OHLCV feature families

## v2.3.0 on 18th of April, 2026

- Change snapshot backtests to execute predictions on the immediate next execution row by default, with `execution_lag_bars=0` available for legacy same-row behavior
- Change snapshot `trade_*` metrics to run-level by default, add explicit `bar_*` metrics, and retain `trades_count_mode='bars'` for legacy-style bar metrics
- Apply the new snapshot defaults explicitly to experiment backtest summaries and SFD inline backtest metrics so the behavior change is intentional and visible in outputs
- Add confusion-bucket mean return columns (`tp/fp/tn/fn_mean_return_pct`) to snapshot backtest summaries when aligned `actuals` are available
- Populate inline snapshot confusion-bucket return metrics from the aligned test labels, with regression SFDs passing directional actuals explicitly
- Align regression experiment/log snapshot backtests with the same directional binary convention as inline reference-architecture backtests
- Add `mean_kelly_pct` to snapshot backtest summaries using the active trade/bar return distribution with breakeven observations kept in the Kelly sample
- Add direct snapshot tests to the canonical `python -m tests.run` path and extend docs for the updated snapshot semantics

## v2.3.1 on 19th of April, 2026

- Revert v2.3.0 snapshot backtest alignment and trade metric changes in full, restoring the pre-v2.3.0 behavior and output surface

## v2.3.2 on 19th of April, 2026

- Add `mean_kelly_pct` back to snapshot backtests without changing the restored legacy execution semantics
- Add TP/FP/TN/FN mean return percentage columns to the existing confusion metrics surfaces, both inline and post-run, using the existing aligned `open` and `price_change` data

## v2.3.3 on 19th of April, 2026

- Fix `backtest_snapshot()` execution alignment for completed-bar pipelines by evaluating prediction row `t` on the immediate next execution row while preserving the existing HOLD-WHILE-1 semantics
- Apply the same snapshot execution alignment explicitly to inline reference-architecture backtests and post-run experiment backtest summaries
- `bars_total`, `bars_in_market_pct`, and `sharpe_per_bar` in `backtest_snapshot()` are now computed on the tradable evaluation window, so trailing `execution_lag_bars` rows and any non-tradable rows are excluded from denominators

## v2.3.4 on 19th of April, 2026

- Align inline and post-run confusion mean-return metrics to the same immediate-next-execution-row contract used by snapshot backtests for completed-bar pipelines
- Make post-run snapshot backtests raise explicitly on unsupported multiclass logged outputs while keeping the existing binary and directional-regression contracts

## v2.3.5 on 19th of April, 2026

- Fix legacy `ParamSpace` sampling for huge discrete spaces so `n_permutations < total_space` no longer overflows when `total_space > sys.maxsize`
- Preserve exact legacy `random.sample(range(N), k)` ordering and downstream RNG-state semantics in the huge-space fallback path
- Add regression coverage for stdlib-equivalent range sampling, large-space reproducibility, prefix stability, and exhaustion behavior on the legacy `ParamSpace` path

## v2.4.0 on 19th of April, 2026

- Remove the repo-root `datasets/` fixture from the supported `HistoricalData` surface
- Drop `HistoricalData.DEFAULT_TEST_FILE_PATH` and `HistoricalData.DEFAULT_TEST_FILE_URL`
- Move historical-data regression coverage to a small checked-in test fixture under `tests/fixtures`
- Update first-run and historical-data examples to use `get_spot_klines()` or explicit caller-owned file paths

## v2.4.1 on 20th of April, 2026

- Harden standard-path experiment CSV serialization so commas, quotes, and embedded newlines in logged string fields are emitted safely
- Make standard-path reruns with the same `experiment_name` write the CSV header only when the file is new or empty, preventing duplicate mid-file headers
- Add regression coverage for special-string round-trips and reruns against an existing standard-path results CSV

## v2.4.2 on 20th of April, 2026

- Add structured runtime profiling to `tests/run.py`, including per-test JSON output, suite totals, and a sorted slowest-tests summary on the canonical coverage-enabled test path
- Add a committed suite runtime budget in `tests/runtime_budget.json` based on recent green `main` CI runs and enforce it in the new `PR Checks Runtime` workflow job
- Publish the runtime profile as a GitHub Actions artifact and render a runtime job summary with threshold counts and slowest tests during `PR Validation`
- Add regression coverage for runtime profile emission plus pass/fail runtime-gate behavior without touching `limen/` package code

## v2.4.3 on 22nd of April, 2026

- Add `limen.sfd.loop` subpackage to compile Loop web UI payloads into Limen SFD objects, runnable through UEL
- Export `forward_breakout_target` from `limen.features`
- Drop NaN rows alongside nulls in `Manifest.prepare_data`

## v2.5.0 on 24th of April, 2026

- Add `Cohort` constructor with strict experiment source resolution and permutation ID validation
- Enforce single-architecture selection across all requested `permutation_ids`
- Set Cohort aggregation mode at construction from architecture capability hints
- Add `Cohort.predict()` probability-weighted aggregation path with tie-to-0 behavior
- Add `Cohort.predict()` majority-vote fallback aggregation path with tie-to-0 behavior
- Add single-decoder passthrough short-circuit for direct member prediction parity
- Add `return_probs` option in `predict()` for per-member P(1) outputs
- Add `return_meta` option in `predict()` for structured inference metadata
- Add probability range validation on decoder `_probs` outputs
- Raise explicit error when `return_probs=True` is requested in fallback mode
- Add comprehensive Cohort tests for constructor validation, aggregation behavior, passthrough behavior, and output-option contracts
- Add integration test coverage validating Cohort as a decoder-compatible replacement at inference

## v2.5.1 on 25th of April, 2026

- Return `Cohort.predict(..., return_probs=True)` probability traces in sample-major order `(n_samples, n_members)` so per-bar cohort inspection matches the supported consumer contract
- Add regression coverage for sample-major probability traces, `permutation_ids` column ordering, and single-member fallback vote conversion on the canonical `tests.run` path

## v2.5.2 on 25th of April, 2026

- Compact the legacy standard-path `experiment_log` during long-running `UniversalExperimentLoop.run()` executions so repeated one-row `vstack()` growth does not degrade throughput as chunk fragmentation accumulates
- Preserve the existing live in-memory log and live CSV write behavior while keeping the standard-path log physically compact with an internal rechunk threshold
- Add regression coverage proving standard-path row ordering and CSV output stay intact while live log chunk growth remains bounded after threshold crossings

## v2.5.3 on 27th of April, 2026

- Add rule-based SFD framework (RFC-1014): boolean predicate logic over TA indicators producing per-bar long/flat signals without writing Python code
- Add predicate library (`limen.sfd.rule_based.predicates`) with threshold, relative, crossover, slope, and `sql_expr` escape-hatch types plus persistence and recency temporal modifiers
- Add `Manifest.with_strategy()` builder for rule-based strategy configuration with per-round predicate template substitution
- Add `RuleBasedStrategy` reference architecture with no-op training, boolean logic tree evaluation, and per-split backtesting
- Add `rule_based_metrics` computing Tier 1 position stats, Tier 2 per-split backtest metrics, and Tier 3 cross-split stability metrics (`sharpe_std`, `drawdown_std`, `sharpe_degradation`, `is_stable`)
- Add `limen.sfd.foundational_sfd.rule_based` foundational SFD with RSI oversold + EMA trend filter strategy and sweepable parameter domain
- Guard `UEL._finalize()` so ML-specific metrics (`experiment_confusion_metrics`, `experiment_backtest_results`) are set to `None` for rule-based runs

## v2.5.4 on 28th of April, 2026

- Apply snapshot backtest fee and slippage costs multiplicatively on entry and exit fills while preserving the fixed bps cost abstraction
- Compute snapshot `trade_*` metrics from compounded HOLD-WHILE-1 trade runs instead of individual in-market bars
- Seed snapshot max drawdown with starting equity so first-trade losses register against the initial high-water mark

## v2.5.5 on 28th of April, 2026

- Enforce strict direct snapshot backtest input contracts for non-empty data, binary `0/1` predictions, numeric price columns, and `price_change == close - open`
- Remove the snapshot bars-count mode so `trades_count` and `mean_kelly_pct` are always computed from compounded consecutive in-market runs
- Add direct snapshot regression coverage for empty input, invalid predictions, missing predictions, non-numeric prices, and inconsistent `price_change`

## v2.5.6 on 28th of April, 2026

- Make standard-path `UniversalExperimentLoop` writes honor `experiment_dir` by storing `<experiment_name>.csv` under the requested directory
- Preserve legacy standard-path CSV placement when `experiment_dir` is not set and keep MSQ artifact filenames unchanged
- Add regression coverage for standard-path `experiment_dir` CSV placement and rerun header behavior

## v3.0.0 on 29th of April, 2026

- Rename `Manifest.with_model()` to `with_reference_architecture()` and fields `model_function` → `architecture_function`, `model_params` → `architecture_params` for RFC-1004 alignment
- Rename `Manifest.with_target()` to `with_target_label()` for RFC-1004 alignment
- Remove `Manifest.fetch_data_for_env()` and `LOOP_ENV` environment variable dispatch
- Add `test_mode: bool = False` to `UniversalExperimentLoop.__init__()`: when `True` and a test data source is configured, `fetch_test_data()` is used instead of `fetch_data()`
- Trainer now always calls `manifest.fetch_data()` directly when no explicit data is provided

## v3.0.1 on 29th of April, 2026

- Add `limen.targets` module with `QuantileBinaryTarget`, `ForwardBreakoutTarget`, `ThresholdBinaryTarget`, `NextReturnTarget`, `RandomBinaryTarget`, and `IdentityTarget` class-based target transforms following the fit-on-train, transform-on-all-splits pattern
- Add `Manifest.with_target_label()` builder method for declarative target configuration; add `TargetClassConfig` dataclass and `_apply_class_based_target()` helper; harden with defensive copy of params and `RuntimeError` on wrong split order
- Migrate `logreg_binary`, `tabpfn_binary`, and `xgboost_regressor` foundational SFDs to `with_target_label()`
- Remove `compute_quantile_cutoff`, `quantile_flag`, and `forward_breakout_target` from `limen.features` (target computation is not a feature transform)
- Remove `with_target_function()`, old `with_target_label()` builder pattern (`TargetBuilder`, `FittedTransformBuilder`, `target_transforms`), and `set_target_column()`
- Add `docs/Targets.md` documenting all six built-in target classes and the custom target class convention
- Remove `limen.sfd.loop` temporary integration package (superseded by RFC-1005 YAML compiler)

## v3.0.2 on 30th of April, 2026

- Add `limen.calibration` submodule with sklearn-based probability calibration and vectorised grid threshold optimisation utilities
- Add `CalibrationBuilder` fluent API to `Manifest` for declarative calibration configuration; supports four modes — raw model, calibration-only, threshold-only, or both — controlled by per-round flags
- Add explicit callable protocols for plug-and-play custom calibrators and threshold optimisers
- Add calibration support to `LogRegBinary` and `TabPFNBinary` reference architectures; results include `optimal_threshold` and `val_score` when calibration is configured
- Remove old calibration utilities from `limen.transforms` (superseded by `limen.calibration`)
- Fix CSV column alignment when rounds return variable result keys; fix graceful shutdown on SIGINT during a round


## v3.0.3 on 6th of May, 2026

- Add `_load_sfd_module` helper to `Trainer.__init__`: prefers a `<sfd_module_name>.py` file inside `experiment_dir` (no `sys.path` mutation, SFD module not registered in `sys.modules`), falls back to `importlib.import_module` for fully-qualified package paths
- Add path-traversal hardening to `_load_sfd_module`: rejects any `sfd_module` name whose dot-separated segments are not valid Python identifiers
- Add `docs/TechnicalDebt.md` with TD-001 documenting residual `import_module` fallback hardening needed before live-trade deploy
- Update `docs/Trainer.md` to describe the two-stage SFD load, path-traversal validator, and trust contract on `experiment_dir`
- Add 3 test groups in `tests/test_trainer.py` pinning the SFD loader contract

## v3.0.4 on 6th of May, 2026

- Split the manifest class into a shared base with separate ML and rule-based subclasses; all foundational SFDs retain the same uniform interface
- `Manifest.prepare_data()` now raises `NotImplementedError` — replace any direct `Manifest()` usage for data preparation with `MLManifest()` (ML pipelines) or `RuleBasedManifest()` (rule-based pipelines)
