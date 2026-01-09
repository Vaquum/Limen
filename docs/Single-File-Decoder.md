# Single File Decoder

The Single File Decoder (SFD) is a convenient way to bring together all artifacts related with a model to be used in an experiment into a single file. These files live in [`limen/sfd`](../limen/sfd). Once an SFD is added to the package, it becomes available to be used as input for [`limen.UniversalExperimentLoop`](Universal-Experiment-Loop.md).

There are two categories of SFDs: **Foundational SFDs** are the official reference SFDs provided by Loop (all use manifest-based configuration), while **Custom SFDs** are user-defined SFDs with flexibility to use either manifest-based configuration or custom implementation of data preparation and model functions.

## SFD Implementation Approaches

### Manifest-Based Configuration (Recommended)

SFDs using manifest-based configuration define the experiment pipeline declaratively. This approach:
- Configures data sources for both production and testing
- Auto-generates data preparation functions from manifest directives
- Enables automatic data fetching in UEL
- Ensures experiment reproducibility

**Required functions:**
- `params()` - Parameter space definition
- `manifest()` - Returns an [Experiment Manifest](Experiment-Manifest.md) configuring the entire pipeline

### Custom Functions Approach

SFDs using the custom functions approach directly implement data preparation and model operations. This approach:
- Requires explicit data to be passed to UEL
- Custom implementation of data preparation logic
- Provides full control for specialized use cases

**Required functions:**
- `params()` - Parameter space definition
- `prep()` - Data preparation implementation
- `model()` - Model training and prediction implementation

## Function Requirements

SFD is a standardized format so certain requirements must be met. The sections below document the requirements for each function type.

### Required for All SFDs: `params`

Contains all parameters and their value ranges to be used in the parameter sweep.

Takes no input and returns a dictionary with keys as parameter names, and lists as parameter values. These set the boundaries for the parameter space to be used in the sweep. The success of your experiment greatly depends on the parameters and their respective parameter value ranges, so choose them well. 

**NOTE**: Generally speaking, it's best to start with as many parameters, with as broad parameter value ranges as possible.

#### REQUIREMENTS

- The output is `round_params` a dictionary where each key has a list as its value
- Individual parameter values can be any scalar values; integers, floats, strings, functions, etc. 
- Individual parameter cannot be aggregate types; lists, tuples, arrays, or objects
- Parameter values in the `round_params` dictionary returned by `params` have to always be in a list, even if it is a single value.

**NOTE**: Parameters can be used to parametrize other parameters, for example, where one parameter is a function, and another parameter is an input argument to that function. Such higher-order parameters can be an extremely powerful way to make Loop play the song of Bitcoin.

### Manifest-Based SFDs: `manifest`

For manifest-based SFDs, the `manifest()` function returns an [Experiment Manifest](Experiment-Manifest.md) that declaratively configures the entire experiment pipeline. This replaces custom implementation of `prep()` and `model()` functions.

The manifest enables:
- **Data source configuration**: Specify production and test data sources
- **Auto-generated data preparation**: Define indicators, features, and transformations declaratively
- **Automatic data fetching**: UEL fetches data based on `LOOP_ENV` (defaults to 'test')
- **Reproducible experiments**: All pipeline configuration in one place

See [Experiment Manifest](Experiment-Manifest.md) for complete documentation, requirements, and examples.

### Custom Functions: `prep` and `model`

The following sections document custom SFD functions for implementations requiring full control over data preparation and model operations.

#### `prep`

Contains all data preparation procedures used in the parameter sweep.

Takes as input data from `limen.HistoricalData.data` and `round_params` which is a dictionary with single value per key. It returns `data_dict`, a  dictionary yielded by `utils.splits.split_data_to_prep_output` where arbitrary key-values can be added before returning the `data_dict`. 

##### REQUIREMENTS

- The input must contain at least `historical.data` but must also contain `round_params` when `uel.run(prep_each_round=True)`
- The input data must always have `datetime` when it is ingested in `prep`
- The function must start with `all_datetimes = data['datetime'].to_list()` immediate after declaration
- The column `datetime` must be in data when it is passed to `split_data_to_prep_output`, where it will be automatically removed
- There must be no randomness; permutation parameters must govern all `prep` operations
 - Prefer deterministic `prep` fully governed by `round_params`. If randomness is required (e.g., sampling), fix seeds parametrically so that per-round reconstruction in `Log` remains aligned with stored predictions.

**NOTE:** If a scaler is fitted as part of `prep`, it can be added to `round_results[_scaler]` for use in subsequent folds.

#### `model` 

Contains the model architecture and all model operation procedures up to predictions on test data.

Takes as input a `data_dict` dictionary yielded by `utils.splits.split_data_to_prep_output`, together with `round_params`. It returns `round_results` dictionary yielded by `limen.metrics.binary_metrics`, `limen.metrics.multiclass_metrics`, or `limen.metrics.continuous_metrics`. 

##### REQUIREMENTS

- The input must accept at least `data` and optionally also `round_params`
- The output must come from one of `binary_metrics`, `multiclass_metrics`, and `continuous_metrics` in `limen.metrics`.

**NOTE**: Any scalar metrics you want recorded into the experiment log must be included directly in `round_results` (returned by the metrics helpers). Use `round_results['extras']` only for complex objects (models, arrays, DataFrames) that should not be flattened into the log. To persist test-set predictions for post-run analysis, set `round_results['_preds'] = ...` (UEL will collect these into `uel.preds`).

# Appendix

## Model Categories

The below table is provided as a guide for identifying model architecture candidates for SFDs.

| **Model family** | **Data it tends to shine on** | **Bitcoin-trading sweet-spot** | **Key caveats & blind spots** |
| --- | --- | --- | --- |
| **Linear models** (OLS, logistic, linear SVM, etc.) | Clean numeric features where the signal is roughly additive & linear (e.g. factor exposures, lagged returns, volume, sentiment scores after aggregation) | Quick-and-dirty return or factor-beta forecasts; stress-testing hedge ratios; intraday alpha when relationships are known to be linear | Fails when relationships are nonlinear or regime-dependent; assumes stationarity; sensitive to multicollinearity & outliers |
| **Tree-based models** (Decision Tree, Random/Extra Forest, XGBoost, LightGBM) | Mixed numeric / categorical features with lots of nonlinear interactions (order-book metrics, on-chain indicators, macro covariates) | Regime classification (bull/bear/sideways), “will the next hour’s move exceed _x_%?” probability, feature-importance discovery | Single tree is unstable; ensembles can overfit noisy crypto data if validation splits leak; poor at extrapolating beyond seen feature range |
| **Kernel & similarity methods** (Kernel SVM/Ridge, Gaussian Process) | Small-to-medium data sets where smooth nonlinear decision boundaries help (e.g. engineered technical-indicator vectors) | Detect subtle micro-structure patterns for very short-horizon direction calls; GP for probabilistic price forecasts with uncertainty bars | Scales ~O(n²) or worse—impractical for millions of ticks; kernel choice is art; hyper-parameter tuning costly |
| **Instance-based / lazy learners** (k-NN, radius-NN, locally weighted regression) | Problems where “look-alike past windows” matter: rolling embeddings of OHLCV, volatility motifs | Find “analog days” or “analog order-book snapshots” to judge likely next-bar move; quick anomaly-detection for data-quality checks | Curse of dimensionality; query-time latency; noisy crypto features give misleading neighbors; no built-in feature weighting |
| **Neural networks** (MLP, CNN, LSTM, Transformer) | Large raw or lightly processed sequences/images/text (tick-level returns, limit-order-book depth, Twitter news) | Sequence-to-sequence price-path forecasting, volatility forecasting, social-sentiment + price joint modeling, market-making signal generation | Data-hungry; black-box; overfits regime shifts; requires heavy compute & careful walk-forward validation |
| **Probabilistic / graphical models** (Naïve Bayes, Hidden Markov Model, Bayesian VAR, GARCH w/ Bayesian priors) | Structured data where quantifying uncertainty is key; low-frequency macro + on-chain metrics | Hidden-state regime detection (e.g. “high-vol” vs “sleepy”), Bayesian volatility forecasting, drawdown probability estimation | Often assume specific distributions (Gaussian, Student-t); inference slow for high-dim models; hyper-prior choices heavily influence results |
| **Ensembles / meta-learners** (Bagging, Boosting, Stacking, Super Learner) | Heterogeneous feature sets and base learners; you have many weak models already | “Blender” that combines short-horizon alpha models, or merges technical + on-chain + sentiment signals for robust direction probability | Harder to interpret; danger of leakage when stacking on expanding windows; compute & memory heavy if each base learner is big |
| **Rule & symbolic learners** (RIPPER, decision-rule lists, association-rule classifiers) | Domains requiring auditability & simple if-then logic | Generating human-readable entry/exit conditions (“IF RSI < 30 AND funding rate < 0 THEN long”) | Limited capacity for subtle nonlinearities; prone to overfit discrete crypto patterns; brittle when regime changes |
| **Clustering & latent-structure models** (k-Means, DBSCAN, Gaussian Mixture, spectral clustering) | Unlabeled episodes you want to group (daily returns distributions, wallet-flow profiles) | Discover market regimes (volume/volatility clusters), flag abnormal days, segment traders by behavior | No guarantee clusters map to profitable actions; cluster count/shape subjective; sensitive to scale and feature choice |
| **Dimensionality-reduction models** (PCA, t-SNE, UMAP, autoencoders) | Very high-dimensional feature spaces (full order-book snapshots, dozens of indicators) | Compress L2-book states into a handful of factors fed to downstream predictive model; visualize regime transitions | Possible loss of predictive information; unsupervised compression may ignore price relevance; t-SNE/UMAP unstable across runs |
| **Reinforcement-learning agents** (Q-Learning, DQN, Policy Gradient, Actor-Critic) | Sequential decision tasks with explicit reward feedback (execution, allocation) | Optimal trade sizing & timing, market-making spread setting, adaptive algorithmic execution in changing liquidity | Sample-inefficient; simulators rarely match true slippage/latency; non-stationary crypto microstructure can break trained policies |
