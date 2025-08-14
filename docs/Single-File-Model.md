# Single-File Model

The Single-File Model (SFM) is a convenient way to bring together all artifacts related with a model to be used in an experiment into a single file. These files live in [`loop/sfm`](../loop/sfm). Once an SFM is added to the package, it becomes available to be used as input for [`Loop.UniversalExperimentLoop`](Universal-Experiment-Loop.md). 

## Requirements

SFM is a standardized format so certain requirements must be met. For example, there are strict requirements for inputs and outputs of the three required functions of the SFM.

### Must-have three functions: `params`, `prep`, and `model`

#### `params`

Contains all parameters and their value ranges to be used in the parameter sweep.

Takes no input and returns a dictionary with keys as parameter names, and lists as parameter values. These set the boundaries for the parameter space to be used in the sweep.

**NOTE**: Paramater values in the `round_params` dictionary returned by `params` have to always be in a list, even if it is a single value.

#### `prep`

Contains all data preparation procedures used in the parameter sweep.

Takes as input data from `loop.HistoricalData.data` and `round_params` which is a dictionary with single value per key. It returns a `data_dict` dictionary yielded by `utils.splits.split_data_to_prep_output` where arbitrary key-values can be added before returning the `data_dict`. 

##### REQUIREMENTS

- The input must contain at least `historical.data` but can also contain `round_params` when `uel.run(prep_each_round=True)`
- The input data must always have `datetime` when it is ingested in `prep`
- The column `datetime` must be in data when it is passed to `split_data_to_prep_output`, where it will be automatically removed

**NOTE:** If a scaler is fitted as part of `prep`, it can be added to `round_results[_scaler]` for use in subsequent folds.

#### `model` 

Contains the model architecture and all model operation procedures up to predictions on test data.

Takes as input a `data_dict` dictionary yielded by `utils.splits.split_data_to_prep_output`, together with `round_params`. It returns `round_results` dictionary yielded by `loop.metrics.binary_metrics`, `loop.metrics.multiclass_metrics`, or `loop.metrics.continuous_metrics`. 

**NOTE**: Additional metrics must be added to `round_result['extras']` for those additional metrics to end up in `uel.log_df`. 

# Appendix

## Model Categories

The below table is provided as a guide for identifying model architecture candidates for SFMs.

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
