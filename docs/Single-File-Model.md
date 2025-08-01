# Single-File Model

The Single-File Model (SFM) is a convenient way to bring together all artifacts related with a model to be used in an experiment into a single file. These files live in [`loop/models`]('loop/models'). Once an SFM is added to the package, it becomes available to be used as input for `Loop.UniversalExperimentLoop`. 

## Requirements

SFM is a standardized format so certain requirements must be met. For example, there are strict requirements for inputs and outputs of the three required functions of the SFM.

### Must-have three functions: `params`, `prep`, and `model`

#### `params`

Takes no input and returns a dictionary with keys as parameter names, and lists as parameter values. These set the boundaries for the parameter space to be used in the sweep.

#### `prep`

Takes as input data from `loop.HistoricalData.data` and `round_params` which is a dictionary with single value per key. It returns a dictionary yielded by `utils.splits.split_data_to_prep_output` where additional key-values are allowed. 

**NOTE:** If a scaler is fitted as part of `prep`, it can be added to `round_results[_scaler]` for use in subsequent folds.

#### `model` 

Takes as input a dictionary yielded by `utils.splits.split_data_to_prep_output` and `round_params`. It returns `round_results` dictionary yielded by `loop.metrics.binary_metrics`, `loop.metrics.multiclass_metrics`, or `loop.metrics.continuous_metrics`. 

**NOTE**: Additional metrics must be added to `round_result['extras']` for those additional metrics to end up in `uel.log_df`. 














As long as the SFM in question adheres to points 2 through 4, `Loop.UniversalExperimentLoop` will accept the SFM as input. 








# Notes on Usage

- Paramater values returned by `params` have to always be in a list, even if it is a single value
- Consequently, if you want to use a constant value but have it in `params` for reference, just have a single value in a list for that key
- If you want to perform per-permutation transformations to parameters, handle them inside `model`