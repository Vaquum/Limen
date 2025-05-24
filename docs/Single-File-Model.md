# Single-File Model

The Single-File Model (SFM) is a convenient way to bring together all artifacts related with a model to be used in an experiment into a single file. These files live in [`loop/models`]('loop/models'). Once an SFM is added to the package, it becomes available to be used as input for `Loop.UniversalExperimentLoop`. 

There are few things to note about an SFM: 

1) An SFM has three functions: `params`, `prep`, and `model`
2) `params` takes no input and returns a dictionary with keys as parameter names, and lists as parameter values.
3) `prep` takes as input data from `Loop.HistoricalData` and `round_params`, and returns a dictionary with data assets and `round_params` dictionary
4) `model` takes as input dictionary with data assets and `round_params`

As long as the SFM in question adheres to points 2 through 4, `Loop.UniversalExperimentLoop` will accept the SFM as input. 

