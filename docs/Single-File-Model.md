# Single-File Model

## Data

A key point here is that all individual contirbutors work based on the same underlying data. We achieve this by always calling data from these provided endpoints. If you don't find what you need through these endpoints, [make an issue](https://github.com/Vaquum/Loop/issues/new) that requests the data that you need. 

## SFM Overview

The Single-File Model (SFM) is a convenient way to bring together all artifacts related with a model to be used in an experiment into a single file. These files live in [`loop/models`]('loop/models'). Once an SFM is added to the package, it becomes available to be used as input for `Loop.UniversalExperimentLoop`. 

There are few things to note about an SFM: 

1) An SFM has three functions: `params`, `prep`, and `model`
2) `params` takes no input and returns a dictionary with keys as parameter names, and lists as parameter values.
3) `prep` takes as input data from `Loop.HistoricalData` and  and returns a dictionary with data assets.
4) `model` takes as input dictionary with data assets and `round_params` and returns `round_results` dictionary.

As long as the SFM in question adheres to points 2 through 4, `Loop.UniversalExperimentLoop` will accept the SFM as input. 

# Notes on Usage

- Paramater values returned by `params` have to always be in a list, even if it is a single value
- Consequently, if you want to use a constant value but have it in `params` for reference, just have a single value in a list for that key
- If you want to perform per-permutation transformations to parameters, handle them inside `model`