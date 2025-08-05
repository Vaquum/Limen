# Universal Experiment Loop

Universal Experiment Loop (UEL) is an integral part of Loop, and takes as its input data and a Single-File Model (SFM). For the time being, `UEL` can be thought of as an advanced parameter sweep. 

## The Meaning of Parameter Sweep (the action)

It is typically thought that the focus of the sweep is specifically the model hyperparameters, and only these. This led to the bastardized term "hyperparameter optimization". This perspective is extremely limiting and entirely misses the point of parameter sweeping. 

In short, the point of parameter sweeping is that since such a practice is possible, and since more or less anything and everything can be readily parametrized, there should be no limit to where this approach can be applied. 

**Not only the idea of sweeping through parameters can be extended beyond the model and its hyperparameters, to data fetching, data pre-processing, feature engineering, and all other aspects of classifier development lifecycle, but it can also be extended well beyond input arguments. For example, conditional logic can be handled as parameters, and even individual fragments of code can be fully parametric, and therefore a subject of a parameter sweep.**

In other words, the idea of performing a parameter sweep is equally relevant to all of the above-mentioned ten steps. This is a crucial key point, and our success depends on undertanding it, putting it into practice, and realizing its unrestrained power to yield the most meaningful probabilities for live trading at any given point in time, regardless of the prevailing circumstances.

## Data

A key point here is that all individual contributors work based on the same underlying data. We achieve this by always calling data from these provided endpoints. If you don't find what you need through these endpoints, [make an issue](https://github.com/Vaquum/Loop/issues/new) that requests the data that you need. 

Read more in [HistoricalData](Historical-Data.md)

## SFM

An SFM contains all parameters, data preparation code, and model operation codes in a single python file. For example, representing an XGBoost bullish regime binary classifier.

Read more in [Universal Experiment Loop](Universal-Experiment-Loop.md).

## `UniversalExperimentLoop`

Initializes the Universal Experiment Loop.

### Args

| Parameter             | Type               | Description                                           |
|-----------------------|--------------------|-------------------------------------------------------|
| `data`                | `pl.DataFrame`     | The data to use for the experiment.                   |
| `single_file_model`   | `SingleFileModel`  | The single file model to use for the experiment.      |


### `run`

Runs the experiment `n_permutations` times. 

**NOTE:** You can always over-ride any or all of `params` or `prep` or `model` by passing them through here as input arguments. Just make sure that the inputs and returns are same as the ones outlined in [Single File Model](Single-File-Model.md).

### Args

| Parameter                     | Type       | Description                                                      |
|-------------------------------|------------|------------------------------------------------------------------|
| `experiment_name`             | `str`      | The name of the experiment.                                      |
| `n_permutations`              | `int`      | The number of permutations to run.                               |
| `prep_each_round`             | `bool`     | Whether to use `prep` for each round or just the first.         |
| `random_search`               | `bool`     | Whether to use random search or not.                             |
| `maintain_details_in_params`  | `bool`     | Whether to maintain experiment details in `params`.              |
| `context_params`              | `dict`     | The context parameters to use for the experiment.                |
| `save_to_sqlite`              | `bool`     | Whether to save the results to a SQLite database.               |
| `params`                      | `dict`     | The parameters to use for the experiment.                        |
| `prep`                        | `function` | The function to use to prepare the data.                         |
| `model`                       | `function` | The function to use to run the model.                            |

### Returns

Adds various artifacts into `UniversalExperimentLoop` class object. 