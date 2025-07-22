import numpy as np
import polars as pl

from loop.utils.metrics import binary_metrics
from loop.utils.splits import split_sequential
from loop.utils.splits import split_data_to_prep_output

def params(): 

    return {'random_weights': [0.4, 0.5, 0.6], 
            'breakout_threshold': [0.05, 0.1, 0.2],
            'shift': [-1, -2, -3]}


def prep(data, round_params=None):

    data = data.with_columns([
        (
            ((pl.col("high") - pl.col("low")) / pl.col("open") * 100) 
            .gt(round_params['breakout_threshold'])
        ).cast(pl.UInt8)
         .shift(round_params['shift'])
         .alias("high_max_delta")
    ]).drop_nulls("high_max_delta")

    cols = ['high', 'low', 'close', 'volume', 'maker_ratio', 'no_of_trades', 'high_max_delta']
    
    split_data = split_sequential(data, (3, 1, 1))

    return split_data_to_prep_output(split_data=split_data, cols=cols)


def model(data, round_params):

    weights = [round_params['random_weights'], 1 - round_params['random_weights']]

    preds = np.random.choice([0, 1], size=len(data['x_test']), p=weights)

    return binary_metrics(data, preds)
