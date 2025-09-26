import numpy as np
import polars as pl

from loop.metrics.binary_metrics import binary_metrics
from loop.utils.splits import split_sequential
from loop.utils.splits import split_data_to_prep_output
from loop.manifest import Manifest

def manifest():
    return (Manifest()
        .set_split_config(3, 1, 1)
        .set_required_bar_columns([
            'datetime', 'high', 'low', 'close', 'volume', 'maker_ratio',
            'no_of_trades'
        ])
        .with_target('outcome')
            .add_transform(lambda data: data.with_columns(
                pl.Series('outcome', np.random.randint(0, 2, size=data.height))
            ))
            .add_transform(lambda data: data[:-100])
            .done()
    )

def params(): 

    return {
        'random_weights': [0.4, 0.5, 0.6],
        'breakout_threshold': [0.05, 0.1, 0.2],
        'shift': [-1, -2, -3]
    }


def prep(data, round_params, manifest):

    return manifest.prepare_data(data, round_params)


def model(data, round_params):

    weights = [round_params['random_weights'], 1 - round_params['random_weights']]

    preds = np.random.choice([0, 1], size=len(data['x_test']), p=weights)
    probs = np.random.choice([0.1, 0.9], size=len(data['x_test']), p=weights)

    round_results = binary_metrics(data, preds, probs)
    round_results['_preds'] = preds

    return round_results
