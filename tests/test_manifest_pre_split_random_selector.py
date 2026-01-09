import polars as pl
import numpy as np
from datetime import datetime, timedelta
from limen.experiment import Manifest
from limen.data.utils import random_slice


def test_pre_split_random_selector():
    np.random.seed(42)
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(1000)]
    data = pl.DataFrame({
        'datetime': dates,
        'value': np.random.randn(1000),
        'target': np.random.randn(1000)
    })

    manifest = (Manifest()
        .set_pre_split_data_selector(
            random_slice,
            rows='random_slice_size',
            safe_range_low='random_slice_min_pct',
            safe_range_high='random_slice_max_pct',
            seed='random_seed'
        )
        .set_split_config(6, 2, 2)
        .with_target('target')
            .done()
    )

    round_params = {
        'random_slice_size': 100,
        'random_slice_min_pct': 0.25,
        'random_slice_max_pct': 0.75,
        'random_seed': 42
    }

    data_dict = manifest.prepare_data(data, round_params)

    total_rows = len(data_dict['x_train']) + len(data_dict['x_val']) + len(data_dict['x_test'])
    assert total_rows == round_params['random_slice_size']

    data_dict2 = manifest.prepare_data(data, round_params)
    assert data_dict['x_train'].equals(data_dict2['x_train'])

    round_params['random_seed'] = 123
    data_dict3 = manifest.prepare_data(data, round_params)
    assert not data_dict['x_train'].equals(data_dict3['x_train'])
