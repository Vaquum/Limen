import polars as pl

from sklearn.linear_model import LogisticRegression

from loop.metrics.binary_metrics import binary_metrics
from loop.features import quantile_flag, kline_imbalance, vwap
from loop.indicators import wilder_rsi, atr, ppo, roc
from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.transforms.logreg_transform import LogRegTransform
from loop.manifest import Manifest, process_manifest

def manifest():
    return Manifest(
        name='logreg_manifest_split_first_v1',
        description='Logistic regression with Universal Split-First architecture',

        required_columns=[
            'datetime',
            'high',
            'low',
            'open',
            'close',
            'mean',
            'std',
            'median',
            'iqr',
            'volume',
            'maker_ratio',
            'no_of_trades'
        ],

        indicators=[
            (roc, {'period': lambda p: p['roc_period']}),
            (atr, {'period': 14}),
            (ppo, {}),
            (wilder_rsi, {}),
        ],

        features=[
            (vwap, {}),
            (kline_imbalance, {}),
        ],

        target_column='quantile_flag',
        split_config=(8,1,2),
        transformations=[],
    )

def params():

    return {
        # data prep parameters
        'shift': [-1, -2, -3, -4, -5],
        'q': [0.35, 0.38, 0.41, 0.44, 0.47, 0.50, 0.53],
        'roc_period': [1, 4, 12, 24, 144],
        'penalty': ['l2'],
        # classifier parameters
        'class_weight': [0.45, 0.55, 0.65, 0.75, 0.85],
        'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'max_iter': [30, 60, 90, 120, 180, 240],
        'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'newton-cholesky'],
        'tol': [0.001, 0.01, 0.03, 0.1, 0.3],
    }


def prep(data, round_params, manifest):

    all_datetimes = data['datetime'].to_list()

    # Column names (resolved with round_params)
    roc_col = f"roc_{round_params['roc_period']}"

    split_data = split_sequential(data, manifest.split_config)

    for i in range(len(split_data)):
        split_data[i] = process_manifest(manifest, split_data[i], round_params)

    # Step 3: Apply target transformations with fit/transform pattern

    # Fit quantile_flag on training data only
    split_data[0], train_cutoff = quantile_flag(
        data=split_data[0],
        col=roc_col,
        q=round_params['q'],
        return_cutoff=True
    )

    # Apply the same cutoff to validation and test sets (transform only)
    for i in range(1, len(split_data)):
        split_data[i] = quantile_flag(
            data=split_data[i],
            col=roc_col,
            q=round_params['q'],
            cutoff=train_cutoff
        )

    # Apply target shifting to all splits
    for i in range(len(split_data)):
        split_data[i] = split_data[i].with_columns(
            pl.col("quantile_flag")
                .shift(round_params['shift'])
                .alias("quantile_flag")).drop_nulls("quantile_flag")

    # Auto-discover columns from processed data
    available_cols = list(split_data[0].columns)

    # Assert required columns are present
    for required_col in manifest.required_columns:
        assert required_col in available_cols, f"Required column '{required_col}' not found in data"

    # Build final column order: auto-discovered + target_column last
    cols = []

    # Add all available columns except target
    for col in available_cols:
        if col != manifest.target_column:
            cols.append(col)

    # Add target column last (if it exists)
    if manifest.target_column and manifest.target_column in available_cols:
        cols.append(manifest.target_column)

    # Create data dictionary from splits
    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)
    # Scale features using training data statistics (fit/transform pattern)
    scaler = LogRegTransform(data_dict['x_train'])

    for col in data_dict.keys():
        if col.startswith('x_'):
            data_dict[col] = scaler.transform(data_dict[col])

    data_dict['_scaler'] = scaler

    return data_dict


def model(data: dict, round_params):

    clf = LogisticRegression(
        solver=round_params['solver'],
        penalty=round_params['penalty'],
        dual=False,
        tol=round_params['tol'],
        C=round_params['C'],
        fit_intercept=True,
        intercept_scaling=1,
        class_weight={0: round_params['class_weight'], 1: 1},
        random_state=None,
        max_iter=round_params['max_iter'],
        verbose=0,
        warm_start=False,
        n_jobs=None,
    )

    clf.fit(data['x_train'], data['y_train'])

    preds = clf.predict(data['x_test'])
    probs = clf.predict_proba(data['x_test'])[:, 1]

    round_results = binary_metrics(data, preds, probs)
    round_results['_preds'] = preds

    return round_results