import numpy as np
from tabpfn import TabPFNClassifier

from limen.metrics.binary_metrics import binary_metrics
from limen.transforms.calibrate_classifier import calibrate_classifier
from limen.transforms.optimize_binary_threshold import optimize_binary_threshold
from limen.utils.data_dict_to_numpy import data_dict_to_numpy


# TabPFN model checkpoint - auto-downloaded by tabpfn library on first use
TABPFN_MODEL_PATH = 'tabpfn-v2-classifier-v2_default.ckpt'

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.05
DEFAULT_THRESHOLD = 0.35


def tabpfn_binary(data: dict,
                  n_ensemble_configurations: int = 4,
                  device: str = 'cpu',
                  use_calibration: bool = True,
                  threshold_metric: str = 'balanced') -> dict:

    '''
    Compute TabPFN binary classification with dynamic threshold tuning on validation set.

    Args:
        data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test
        n_ensemble_configurations (int): Number of ensemble configurations for TabPFN
        device (str): Device to run on ('cpu' or 'cuda')
        use_calibration (bool): Whether to apply isotonic calibration on validation set
        threshold_metric (str): Metric to optimize ('f1', 'precision', 'accuracy', 'balanced')

    Returns:
        dict: Results from binary_metrics with '_preds', 'optimal_threshold', 'val_score' added
    '''

    arrays = data_dict_to_numpy(data, ['x_train', 'y_train', 'x_val', 'y_val', 'x_test'])
    x_train = arrays['x_train']
    y_train = arrays['y_train']
    x_val = arrays['x_val']
    y_val = arrays['y_val']
    x_test = arrays['x_test']

    clf = TabPFNClassifier(
        device=device,
        n_estimators=n_ensemble_configurations,
        model_path=TABPFN_MODEL_PATH,
        ignore_pretraining_limits=True
    )

    clf.fit(x_train, y_train)

    if use_calibration:
        y_val_proba, y_test_proba = calibrate_classifier(
            clf, x_val, y_val, [x_val, x_test], method='isotonic'
        )
    else:
        y_val_proba = clf.predict_proba(x_val)[:, 1]
        y_test_proba = clf.predict_proba(x_test)[:, 1]

    best_threshold, best_score = optimize_binary_threshold(
        y_val,
        y_val_proba,
        threshold_min=THRESHOLD_MIN,
        threshold_max=THRESHOLD_MAX,
        threshold_step=THRESHOLD_STEP,
        default_threshold=DEFAULT_THRESHOLD,
        metric=threshold_metric
    )

    y_pred = (y_test_proba >= best_threshold).astype(np.int8)

    round_results = binary_metrics(data, y_pred, y_test_proba)
    round_results['_preds'] = y_pred
    round_results['optimal_threshold'] = best_threshold
    round_results['val_score'] = best_score

    return round_results
