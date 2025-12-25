import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, accuracy_score
from tabpfn import TabPFNClassifier

from loop.metrics.binary_metrics import binary_metrics


TABPFN_MODEL_PATH = 'tabpfn-v2-classifier-v2_default.ckpt'

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.05
DEFAULT_THRESHOLD = 0.35


def tabpfn_binary_dynamic(data: dict,
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

    X_train = data['x_train'].to_numpy() if hasattr(data['x_train'], 'to_numpy') else data['x_train']
    y_train = data['y_train'].to_numpy() if hasattr(data['y_train'], 'to_numpy') else data['y_train']
    X_val = data['x_val'].to_numpy() if hasattr(data['x_val'], 'to_numpy') else data['x_val']
    y_val = data['y_val'].to_numpy() if hasattr(data['y_val'], 'to_numpy') else data['y_val']
    X_test = data['x_test'].to_numpy() if hasattr(data['x_test'], 'to_numpy') else data['x_test']

    if len(y_train.shape) > 1:
        y_train = y_train.ravel()
    if len(y_val.shape) > 1:
        y_val = y_val.ravel()

    clf = TabPFNClassifier(
        device=device,
        n_estimators=n_ensemble_configurations,
        model_path=TABPFN_MODEL_PATH,
        ignore_pretraining_limits=True
    )

    clf.fit(X_train, y_train)

    if use_calibration:
        calibrator = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
        calibrator.fit(X_val, y_val)
        y_val_proba = calibrator.predict_proba(X_val)[:, 1]
        y_test_proba = calibrator.predict_proba(X_test)[:, 1]
    else:
        y_val_proba = clf.predict_proba(X_val)[:, 1]
        y_test_proba = clf.predict_proba(X_test)[:, 1]

    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
    best_threshold = DEFAULT_THRESHOLD
    best_score = -1.0

    def balanced_metric(y_true, y_pred):

        if y_pred.sum() == 0:
            return 0.0
        prec = precision_score(y_true, y_pred, zero_division=0)
        trade_rate = y_pred.sum() / len(y_pred)
        return prec * np.sqrt(trade_rate)

    metric_fn = {
        'f1': lambda y, p: f1_score(y, p, zero_division=0),
        'precision': lambda y, p: precision_score(y, p, zero_division=0),
        'accuracy': accuracy_score,
        'balanced': balanced_metric,
    }.get(threshold_metric, balanced_metric)

    for thresh in thresholds:
        y_val_pred = (y_val_proba >= thresh).astype(np.int8)
        if y_val_pred.sum() == 0:
            continue
        score = metric_fn(y_val, y_val_pred)
        if score > best_score:
            best_score = score
            best_threshold = thresh

    y_pred = (y_test_proba >= best_threshold).astype(np.int8)

    round_results = binary_metrics(data, y_pred, y_test_proba)
    round_results['_preds'] = y_pred
    round_results['optimal_threshold'] = best_threshold
    round_results['val_score'] = best_score

    return round_results
