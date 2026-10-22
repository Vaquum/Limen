"""Caller-owned three-class evaluation on recorded next-close returns."""
import pickle

import numpy as np
import polars as pl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from limen.metrics.multiclass_metrics import multiclass_metrics
from limen.metrics.safe_ovr_auc import safe_ovr_auc


def learned_state(fitted):
    scaler = fitted.named_steps['standardscaler']
    model = fitted.named_steps['kneighborsclassifier']
    return pickle.dumps((scaler.mean_, scaler.scale_, model.classes_, model._fit_X, model._y))


# region multiclass_evaluation
def evaluate_three_classes(raw):
    def prepare(frame):
        return frame.with_columns(
            (pl.col('close') / pl.col('close').shift(6) - 1).alias('momentum'),
            (pl.col('close').shift(-1) / pl.col('close') - 1).alias('next_return'),
        ).drop_nulls(['momentum', 'next_return'])

    train, evaluation = prepare(raw[:864]), prepare(raw[864:1152])
    cutoffs = np.quantile(train['next_return'].to_numpy(), [.25, .75])
    labels = [np.digitize(frame['next_return'].to_numpy(), cutoffs) for frame in (train, evaluation)]
    fitted = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
    fitted.fit(train.select('momentum'), labels[0])
    state = learned_state(fitted)
    probabilities = fitted.predict_proba(evaluation.select('momentum'))
    predicted = fitted.predict(evaluation.select('momentum'))
    metrics = multiclass_metrics({'y_test': labels[1]}, predicted, probabilities)
    if learned_state(fitted) != state:
        raise ValueError('Evaluation changed learned state in the three-class model')
    missing_class = safe_ovr_auc(np.zeros(len(predicted), dtype=int), probabilities)
    return {'cutoffs': cutoffs.tolist(), 'classes': fitted.classes_.tolist(),
            'training_rows': train.height, 'evaluation_rows': evaluation.height,
            'truth': labels[1].tolist(), 'predictions': predicted.tolist(),
            'probabilities': probabilities.tolist(), 'metrics': metrics,
            'missing_class_auc': None if np.isnan(missing_class) else float(missing_class),
            'fitted_state_unchanged': True}
# endregion multiclass_evaluation
