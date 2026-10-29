"""Caller-owned residual learning on chronological recorded observations."""
import hashlib

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import research as project


# region residual_fit
def fit_residual(train, depth):
    from xgboost import XGBRegressor

    x = train.select('momentum').to_numpy()
    y = train['next_return'].to_numpy()
    baseline = make_pipeline(StandardScaler(), Ridge(alpha=1., solver='svd')).fit(x, y)
    residuals = y - baseline.predict(x)
    correction = XGBRegressor(n_estimators=8, max_depth=depth, learning_rate=.1,
                              random_state=42, n_jobs=1).fit(x, residuals)
    return baseline, correction


def residual_predict(fitted, frame):
    baseline, correction = fitted
    x = frame.select('momentum').to_numpy()
    return baseline.predict(x) + correction.predict(x)
# endregion residual_fit


# region residual_research
def residual_research(raw, output):
    output.mkdir(parents=True)
    partitions = [project.prepare_split(frame, {'window': 4})
                  for frame in (raw[:864], raw[864:1152], raw[1152:1440])]
    train, validation, test = partitions
    retained = {depth: fit_residual(train, depth) for depth in (1, 2)}
    rows = []
    for depth, fitted in retained.items():
        forecasts = residual_predict(fitted, validation)
        rows.append({'depth': depth, 'validation_net_return': project.backtest(validation, forecasts)})
        for name, frame in zip(('train', 'validation', 'test'), partitions, strict=True):
            x = frame.select('momentum').to_numpy()
            baseline, correction = fitted
            frame.select('datetime', 'momentum', 'next_return').with_columns(
                pl.Series('baseline', baseline.predict(x)),
                pl.Series('correction', correction.predict(x)),
                pl.Series('forecast', residual_predict(fitted, frame)),
            ).write_parquet(output / (str(depth) + '-' + name + '.parquet'))
    selected = sorted(rows, key=lambda row: (-row['validation_net_return'], row['depth']))[0]
    forecasts = residual_predict(retained[selected['depth']], test)
    return {'candidates': rows, 'selected_depth': selected['depth'],
            'partition_rows': [frame.height for frame in partitions],
            'test_rmse': float(np.sqrt(np.mean((forecasts - test['next_return'].to_numpy()) ** 2))),
            'test_net_return': project.backtest(test, forecasts),
            'artifacts': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(output.glob('*.parquet'))}}
# endregion residual_research
