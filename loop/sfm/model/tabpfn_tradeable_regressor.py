import numpy as np
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from sklearn.preprocessing import StandardScaler


def tabpfn_tradeable_regressor(data: dict,
                                 n_estimators: int = 4,
                                 device: str = 'cpu',
                                 random_state: int = 42,
                                 **kwargs) -> dict:

    '''
    Compute TabPFN regression predictions for tradeable score.

    Args:
        data (dict): Data dictionary with train/val/test splits and tradeable-specific entries
        n_estimators (int): Number of TabPFN estimators (default: 4)
        device (str): Device for TabPFN ('cpu' or 'mps') (default: cpu)
        random_state (int): Random seed (default: 42)
        **kwargs: Additional parameters (ignored)

    Returns:
        dict: Results with models, metrics, predictions and tradeable-specific extras
    '''

    train_data = data['_train_clean']
    val_data = data['_val_clean']

    y_train = train_data.select('tradeable_score').to_numpy().flatten()
    y_val = val_data.select('tradeable_score').to_numpy().flatten()

    numeric_features = data['_numeric_features']
    X_train = train_data.select(numeric_features).to_numpy()
    X_val = val_data.select(numeric_features).to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_full = np.vstack([X_train_scaled, X_val_scaled])
    y_train_full = np.concatenate([y_train, y_val])

    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        n_estimators=n_estimators,
        random_state=random_state,
        device=device,
        ignore_pretraining_limits=True,
    )

    model.fit(X_train_full, y_train_full)

    test_clean = data['_test_clean'] if '_test_clean' in data else None
    if test_clean is not None:
        X_test = test_clean.select(numeric_features).to_numpy()
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
    else:
        X_test_scaled = scaler.transform(data['x_test'])
        y_pred = model.predict(X_test_scaled)

    y_test_actual = data.get('_test_tradeable_scores', data['y_test'])
    rmse = np.sqrt(np.mean((y_pred - y_test_actual) ** 2))
    mae = np.mean(np.abs(y_pred - y_test_actual))
    correlation = np.corrcoef(y_pred, y_test_actual)[0, 1] if len(y_pred) > 1 else 0.0

    n_samples = len(X_train)

    return {
        'models': [model],
        'val_rmse': rmse,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_correlation': correlation,
        'n_regimes_trained': 0,
        '_preds': y_pred,
        'universal_val_rmse': rmse,
        'universal_samples': n_samples,
        '_scaler': scaler,
        'extras': {
            'regime_models': {'universal': model},
            'test_predictions': y_pred,
            'test_clean': data.get('_test_clean'),
            'test_tradeable_scores': data.get('_test_tradeable_scores', None),
            'numeric_features': numeric_features,
            'regime_metrics': {'universal': {'samples': n_samples, 'final_val_rmse': rmse}},
            'model_type': 'TabPFN',
            'n_estimators': n_estimators,
        }
    }
