import numpy as np
import lightgbm as lgb


def lgb_tradeable_regression(data: dict,
                              objective: str = 'regression',
                              metric: str = 'rmse',
                              learning_rate: float = 0.05,
                              num_leaves: int = 31,
                              feature_fraction: float = 0.8,
                              bagging_fraction: float = 0.8,
                              bagging_freq: int = 5,
                              boosting_type: str = 'gbdt',
                              num_iterations: int = 100,
                              force_col_wise: bool = True,
                              num_boost_round: int = 300,
                              early_stopping_rounds: int = 30,
                              weight_target_achieved: int = 20,
                              weight_quick_target: int = 30,
                              weight_high_score_p90: int = 20,
                              weight_high_score_p95: int = 50,
                              weight_high_score_p99: int = 100,
                              weight_profitable_multiplier: float = 1.5,
                              **kwargs) -> dict:

    '''
    Compute LightGBM regression predictions for tradeable score with custom sample weighting.

    Args:
        data (dict): Data dictionary with train/val/test splits and tradeable-specific entries
        objective (str): LightGBM objective function (default: regression)
        metric (str): Evaluation metric (default: rmse)
        learning_rate (float): Learning rate (default: 0.05)
        num_leaves (int): Maximum number of leaves in one tree (default: 31)
        feature_fraction (float): Feature sampling ratio (default: 0.8)
        bagging_fraction (float): Data sampling ratio (default: 0.8)
        bagging_freq (int): Frequency for bagging (default: 5)
        boosting_type (str): Type of boosting algorithm (default: gbdt)
        num_iterations (int): Number of boosting iterations (default: 100)
        force_col_wise (bool): Force column-wise histogram building (default: True)
        num_boost_round (int): Number of boosting rounds for training (default: 300)
        early_stopping_rounds (int): Early stopping rounds (default: 30)
        weight_target_achieved (int): Weight for samples that achieved target (default: 20)
        weight_quick_target (int): Weight for samples that achieved target quickly (default: 30)
        weight_high_score_p90 (int): Weight for high score samples >p90 (default: 20)
        weight_high_score_p95 (int): Weight for high score samples >p95 (default: 50)
        weight_high_score_p99 (int): Weight for high score samples >p99 (default: 100)
        weight_profitable_multiplier (float): Multiplier for profitable samples (default: 1.5)
        **kwargs: Additional parameters (ignored)

    Returns:
        dict: Results with models, metrics, predictions and tradeable-specific extras
    '''

    train_data = data['_train_clean']
    arrays = train_data.select(['achieves_dynamic_target', 'exit_bars', 'tradeable_score', 'exit_net_return']).to_numpy()
    achieved, exit_bars, y_values, net_returns = arrays[:, 0].astype(bool), arrays[:, 1], arrays[:, 2], arrays[:, 3]
    profitable = net_returns > 0.001

    weights = np.ones(len(train_data))
    p90, p95, p99 = np.percentile(y_values, [90, 95, 99])

    weights[achieved] = weight_target_achieved
    weights[achieved & (exit_bars <= 6)] = weight_quick_target
    weights[y_values > p90] = weight_high_score_p90
    weights[y_values > p95] = weight_high_score_p95
    weights[y_values > p99] = weight_high_score_p99
    weights[profitable] *= weight_profitable_multiplier

    numeric_features = data['_numeric_features']
    X_train = train_data.select(numeric_features).to_numpy()
    X_val = data['_val_clean'].select(numeric_features).to_numpy()
    y_val = data['_val_clean'].select('tradeable_score').to_numpy().flatten()

    dtrain = lgb.Dataset(X_train, label=y_values, weight=weights)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        'objective': objective,
        'metric': metric,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'boosting_type': boosting_type,
        'num_iterations': num_iterations,
        'force_col_wise': force_col_wise,
        'verbose': -1,
    }

    evals_result = {}
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                   lgb.record_evaluation(evals_result)]
    )

    test_clean = data['_test_clean'] if '_test_clean' in data else None
    y_pred = model.predict(test_clean.select(numeric_features).to_numpy()) if test_clean is not None else model.predict(data['x_test'])

    val_rmse = float(evals_result['val']['rmse'][-1])
    n_samples = len(data['x_train'])

    return {
        'models': [model], 'val_rmse': val_rmse, 'n_regimes_trained': 0, '_preds': y_pred,
        'universal_val_rmse': val_rmse, 'universal_samples': n_samples,
        'extras': {
            'regime_models': {'universal': model}, 'test_predictions': y_pred,
            'test_clean': data['_test_clean'], 'test_tradeable_scores': data.get('_test_tradeable_scores', None),
            'numeric_features': numeric_features,
            'regime_metrics': {'universal': {'samples': n_samples, 'final_val_rmse': val_rmse}}
        }
    }
