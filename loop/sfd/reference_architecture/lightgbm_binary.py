import lightgbm as lgb

from loop.metrics.binary_metrics import binary_metrics


def lightgbm_binary(data: dict,
               objective: str = 'binary',
               metric: str = 'auc',
               learning_rate: float = 0.05,
               num_leaves: int = 31,
               max_depth: int = -1,
               min_data_in_leaf: int = 20,
               feature_fraction: float = 0.9,
               bagging_fraction: float = 0.8,
               bagging_freq: int = 5,
               lambda_l1: float = 0.0,
               lambda_l2: float = 0.0,
               feature_pre_filter: str = 'false',
               num_boost_round: int = 4000,
               early_stopping_rounds: int = 200,
               pred_threshold: float = 0.5) -> dict:

    '''
    Compute LightGBM binary classification predictions and evaluation metrics.

    Args:
        data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test
        objective (str): LightGBM objective function
        metric (str): Evaluation metric
        learning_rate (float): Learning rate
        num_leaves (int): Maximum number of leaves in one tree
        max_depth (int): Maximum tree depth (-1 means no limit)
        min_data_in_leaf (int): Minimum number of data in one leaf
        feature_fraction (float): Feature sampling ratio
        bagging_fraction (float): Data sampling ratio
        bagging_freq (int): Frequency for bagging
        lambda_l1 (float): L1 regularization
        lambda_l2 (float): L2 regularization
        feature_pre_filter (str): Whether to pre-filter features
        num_boost_round (int): Number of boosting iterations
        early_stopping_rounds (int): Early stopping rounds
        pred_threshold (float): Threshold for binary predictions

    Returns:
        dict: Results with binary metrics and predictions
    '''

    dtrain = lgb.Dataset(data['x_train'], label=data['y_train'].to_numpy())
    dval = lgb.Dataset(data['x_val'], label=data['y_val'].to_numpy(), reference=dtrain)

    pos_cnt = data['y_train'].sum()
    neg_cnt = len(data['y_train']) - pos_cnt
    scale_pos_weight = round((neg_cnt / pos_cnt) if pos_cnt > 0 else 1.0, 4)

    params = {
        'objective': objective,
        'metric': metric,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'min_data_in_leaf': min_data_in_leaf,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'feature_pre_filter': feature_pre_filter,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
    }

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    pred_prob = model.predict(data['x_test'], num_iteration=model.best_iteration)
    pred_bin = (pred_prob >= pred_threshold).astype(int)

    round_results = binary_metrics(data, pred_bin, pred_prob)
    round_results['_preds'] = pred_prob

    return round_results
