def params():
    return {
        "lookback": [30, 60, 120],
        "hidden_dim": [64, 128],
        "n_layers": [2, 4],
        "n_heads": [2, 4],
        "dropout": [0.1, 0.2],
        "learning_rate": [1e-3, 5e-4],
        "batch_size": [32, 64],
        "epochs": [5]
    }

def prep(data, round_params=None):
    """
    Prepares data for training.
    Should return dict with x_train, y_train, x_val, y_val, x_test, y_test.
    """
    # TODO: implement transformer-specific prep
    return {}

def model(data, round_params):
    """
    Builds + trains a binary transformer classifier.
    Should return dict with metrics, preds, and models.
    """
    # TODO: implement transformer training
    return {"metrics": {}, "models": []}
