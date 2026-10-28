from limen.sfd.reference_architecture.tabpfn_binary import TABPFN_MODEL_PATH

__all__ = ['model_params']


def model_params() -> dict[str, list[object]]:
    return {
        'n_ensemble_configurations': [4, 8],
        'device': ['cpu'],
        'categorical_features_indices': [None],
        'softmax_temperature': [0.7, 0.9, 1.1],
        'balance_probabilities': [False, True],
        'average_before_softmax': [False, True],
        'model_path': [TABPFN_MODEL_PATH],
        'ignore_pretraining_limits': [True],
        'inference_precision': ['auto'],
        'fit_mode': ['fit_preprocessors'],
        'memory_saving_mode': ['auto'],
        'random_state': [0],
        'n_preprocessing_jobs': [1],
        'inference_config': [None],
        'differentiable_input': [False],
        'eval_metric': [None],
        'tuning_config': [None],
    }
