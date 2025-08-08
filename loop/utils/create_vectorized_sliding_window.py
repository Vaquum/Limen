from typing import List, Tuple
import polars as pl
import numpy as np


def create_vectorized_sliding_window(df: pl.DataFrame,
                                     feature_cols: List[str],
                                     target_col: str,
                                     window_size: int,
                                     prediction_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    
    '''
    Compute vectorized sliding window arrays for time series modeling.
    
    Args:
        df (pl.DataFrame): Input DataFrame with time series data
        feature_cols (List[str]): Column names to use as features
        target_col (str): Column name to use as target variable
        window_size (int): Number of time steps in each input window
        prediction_horizon (int): Number of time steps to predict ahead
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix (M, window_size * F) and target matrix (M, prediction_horizon)
    '''

    X_mat = df.select(feature_cols).to_numpy().astype(np.float32)
    y_vec = df[target_col].to_numpy().astype(np.float32)
    N, F  = X_mat.shape
    M = N - window_size - prediction_horizon + 1

    X = np.lib.stride_tricks.sliding_window_view(X_mat,
                                                 window_shape=window_size,
                                                 axis=0,
                                                 writeable=False)[:M].reshape(M, -1)

    Y = np.lib.stride_tricks.sliding_window_view(y_vec,
                                                 window_shape=prediction_horizon,
                                                 axis=0,
                                                 writeable=False)[window_size : window_size + M]

    if Y.ndim > 1:
        Y = (Y.max(axis=1) > 0).astype(int)

    return X, Y
