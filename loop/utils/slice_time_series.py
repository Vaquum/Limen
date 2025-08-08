from sklearn.cluster import KMeans
import numpy as np
import polars as pl
from typing import List


def slice_time_series(df: pl.DataFrame, 
                      target_col: str,
                      k: int) -> List[pl.DataFrame]:
    
    '''
    Compute time series segments using K-means clustering on target column.
    
    Args:
        df (pl.DataFrame): Input DataFrame with time series data
        target_col (str): Column name to use for clustering-based slicing
        k (int): Number of slices to create
        
    Returns:
        List[pl.DataFrame]: List of DataFrame slices based on clustering
    '''
    
    mags = np.log1p(df[target_col].to_numpy()).reshape(-1, 1)
    
    labels = KMeans(n_clusters=k, n_init="auto",
                    random_state=0).fit_predict(mags)
    
    df = df.with_columns(pl.Series("band", labels))
    slices = [df.filter(pl.col("band") == i) for i in range(k)]

    return slices
