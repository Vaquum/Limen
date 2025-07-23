from sklearn.cluster import KMeans
import numpy as np
import polars as pl
from typing import List


def slice_time_series(df: pl.DataFrame, 
                      target_col: str,
                      k: int) -> List[pl.DataFrame]:
    '''
    Slice time series into segments based on percentile gaps in the target column.
    
    Args:
        df (pl.DataFrame): Input DataFrame
        target_col (str): Column to use for slicing
        k (int): Number of slices
        
    Returns:
        List of DataFrame slices
    '''
    
    mags = np.log1p(df[target_col].to_numpy()).reshape(-1, 1)
    
    labels = KMeans(n_clusters=k, n_init="auto",
                    random_state=0).fit_predict(mags)
    
    df = df.with_columns(pl.Series("band", labels))
    slices = [df.filter(pl.col("band") == i) for i in range(k)]

    return slices
