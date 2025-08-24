import polars as pl
from typing import List


def get_numeric_feature_columns(df: pl.DataFrame, exclude_cols: List[str]) -> List[str]:
    
    '''
    Compute numeric feature column names from DataFrame excluding specified columns.
    
    Args:
        df (pl.DataFrame): Dataset with mixed column types
        exclude_cols (List[str]): List of column names to exclude
        
    Returns:
        List[str]: List of column names with numeric data types
    '''
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_features = [col for col in feature_cols 
                       if df.schema[col] in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]]
    return numeric_features