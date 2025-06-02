import polars as pl


def scale_data_dict(data_dict: dict) -> dict:
    
    feature_cols = list(data_dict["x_train"].columns)
    mean_exprs = [pl.col(c).mean().alias(f"{c}__mean") for c in feature_cols]
    std_exprs = [pl.col(c).std().alias(f"{c}__std") for c in feature_cols]
    stats_df = data_dict["x_train"].select(mean_exprs + std_exprs)
    stats_row = stats_df.row(0)
    stats_dict = dict(zip(stats_df.columns, stats_row))
    train_means = {c: stats_dict[f"{c}__mean"] for c in feature_cols}
    train_stds = {c: stats_dict[f"{c}__std"] for c in feature_cols}

    def _scale_df(df: pl.DataFrame) -> pl.DataFrame:
        
        return df.with_columns([
            ((pl.col(c) - train_means[c]) / train_stds[c]).alias(c)
            for c in feature_cols
        ])

    data_dict["x_train"] = _scale_df(data_dict["x_train"])
    data_dict["x_val"] = _scale_df(data_dict["x_val"])
    data_dict["x_test"] = _scale_df(data_dict["x_test"])
    
    return data_dict
