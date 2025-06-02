import polars as pl

def ppo(data: pl.DataFrame,
        price_col: str = "close",
        span_short: int = 12,
        span_long: int = 26,
        ppo_name: str = "ppo") -> pl.DataFrame:
    
    '''
    Compute the Percentage Price Oscillator (PPO) in pure Polars and append it.
    
    PPO = 100 * (EMA_short - EMA_long) / EMA_long
    '''
    alpha_short = 2.0 / (span_short + 1)
    alpha_long  = 2.0 / (span_long  + 1)
    price = pl.col(price_col)

    return (
        data
        .with_columns([
            price.ewm_mean(alpha=alpha_short, adjust=False).alias(f"EMA_{span_short}"),
            price.ewm_mean(alpha=alpha_long,  adjust=False).alias(f"EMA_{span_long}"),
        ])
        .with_columns([
            (
                (pl.col(f"EMA_{span_short}") - pl.col(f"EMA_{span_long}"))
                / pl.col(f"EMA_{span_long}") * 100
            ).alias(ppo_name)
        ]).drop([f"EMA_{span_short}", f"EMA_{span_long}"])
    )
