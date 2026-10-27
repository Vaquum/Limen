import polars as pl

__all__: list[str] = []


def aggregate_bars(data: pl.DataFrame, base_interval: float) -> pl.DataFrame:
    return (
        data
        .group_by('bar_group', maintain_order=True)
        .agg([
            pl.col('datetime').first().alias('datetime'),
            pl.col('open').first().alias('open'),
            pl.col('high').max().alias('high'),
            pl.col('low').min().alias('low'),
            pl.col('close').last().alias('close'),
            pl.col('volume').sum().alias('volume'),
            pl.col('no_of_trades').sum().alias('no_of_trades'),
            pl.col('liquidity_sum').sum().alias('liquidity_sum'),
            pl.col('maker_volume').sum().alias('maker_volume'),
            pl.col('maker_liquidity').sum().alias('maker_liquidity'),
            ((pl.col('maker_ratio') * pl.col('no_of_trades')).sum()
             / pl.col('no_of_trades').sum()).alias('maker_ratio'),
            ((pl.col('mean') * pl.col('no_of_trades')).sum()
             / pl.col('no_of_trades').sum()).alias('mean'),
            pl.len().alias('bar_count'),
            pl.lit(base_interval).alias('base_interval')
        ])
        .sort('bar_group')
        .select([
            'datetime', 'open', 'high', 'low', 'close', 'volume',
            'no_of_trades', 'liquidity_sum', 'maker_ratio', 'maker_volume',
            'maker_liquidity', 'mean', 'bar_count', 'base_interval'
        ])
    )
