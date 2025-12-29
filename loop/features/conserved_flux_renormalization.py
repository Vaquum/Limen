import polars as pl
import numpy as np


def conserved_flux_renormalization(trades_df: pl.DataFrame,
                                   *,
                                   kline_interval: str = "1h",
                                   base_window_s: int = 60,
                                   levels: int = 6) -> pl.DataFrame:
    
    '''
    Compute multi-scale, conserved-flux features and their deviation scores
    for each k-line—turning raw trade ticks into a six-value fingerprint that
    flags hours where the dollar flow or trade-size entropy breaks
    scale-invariant behaviour.

    Args:
        trades_df (pl.DataFrame): The trades DataFrame.
        kline_interval (str): The kline interval.
        base_window_s (int): The base window size.
        levels (int): The number of levels to compute.

    Returns:
        pl.DataFrame: A klines DataFrame with the CFR features
    '''

    trades = (trades_df
              .with_columns((pl.col('price') * pl.col('quantity')).alias('value'))
              .sort('datetime'))

    kl = (trades
          .group_by_dynamic('datetime', every=kline_interval, closed='left')
          .agg(
              pl.col('price').first().alias('open'),
              pl.col('price').max().alias('high'),
              pl.col('price').min().alias('low'),
              pl.col('price').last().alias('close'),
              pl.col('quantity').sum().alias('volume'),
              pl.col('value').sum().alias('value_sum'),
          )
          .with_columns(
              (pl.col('value_sum') / pl.col('volume')).alias('vwap'),
              pl.col('datetime').cast(pl.Datetime('ms')),
          ))

    mult = {'s': 1_000, 'm': 60_000, 'h': 3_600_000, 'd': 86_400_000}
    bucket_ms = int(kline_interval[:-1]) * mult[kline_interval[-1]]

    feature_schema = {
        'datetime':            pl.Datetime('ms'),
        'flux_rel_std_mean':   pl.Float64,
        'flux_rel_std_var':    pl.Float64,
        'entropy_mean':        pl.Float64,
        'entropy_var':         pl.Float64,
        'Δflux_rms':           pl.Float64,
        'Δentropy_rms':        pl.Float64,
    }

    def _bar_features(g: pl.DataFrame) -> pl.DataFrame:
        
        flux_vec, ent_vec = _per_scale_stats(
            g, base_window_s=base_window_s, levels=levels
        )
        
        if len(flux_vec) == 0:
            return pl.DataFrame({k: [None] for k in feature_schema})  # empty bar

        flux_mean, flux_var = float(np.nanmean(flux_vec)), float(np.nanvar(flux_vec))
        ent_mean,  ent_var  = float(np.nanmean(ent_vec)),  float(np.nanvar(ent_vec))

        ideal_f = np.full(len(flux_vec), flux_mean)
        ideal_e = np.array([ent_vec[0] - i for i in range(len(ent_vec))])

        rms_flux = float(np.sqrt(((flux_vec - ideal_f) ** 2).mean()))
        rms_ent  = float(np.sqrt(((ent_vec  - ideal_e) ** 2).mean()))

        ts_ms = int(g['datetime'].min().timestamp() * 1_000)
        bucket = (ts_ms // bucket_ms) * bucket_ms

        return pl.DataFrame(
            {
                'datetime': pl.Series([bucket]).cast(pl.Datetime('ms')),
                'flux_rel_std_mean': [flux_mean],
                'flux_rel_std_var':  [flux_var],
                'entropy_mean':      [ent_mean],
                'entropy_var':       [ent_var],
                'Δflux_rms':         [rms_flux],
                'Δentropy_rms':      [rms_ent],
            }
        )

    feats = (trades
             .group_by_dynamic('datetime', every=kline_interval, closed='left')
             .map_groups(_bar_features, schema=feature_schema))

    return (kl
            .join(feats, on='datetime', how='left')
            .sort('datetime'))


def _per_scale_stats(trades: pl.DataFrame, *, base_window_s=60, levels=6):
    
    rel_std, ent = [], []
    
    for k in range(levels):
        
        span = f"{base_window_s * 2**k}s"
        bins = trades.group_by_dynamic('datetime', every=span, closed='left').agg(
            pl.col('value').sum().alias('flux'),
            (
                -((pl.col('value') / pl.col('value').sum())
                  * (pl.col('value') / pl.col('value').sum()).log(base=2)
                 ).sum()
            ).alias('entropy')
        )
        if bins.height < 2:
            continue
        rel_std.append(float(bins['flux'].std(ddof=0) / bins['flux'].mean()))
        ent.append(float(bins['entropy'].mean()))
    
    return np.array(rel_std), np.array(ent)
