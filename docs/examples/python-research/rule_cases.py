"""Causal Boolean rules on one declared recorded assessment partition."""
import polars as pl


# region rule_signals
def evaluate_rules(raw):
    bars = raw.filter(pl.col('datetime').is_between(
        pl.datetime(2024, 1, 18, time_zone='UTC'), pl.datetime(2024, 1, 22, time_zone='UTC'), closed='left'))
    up = pl.col('close') > pl.col('open')
    cross = up & (pl.col('close').shift(1) <= pl.col('open').shift(1))
    persistent = (up.cast(pl.Int8).rolling_sum(2, min_samples=2) == 2).fill_null(False)
    recent = (cross.fill_null(False).cast(pl.Int8).rolling_sum(3, min_samples=1) >= 1)
    rising = (pl.col('close') > pl.col('close').shift(3)).fill_null(False)
    rows = []
    for floor in (40000., 41000.):
        signal = bars.select(((pl.col('close') >= floor) & (persistent | recent) & rising &
                              ~(pl.col('close') < pl.col('open'))).cast(pl.Int8).alias('signal'))['signal']
        rows.append({'price_floor': floor, 'signals': signal.to_list(), 'positive_decisions': int(signal.sum())})
    return {'rows': rows, 'assessment_rows': bars.height, 'history_resets_at_split': True}
# endregion rule_signals
