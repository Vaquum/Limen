import hashlib
import json
from zipfile import ZipFile

import polars as pl

from limen import HistoricalData
from limen.features.conserved_flux_renormalization import conserved_flux_renormalization

FUNDING = 'btcusdt_funding_20240101_20240108.json'
TRADES = 'binance_trades_btcusdt_2025-05-23_90s.zip'
HASHES = {
    FUNDING: '4352c58c5467ba48b5d13039052085e3bd59c089d6e1455bf99f4063279c3df2',
    TRADES: 'ecc0d2c54aad5c69bba510fe0453d91b288f23579e568d665c06ecfae49757b5',
}


def checked_path(data_dir, name):
    path = data_dir / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
        raise ValueError('Recorded input digest differs: ' + name)
    return path


# region funding
def funding_history(data_dir):
    records = json.loads(checked_path(data_dir, FUNDING).read_text())['records']
    slot_ms = 8 * 3600 * 1000
    slots = [row['fundingTime'] // slot_ms * slot_ms for row in records]
    return pl.DataFrame({
        'event_id': [f"BTCUSDT:{row['fundingTime']}" for row in records],
        'settlement_at': [row['fundingTime'] * 1_000_000 for row in records],
        'settlement_slot_ns': [slot * 1_000_000 for slot in slots],
        'rate_decimal': [float(row['fundingRate']) for row in records],
        'valuation_price': [float(row['markPrice']) for row in records],
        'schedule_start': [min(slots) * 1_000_000] * len(records),
        'schedule_end': [(max(slots) + slot_ms) * 1_000_000] * len(records),
        'settlement_interval_seconds': [28800.0] * len(records),
        'settlement_phase_utc_seconds': [0.0] * len(records),
    })
# endregion funding


# region trade_profile
def trade_profile(data_dir, output):
    historical = HistoricalData()
    historical.get_any_file(str(checked_path(data_dir, TRADES)), has_header=False,
                            columns=['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker', '_null'])
    trades = historical.data
    if not isinstance(trades, pl.DataFrame):
        raise ValueError('The recorded trade source did not produce a dataframe')
    profile = conserved_flux_renormalization(trades, kline_interval='30s', base_window_s=5, levels=3)
    profile.write_parquet(output / 'trade-profile.parquet')
    rows = [{**row, 'datetime': row['datetime'].isoformat()} for row in profile.to_dicts()]
    return {'trade_rows': trades.height, 'bars': rows, 'input_sha256': HASHES[TRADES], 'population': '90 recorded seconds for flow diagnostics; no compiled training population is asserted'}
# endregion trade_profile


# region trade_points
def trade_points(data_dir):
    with ZipFile(checked_path(data_dir, TRADES)) as archive, archive.open(archive.namelist()[0]) as stream:
        recorded = pl.read_csv(stream, has_header=False)
    points = recorded.select(
        pl.from_epoch('column_5', time_unit='us').dt.replace_time_zone('UTC').alias('datetime'),
        *[pl.col('column_2').alias(name) for name in ('open', 'high', 'low', 'close', 'mean')],
        pl.col('column_3').alias('volume'), pl.lit(1).alias('no_of_trades'),
        pl.col('column_4').alias('liquidity_sum'), pl.col('column_6').cast(pl.Float64).alias('maker_ratio'),
        (pl.col('column_3') * pl.col('column_6')).alias('maker_volume'),
        (pl.col('column_4') * pl.col('column_6')).alias('maker_liquidity'),
    ).unique(subset='datetime', keep='first', maintain_order=True)
    return points.with_columns([pl.col('datetime').dt.epoch('ns').alias(name)
        for name in ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns')])
# endregion trade_points


# region target_label_inputs
def target_label_inputs(data_dir):
    """Derive software labels from recorded closes, independently within each declared split."""
    from datetime import datetime, timezone
    source = data_dir / 'spot_1h_20240101_20241231.parquet'
    if hashlib.sha256(source.read_bytes()).hexdigest() != 'a634bbd4f3692fac7bc88512da588e7772f4bd1b749567606d5314e54d0adf72':
        raise ValueError('Recorded hourly source changed before label derivation')
    raw = pl.read_parquet(source).select('datetime', 'open', 'high', 'low', 'close', 'volume')
    outputs = {name: [] for name in ('exit', 'risk', 'identity')}
    for start, end in [('2024-01-01', '2024-02-15'), ('2024-02-15', '2024-03-15'), ('2024-03-15', '2024-04-15')]:
        left, right = (datetime.fromisoformat(value).replace(tzinfo=timezone.utc) for value in (start, end))
        frame = raw.filter(pl.col('datetime').is_between(left, right, closed='left'))
        close, following, second = pl.col('close'), pl.col('close').shift(-1), pl.col('close').shift(-2)
        outputs['exit'].append(frame.with_columns(
            pl.when(following.is_not_null()).then(pl.lit('timeout')).otherwise(None).alias('exit_reason'),
            (following / close - 1 - .0002).alias('exit_net_return')).filter(pl.col('exit_reason').is_not_null()))
        outputs['risk'].append(frame.with_columns(
            pl.when(second.is_not_null()).then(pl.max_horizontal(close, following, second) / close - 1).otherwise(None).alias('capturable_breakout'),
            pl.when(second.is_not_null()).then(pl.min_horizontal(close, following, second) / close - 1).otherwise(None).alias('max_drawdown')))
        outputs['identity'].append(frame.with_columns((following > close).cast(pl.UInt8).alias('label')))
    return {name: pl.concat(parts) for name, parts in outputs.items()}
# endregion target_label_inputs
