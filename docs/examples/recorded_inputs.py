import hashlib
import json

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
