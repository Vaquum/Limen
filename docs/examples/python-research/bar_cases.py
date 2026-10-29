"""Adaptive formation on a declared unique-clock view of recorded trades."""
import hashlib

from limen.data.utils import compute_data_bars
from recorded_inputs import trade_points


# region adaptive_formation
def adaptive_trade_bars(data_dir, output):
    output.mkdir(parents=True)
    points = trade_points(data_dir)
    rows = []
    for kind in ('imbalance', 'run'):
        formed = compute_data_bars(points, bar_type=kind, expected_num_ticks=20, ema_span=10)
        path = output / (kind + '.parquet')
        formed.write_parquet(path)
        consumed = int(formed['bar_count'].sum())
        rows.append({'kind': kind, 'rows': formed.height, 'consumed_trade_rows': consumed,
                     'omitted_trade_rows': points.height - consumed,
                     'volume': float(formed['volume'].sum()),
                     'last_available_at_ns': int(points['available_at_ns'][consumed - 1]),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'trade_rows': points.height, 'expected_num_ticks': 20, 'ema_span': 10,
            'population': 'first recorded trade per unique microsecond timestamp', 'formed': rows}
# endregion adaptive_formation
