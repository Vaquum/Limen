"""Read the same recorded observations through each admitted local file format."""
import hashlib
from limen import HistoricalData


# region source_formats
def read_source_formats(data_dir):
    report = {}
    for suffix in ('csv', 'zip', 'arrow'):
        path = data_dir / ('recorded-hourly.' + suffix)
        historical = HistoricalData()
        frame = (historical.get_arrow_file(str(path)) if suffix == 'arrow' else historical.get_any_file(str(path)))
        latest = (historical.get_arrow_file(str(path), row_count_limit=12) if suffix == 'arrow' else historical.get_any_file(str(path), row_count_limit=12))
        if not latest.equals(frame.tail(12)) or historical.data is not latest:
            raise ValueError('Local source changed row-limit or retained-frame semantics')
        report[suffix] = {'rows': frame.height, 'columns': frame.columns, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'timestamps_ns': frame['datetime'].dt.epoch('ns').to_list(),
                         'values': frame.select('open', 'high', 'low', 'close', 'volume').to_dicts()}
    return report
# endregion source_formats
