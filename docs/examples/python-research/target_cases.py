"""Label contracts on separate chronological populations of recorded hourly bars."""
import hashlib
import pickle

import polars as pl
from limen import targets
from recorded_inputs import target_label_inputs

# Each choice is a separate labeling question, rather than another feature column.
CASES = {
    'EmaBreakoutTarget': ({}, {'ema_span': 12, 'breakout_delta': .005, 'breakout_horizon': 3}),
    'ForwardBreakoutTarget': ({}, {'forward_periods': 3, 'threshold': .005, 'shift': -1}),
    'ForwardVolNormalizedReturnTarget': ({'periods': 3, 'halflife': 12, 'min_periods': 24}, {}),
    'IdentityTarget': ({}, {}),
    'NextBarDownTarget': ({}, {}),
    'NextBarUpTarget': ({}, {}),
    'NextReturnTarget': ({}, {'periods': 3, 'scale': 100.}),
    'QuantileBinaryTarget': ({'source_column': 'recorded_return', 'quantile': .5}, {'shift': -1}),
    'RandomBinaryTarget': ({'random_state': 42}, {}),
    'ThresholdBinaryTarget': ({'source_column': 'recorded_return', 'threshold': 0.}, {'shift': -1}),
    'TradelineLongBinaryTarget': ({'max_duration_hours': 48, 'min_height_pct': .005, 'long_threshold_percentile': 50}, {'lookahead_hours': 12, 'confirmation_hours': 6}),
    'TripleBarrierTarget': ({'upper_multiple': 1.5, 'lower_multiple': 1., 'max_horizon': 3, 'span': 8, 'min_periods': 8}, {}),
    'VolNormalizedReturnTarget': ({'halflife': 12, 'min_periods': 24}, {}),
}


# region target_diagnostics
def label_diagnostics(raw, data_dir, output):
    output.mkdir(parents=True)
    partitions = [raw[:864], raw[864:1152], raw[1152:1440]]
    report = {}
    for name, (fit, transform) in CASES.items():
        label = 'close' if name == 'IdentityTarget' else 'label'
        prepared = [part.with_columns(pl.col('close').pct_change().alias('recorded_return')) for part in partitions]
        constructor = getattr(targets, name)
        try:
            target = constructor(prepared[0], label, **fit)
        except ValueError as error:
            if name not in ('VolNormalizedReturnTarget', 'ForwardVolNormalizedReturnTarget') or 'Parkinson-to-close volatility ratio' not in str(error):
                raise
            report[name] = {'status': 'refused', 'reason': str(error), 'gate': 'training volatility sanity'}
            continue
        fitted = pickle.dumps(vars(target))
        samples = []
        for index, frame in enumerate(prepared):
            labeled = target.transform(frame, **transform).select('datetime', pl.col(label).alias('label'))
            path = output / (name + '-' + str(index) + '.parquet')
            labeled.write_parquet(path)
            samples.append({'rows': labeled.height, 'null_rows': labeled['label'].null_count(),
                            'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        if name != 'RandomBinaryTarget' and pickle.dumps(vars(target)) != fitted:
            raise ValueError('Label transformation changed fitted target state: ' + name)
        report[name] = {'status': 'labeled', 'partitions': samples, 'fitted_only_on_training': True}
    # These contracts require supplied trade-outcome columns, absent from raw bars.
    for name, required in [('ExitQualityTarget', ['exit_reason', 'exit_net_return']),
                           ('RiskRewardRatioTarget', ['capturable_breakout', 'max_drawdown'])]:
        target = getattr(targets, name)(partitions[0], 'label')
        try:
            target.transform(partitions[1])
        except ValueError as error:
            if not all(column in str(error) for column in required):
                raise
            report[name] = {'status': 'refused', 'reason': str(error), 'required_input_columns': required}
        else:
            raise ValueError('A trade-context target accepted missing outcome columns: ' + name)
    supplied = target_label_inputs(data_dir)
    for name, context in [('ExitQualityTarget', 'exit'), ('RiskRewardRatioTarget', 'risk')]:
        frames = supplied[context]
        constructor = getattr(targets, name)
        target = constructor(frames.filter(pl.col('datetime') < pl.datetime(2024, 2, 15, time_zone='UTC')), 'label')
        labeled = target.transform(frames).select('datetime', 'label')
        path = output / (name + '-context.parquet')
        labeled.write_parquet(path)
        report[name]['supplied_context'] = {'rows': labeled.height, 'path': path.name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source': 'recorded-close software labels; not observed venue exits'}
    return report
# endregion target_diagnostics
