"""Recorded-data preparation diagnostics used by the research project's search variant."""
import hashlib
import pickle

import numpy as np
import polars as pl

from limen import features, indicators, transforms
from limen.scalers.registry import SCALER_REGISTRY
from limen.utils.find_price_lines import find_price_lines
from recorded_inputs import trade_profile

PARAMETERS = {
    'price_lines': {'max_duration_hours': 48, 'min_height_pct': .01, 'include_research_only': False},
    'quantile_price_lines': {'max_duration_hours': 48, 'min_height_pct': .01, 'include_research_only': False},
    'dynamic_target': {'base_min_breakout': .005, 'target_volatility_multiplier': 1.},
    'dynamic_stop_loss': {'base_stop_loss': .005, 'stop_volatility_multiplier': 1.},
    'dollar_bar_crash_reversal': {'momentum_threshold_bps': 50., 'flow_z_threshold': 1., 'hold_minutes': 60},
    'lag_column': {'col': 'close', 'lag': 1},
    'lag_columns': {'cols': ['close', 'volume'], 'lag': 1},
    'lag_range': {'col': 'close', 'start': 1, 'end': 3},
    'lag_range_cols': {'cols': ['close', 'volume'], 'start': 1, 'end': 3},
    'rolling_zscore': {'column': 'close', 'window': 12},
    'find_min_d': {'col': 'close'},
    'fractional_diff': {'d': .4, 'cols': ['close'], 'threshold': .05},
}


# region helper_context
def prepared_context(raw):
    returns = raw['close'].pct_change()
    frame = raw.with_columns(
        returns.alias('returns'), returns.rolling_std(12).alias('returns_volatility_12'),
        returns.rolling_std(12).alias('rolling_volatility'), returns.rolling_std(60).alias('vol_60h'))
    frame = features.volatility_measure(features.atr_percent_sma(frame))
    training = frame['volatility_measure'].head(864).drop_nulls()
    low, high = training.quantile(.25), training.quantile(.75)
    if low is None or high is None:
        raise ValueError('The recorded training population has no volatility quantiles')
    frame = frame.with_columns(
        pl.when(pl.col('volatility_measure') < low).then(pl.lit('low'))
        .when(pl.col('volatility_measure') > high).then(pl.lit('high'))
        .otherwise(pl.lit('normal')).alias('volatility_regime'),
        pl.Series('vol_percentile', np.searchsorted(np.sort(training.to_numpy()),
                  frame['volatility_measure'].to_numpy(), side='right') / len(training) * 100))
    frame = features.dynamic_target(features.regime_multiplier(frame), .005, 1.)
    frame = features.entry_score_microstructure(features.momentum_confirmation(frame))
    frame = features.bulk_volume_classification(frame)
    forward = raw['close'].shift(-12) / raw['close'] - 1
    return frame.with_columns(
        forward.alias('breakout_long'), (-forward).alias('breakout_short'), forward.alias('breakout_pct'),
        *[(pl.col('volatility_regime') == name).cast(pl.Int32).alias('regime_' + name)
          for name in ('low', 'normal', 'high')])
# endregion helper_context


def retained_frame(output, package, name, result, original):
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    if not isinstance(result, pl.DataFrame):
        value = float(result)
        if not np.isfinite(value):
            raise ValueError('A scalar preparation diagnostic is not finite: ' + name)
        return {'value': value, 'kind': 'fitted diagnostic'}
    columns = [column for column in result.columns if column not in original.columns]
    idempotent = {
        'momentum_confirmation': ['momentum_1', 'momentum_3', 'momentum_score'],
        'entry_score_microstructure': ['entry_score'],
        'bulk_volume_classification': ['bvc_buy_volume', 'bvc_sell_volume'],
    }
    if not columns and name in idempotent:
        columns = idempotent[name]
    if not columns and name in result.columns:
        columns = [name]
    if not columns:
        raise ValueError('The helper has no retained output columns: ' + name)
    path = output / (package + '-' + name + '.parquet')
    result.select('datetime', *columns).write_parquet(path)
    return {'rows': result.height, 'columns': columns, 'path': path.name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


# region regime_labels
def regime_labels(raw):
    from limen.features.classify_market_regime import RegimeConfig
    bars = raw.with_columns(pl.lit('recorded-spot').alias('series_id'))
    return features.classify_market_regime(bars, config=RegimeConfig(version='baseline-v1'), include_audit=True)
# endregion regime_labels


# region helper_diagnostics
def preparation_helpers(raw, data_dir, output):
    output.mkdir(parents=True)
    result = {'indicators': {}, 'features': {}, 'scalers': {}, 'transforms': {}}
    for name in indicators.__all__:
        source = indicators.bollinger_bands(raw) if name == 'bollinger_position' else raw
        value = getattr(indicators, name)(source.clone())
        result['indicators'][name] = retained_frame(output, 'indicator', name, value, source)
    context = prepared_context(raw)
    long_lines, short_lines = find_price_lines(raw['close'].to_numpy(), 48, .01)
    quantiles = [[line for line in lines if abs(line['height_pct']) >= np.quantile(
                  [abs(item['height_pct']) for item in lines], .75)] if lines else []
                 for lines in (long_lines, short_lines)]
    line_arguments = {
        'active_lines': {'long_lines': long_lines, 'short_lines': short_lines},
        'hours_since_big_move': {'long_lines': long_lines, 'short_lines': short_lines, 'lookback_hours': 48},
        'active_quantile_count': {'long_lines_q': quantiles[0], 'short_lines_q': quantiles[1]},
        'hours_since_quantile_line': {'long_lines_q': quantiles[0], 'short_lines_q': quantiles[1], 'lookback_hours': 48},
        'quantile_line_density': {'long_lines_q': quantiles[0], 'short_lines_q': quantiles[1], 'lookback_hours': 48},
    }
    for name in features.__all__:
        if name == 'conserved_flux_renormalization':
            profile = trade_profile(data_dir, output)
            result['features'][name] = {'rows': len(profile['bars']), 'path': 'trade-profile.parquet',
                                        'sha256': hashlib.sha256((output / 'trade-profile.parquet').read_bytes()).hexdigest(),
                                        'columns': list(profile['bars'][0]), 'kind': 'recorded-trade diagnostic'}
        elif name == 'classify_market_regime':
            value = regime_labels(context)
            result['features'][name] = retained_frame(output, 'feature', name, value, context)
        else:
            params = PARAMETERS.get(name, {}) | line_arguments.get(name, {})
            value = getattr(features, name)(context.clone(), **params)
            result['features'][name] = retained_frame(output, 'feature', name, value, context)
    training = raw.head(864).select('close', 'volume')
    later = raw[864:].select('close', 'volume')
    for name, constructor in SCALER_REGISTRY.items():
        fitted = constructor(training)
        before = pickle.dumps(vars(fitted))
        scaled = fitted.transform(later)
        if pickle.dumps(vars(fitted)) != before:
            raise ValueError('Transform changed fitted scaler state: ' + name)
        scaled.write_parquet(output / ('scaler-' + name + '.parquet'))
        result['scalers'][name] = {'rows': scaled.height, 'columns': scaled.columns,
                                  'fitted_rows': training.height, 'fitted_state_unchanged': True}
    for name in transforms.__all__:
        source = raw.head(864).select('datetime', 'close', 'volume')
        params = {'shift': 1, 'column': 'close'} if name == 'shift_column_transform' else {}
        transformed = getattr(transforms, name)(source, **params)
        transformed.write_parquet(output / ('transform-' + name + '.parquet'))
        result['transforms'][name] = {'rows': transformed.height, 'columns': transformed.columns}
    result['counts'] = {name: len(cases) for name, cases in result.items()}
    return result
# endregion helper_diagnostics
