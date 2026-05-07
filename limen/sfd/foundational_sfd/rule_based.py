from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.experiment import RuleBasedManifest
from limen.indicators import ema
from limen.indicators import wilder_rsi
from limen.sfd.reference_architecture.rule_based import rule_based


_CONDITIONS = [
    {
        'id': 'rsi_oversold',
        'type': 'threshold',
        'column': 'wilder_rsi_{rsi_period}',
        'operator': '<',
        'value': '{rsi_threshold}',
    },
    {
        'id': 'above_ema',
        'type': 'relative',
        'column': 'close',
        'operator': '>',
        'other_column': 'ema_{ema_period}',
    },
    {
        'id': 'entry',
        'operator': 'and',
        'operands': ['rsi_oversold', 'above_ema'],
    },
]


def params() -> dict:

    return {
        'rsi_period': [7, 14, 21],
        'rsi_threshold': [20, 25, 30, 35],
        'ema_period': [50, 100, 200],
        'sharpe_std_threshold': [0.5],
        'sharpe_degradation_threshold': [0.3],
    }


def manifest() -> Manifest:

    return (RuleBasedManifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
        )
        .set_test_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 7200, 'row_count_limit': 5000},
        )
        .set_split_config(8, 1, 2)
        .add_indicator(wilder_rsi, period='rsi_period', group='momentum')
        .add_indicator(ema, period='ema_period', group='trend')
        .with_strategy(_CONDITIONS, entry='entry')
        .with_reference_architecture(rule_based)
    )
