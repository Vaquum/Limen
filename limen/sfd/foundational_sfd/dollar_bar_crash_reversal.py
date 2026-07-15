from datetime import date
from typing import Any

from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.experiment import RuleBasedManifest
from limen.features import dollar_bar_crash_reversal
from limen.sfd.reference_architecture.rule_based import rule_based


_CONDITIONS = [
    {
        'id': 'crash_reversal_position',
        'name': 'Crash-reversal position is active',
        'type': 'threshold',
        'column': 'dollar_bar_crash_reversal_position',
        'operator': '>',
        'value': 0,
    },
]


def params() -> dict[str, list[Any]]:
    return {
        'momentum_threshold_bps': [-525.0, -550.0, -575.0, -600.0],
        'flow_z_threshold': [-1.0, -0.5, 0.0, 0.5],
        'hold_minutes': [30, 45, 60, 90, 120],
    }


def manifest() -> Manifest:
    configured = RuleBasedManifest()
    _ = configured.set_data_source(
        method=HistoricalData.get_spot_dollar_klines,
        params={
            'dollar_bar_size': 15_000_000,
            'start_date_limit': '2020-02-01',
            'end_date_limit': '2026-07-10',
        },
    )
    _ = configured.set_required_bar_columns([
        'datetime',
        'open',
        'close',
        'liquidity_sum',
        'maker_liquidity',
    ])
    _ = configured.set_split_dates(
        date(2020, 2, 1),
        date(2024, 1, 1),
        date(2024, 1, 1),
        date(2025, 1, 1),
        date(2025, 1, 1),
        date(2026, 7, 10),
    )
    _ = configured.add_indicator(
        dollar_bar_crash_reversal,
        momentum_threshold_bps='momentum_threshold_bps',
        flow_z_threshold='flow_z_threshold',
        hold_minutes='hold_minutes',
    )
    _ = configured.set_backtest_config(fee_bps=10.0, slip_bps=5.0)

    return (
        configured
        .with_strategy(_CONDITIONS, entry='crash_reversal_position')
        .with_reference_architecture(rule_based)
    )
