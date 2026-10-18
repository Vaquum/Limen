from limen.data import HistoricalData
from limen.experiment import Manifest, RuleBasedManifest
from limen.features.common_strategy_positions import POSITION_COLUMN, common_strategy_positions
from limen.sfd.reference_architecture.rule_based import rule_based
from limen.sfd.rule_based.common_strategies import STRATEGY_NAMES


__all__ = ['manifest', 'params']


def params() -> dict[str, list[str]]:
    return {'strategy': list(STRATEGY_NAMES)}


def manifest() -> Manifest:
    configured = RuleBasedManifest()
    _ = configured.set_data_source(
        method=HistoricalData.get_spot_klines,
        params={
            'kline_size': 3600,
            'start_date_limit': '2024-01-01',
            'end_date_limit': '2024-12-31',
        },
    )
    _ = configured.set_test_data_source(
        method=HistoricalData.get_spot_klines,
        params={
            'kline_size': 3600,
            'start_date_limit': '2024-01-01',
            'end_date_limit': '2024-12-31',
        },
    )
    _ = configured.set_required_bar_columns(['datetime', 'open', 'high', 'low', 'close'])
    _ = configured.set_split_config(8, 1, 2)
    _ = configured.add_feature(common_strategy_positions, strategy='strategy')
    _ = configured.set_backtest_config(fee_bps=10.0, slip_bps=5.0)
    return (
        configured.with_strategy([
            {
                'id': 'common_strategy_long',
                'type': 'threshold',
                'column': POSITION_COLUMN,
                'operator': '>',
                'value': 0,
            },
        ], entry='common_strategy_long')
        .with_reference_architecture(rule_based)
    )
