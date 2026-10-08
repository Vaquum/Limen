from collections.abc import Mapping
from dataclasses import dataclass

from limen.backtest.trade_contract import JsonValue

PRESET_VERSION = '2026-10-08'


@dataclass(frozen=True)
class FundingPreset:
    instrument: str
    params: Mapping[str, JsonValue]
    calibration: Mapping[str, JsonValue]


def _calibration(venue: str, instrument: str, endpoint: str) -> dict[str, JsonValue]:
    return {'venue': venue, 'instrument': instrument, 'endpoint': endpoint,
            'status': 'redistribution_rights_unverified'}


PRESETS: Mapping[str, FundingPreset] = {
    'binance_btcusdt': FundingPreset('BTCUSDT',
        {'mechanism': 'discrete', 'rate_unit': 'decimal', 'rate_basis_seconds': 28800.0, 'settlement_interval_seconds': 28800.0, 'settlement_phase_utc_seconds': 0.0, 'valuation': 'execution_proxy', 'currency': 'USDT', 'approximation': 'scenario'},
        _calibration('binance', 'BTCUSDT', 'https://fapi.binance.com/fapi/v1/fundingRate')),
    'hyperliquid_btc': FundingPreset('BTC',
        {'mechanism': 'discrete', 'rate_unit': 'decimal', 'rate_basis_seconds': 3600.0, 'settlement_interval_seconds': 3600.0, 'settlement_phase_utc_seconds': 0.0, 'valuation': 'execution_proxy', 'currency': 'USDC', 'approximation': 'scenario'},
        _calibration('hyperliquid', 'BTC', 'https://api.hyperliquid.xyz/info')),
    'deribit_btc_usdc': FundingPreset('BTC_USDC-PERPETUAL',
        {'mechanism': 'continuous', 'rate_unit': 'decimal', 'rate_basis_seconds': 28800.0, 'cash_settlement_interval_seconds': 86400.0, 'cash_settlement_phase_utc_seconds': 28800.0, 'valuation': 'execution_proxy', 'currency': 'USDC', 'approximation': 'scenario'},
        _calibration('deribit', 'BTC_USDC-PERPETUAL', 'https://www.deribit.com/api/v2/public/get_funding_rate_history')),
}


__all__ = ['PRESETS', 'PRESET_VERSION', 'FundingPreset']
