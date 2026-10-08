from collections.abc import Mapping
from dataclasses import dataclass

from limen.backtest.trade_contract import JsonValue

PRESET_VERSION = '2026-10-08'
BASELINE_RATE_8H = 0.000028


@dataclass(frozen=True)
class FundingPreset:
    instrument: str
    params: Mapping[str, JsonValue]
    calibration: Mapping[str, JsonValue]


def _calibration() -> dict[str, JsonValue]:
    return {'status': 'scenario_assumption', 'asset': 'BTC',
            'window_start_utc': '2025-10-08T00:00:00Z',
            'window_end_utc': '2026-10-08T00:00:00Z',
            'rate_basis_seconds': 28800.0, 'baseline_rate_decimal': BASELINE_RATE_8H,
            'evidence_sha256': '3428ede4f1b657bc4bf8ca3ab98b317e336bce47ff558e3ba763a5d07fd9fae3', 'version': PRESET_VERSION}


PRESETS: Mapping[str, FundingPreset] = {
    'binance_btcusdt': FundingPreset('BTCUSDT',
        {'rate': BASELINE_RATE_8H, 'mechanism': 'discrete', 'rate_unit': 'decimal', 'rate_basis_seconds': 28800.0, 'settlement_interval_seconds': 28800.0, 'settlement_phase_utc_seconds': 0.0, 'valuation': 'execution_proxy', 'currency': 'USDT', 'approximation': 'scenario'},
        _calibration()),
    'hyperliquid_btc': FundingPreset('BTC',
        {'rate': BASELINE_RATE_8H / 8, 'mechanism': 'discrete', 'rate_unit': 'decimal', 'rate_basis_seconds': 3600.0, 'settlement_interval_seconds': 3600.0, 'settlement_phase_utc_seconds': 0.0, 'valuation': 'execution_proxy', 'currency': 'USDC', 'approximation': 'scenario'},
        _calibration()),
    'deribit_btc_usdc': FundingPreset('BTC_USDC-PERPETUAL',
        {'rate': BASELINE_RATE_8H, 'mechanism': 'continuous', 'rate_unit': 'decimal', 'rate_basis_seconds': 28800.0, 'cash_settlement_interval_seconds': 86400.0, 'cash_settlement_phase_utc_seconds': 28800.0, 'valuation': 'execution_proxy', 'currency': 'USDC', 'approximation': 'scenario'},
        _calibration()),
}


__all__ = ['BASELINE_RATE_8H', 'PRESETS', 'PRESET_VERSION', 'FundingPreset']
