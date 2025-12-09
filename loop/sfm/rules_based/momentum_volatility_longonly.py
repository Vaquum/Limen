from loop.manifest import Manifest
from loop.sfm.model.momentum_volatility_longonly import momentum_volatility_longonly


def params():
    return {
        'window_size': [24, 48, 72],
        'momentum_buy_pct': [55, 60, 65, 70, 75, 80],
        'momentum_sell_pct': [30, 35, 40, 45, 50, 55],
        'volatility_buy_pct': [70, 75, 80, 85, 90],
        'volatility_sell_pct': [80, 85, 90, 95],
        'lookback_window': [300, 500, 750],
        'trading_cost': [0.001]
    }


def manifest():
    return (Manifest()
        .set_split_config(80, 10, 10)
        .with_model(momentum_volatility_longonly)
    )
