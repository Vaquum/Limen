import numpy as np

from loop.metrics.binary_metrics import binary_metrics


EPSILON = 1e-10
TRADING_DAYS_PER_YEAR = 252
HOURS_PER_DAY = 12
RETURN_THRESHOLD = 0.001


def momentum_volatility_longonly(data: dict,
                                  window_size: int = 48,
                                  momentum_buy_pct: int = 65,
                                  momentum_sell_pct: int = 40,
                                  volatility_buy_pct: int = 80,
                                  volatility_sell_pct: int = 85,
                                  lookback_window: int = 500,
                                  trading_cost: float = 0.001) -> dict:

    '''
    Rules-based long-only momentum-volatility strategy using dynamic percentile thresholds.

    Args:
        data (dict): Data dictionary with x_test containing close prices
        window_size (int): Rolling window size for momentum/volatility calculation (default: 48)
        momentum_buy_pct (int): Percentile threshold for long entry (default: 65)
        momentum_sell_pct (int): Percentile threshold for long exit (default: 40)
        volatility_buy_pct (int): Volatility percentile threshold for entry (default: 80)
        volatility_sell_pct (int): Volatility percentile threshold for exit (default: 85)
        lookback_window (int): Historical window for threshold calculation (default: 500)
        trading_cost (float): Trading cost per trade (default: 0.001)

    Returns:
        dict: Results with binary metrics and trading performance extras
    '''

    closes = data['x_test']['close'].to_numpy()

    returns_history = []
    for i in range(len(closes)):
        if i > 1:
            ret = (closes[i-1] - closes[i-2]) / (closes[i-2] + EPSILON)
            returns_history.append(ret)

    positions = []
    position_open = False
    pending_order = None

    for i in range(len(closes)):
        if pending_order is not None and i > 0:
            if pending_order['type'] == 'buy' and not position_open:
                position_open = True
            elif pending_order['type'] == 'sell' and position_open:
                position_open = False
            pending_order = None

        if i < 2 or len(returns_history) < window_size:
            positions.append(1 if position_open else 0)
            continue

        current_returns_idx = min(i - 2, len(returns_history) - 1)

        if current_returns_idx < window_size - 1:
            positions.append(1 if position_open else 0)
            continue

        recent_returns = returns_history[max(0, current_returns_idx - window_size + 1):current_returns_idx + 1]

        if len(recent_returns) < window_size:
            positions.append(1 if position_open else 0)
            continue

        current_momentum = np.mean(recent_returns)
        current_volatility = np.std(recent_returns)

        if current_returns_idx + 1 >= min(lookback_window, 100):
            lookback = min(lookback_window, current_returns_idx + 1)
            historical_returns = returns_history[max(0, current_returns_idx - lookback + 1):current_returns_idx + 1]

            momentums = []
            volatilities = []

            for j in range(window_size, len(historical_returns) + 1):
                window = historical_returns[j-window_size:j]
                momentums.append(np.mean(window))
                volatilities.append(np.std(window))

            if momentums and volatilities:
                mom_buy_thresh = np.percentile(momentums, momentum_buy_pct)
                mom_sell_thresh = np.percentile(momentums, momentum_sell_pct)
                vol_buy_thresh = np.percentile(volatilities, volatility_buy_pct)
                vol_sell_thresh = np.percentile(volatilities, volatility_sell_pct)
            else:
                positions.append(1 if position_open else 0)
                continue
        else:
            positions.append(1 if position_open else 0)
            continue

        if position_open and pending_order is None:
            if current_momentum < mom_sell_thresh or current_volatility > vol_sell_thresh:
                if i < len(closes) - 1:
                    pending_order = {'type': 'sell', 'bar': i}

        if not position_open and pending_order is None:
            if current_momentum > mom_buy_thresh and current_volatility < vol_buy_thresh:
                if i < len(closes) - 1:
                    pending_order = {'type': 'buy', 'bar': i}

        positions.append(1 if position_open else 0)

    positions_array = np.array(positions)

    num_trades = 0
    for i in range(1, len(positions)):
        if positions[i] != positions[i-1]:
            num_trades += 1

    actual_returns = []
    for i in range(len(closes) - 1):
        actual_returns.append((closes[i+1] - closes[i]) / (closes[i] + EPSILON))
    actual_returns.append(0)

    actual_returns_array = np.array(actual_returns)

    strategy_returns = positions_array[:-1] * actual_returns_array[:-1]

    cost_adjustment = num_trades * trading_cost

    total_positions = np.sum(positions_array)
    total_return = np.sum(strategy_returns) - cost_adjustment

    if total_positions > 0:
        positions_sum = np.sum(positions_array[:-1])
        avg_return_when_long = total_return / (positions_sum if positions_sum > 0 else 1)
        winning_periods = np.sum(strategy_returns > 0)
        periods_in_position = np.sum(positions_array[:-1] > 0)
        win_rate = winning_periods / periods_in_position if periods_in_position > 0 else 0

        annualization_factor = np.sqrt(TRADING_DAYS_PER_YEAR * HOURS_PER_DAY)
        if len(strategy_returns) > 1:
            returns_std = np.std(strategy_returns)
            if returns_std > EPSILON:
                sharpe = np.mean(strategy_returns) / returns_std * annualization_factor
            else:
                sharpe = 0
        else:
            sharpe = 0
    else:
        avg_return_when_long = 0
        win_rate = 0
        sharpe = 0

    y_true = np.zeros(len(actual_returns_array) - 1)
    y_true[actual_returns_array[:-1] > RETURN_THRESHOLD] = 1

    y_pred = positions_array[:-1]

    data_dict = {'y_test': y_true}

    y_proba = np.zeros(len(y_pred))
    y_proba[y_pred == 1] = 0.8
    y_proba[y_pred == 0] = 0.2

    metrics = binary_metrics(data_dict, y_pred, y_proba)

    return {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'auc': metrics['auc'],
        '_preds': positions,
        'extras': {
            'total_return': round(total_return * 100, 2),
            'avg_return_when_long': round(avg_return_when_long * 100, 3),
            'num_long_positions': int(total_positions),
            'position_rate': round(total_positions / len(positions), 3) if len(positions) > 0 else 0,
            'sharpe_ratio': round(sharpe, 2),
            'num_trades': num_trades,
            'win_rate': round(win_rate * 100, 2),
            'momentum_buy_pct': momentum_buy_pct,
            'momentum_sell_pct': momentum_sell_pct,
            'volatility_buy_pct': volatility_buy_pct,
            'volatility_sell_pct': volatility_sell_pct,
            'lookback_window': lookback_window
        }
    }
