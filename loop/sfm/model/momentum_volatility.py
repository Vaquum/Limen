import numpy as np

from loop.metrics.multiclass_metrics import multiclass_metrics


EPSILON = 1e-10
TRADING_DAYS_PER_YEAR = 252
HOURS_PER_DAY = 12
RETURN_THRESHOLD = 0.001


def momentum_volatility(data: dict,
                        window_size: int = 48,
                        momentum_buy_pct: int = 65,
                        momentum_sell_pct: int = 40,
                        momentum_short_pct: int = 30,
                        momentum_cover_pct: int = 60,
                        volatility_entry_pct: int = 80,
                        volatility_exit_pct: int = 85,
                        lookback_window: int = 500,
                        trading_cost: float = 0.001,
                        **kwargs) -> dict:

    '''
    Rules-based bidirectional momentum-volatility strategy using dynamic percentile thresholds.
    Supports both long and short positions.

    Args:
        data (dict): Data dictionary with x_test containing close prices
        window_size (int): Rolling window size for momentum/volatility calculation (default: 48)
        momentum_buy_pct (int): Percentile threshold for long entry (default: 65)
        momentum_sell_pct (int): Percentile threshold for long exit (default: 40)
        momentum_short_pct (int): Percentile threshold for short entry (default: 30)
        momentum_cover_pct (int): Percentile threshold for short exit (default: 60)
        volatility_entry_pct (int): Volatility percentile threshold for entry (default: 80)
        volatility_exit_pct (int): Volatility percentile threshold for exit (default: 85)
        lookback_window (int): Historical window for threshold calculation (default: 500)
        trading_cost (float): Trading cost per trade (default: 0.001)
        **kwargs: Additional parameters (ignored)

    Returns:
        dict: Results with multiclass metrics and trading performance extras
    '''

    closes = data['x_test']['close'].to_numpy()

    returns_history = []
    for i in range(len(closes)):
        if i > 1:
            ret = (closes[i-1] - closes[i-2]) / (closes[i-2] + EPSILON)
            returns_history.append(ret)

    positions = []
    position = 0
    pending_order = None

    long_trades = 0
    short_trades = 0
    long_wins = 0
    short_wins = 0
    entry_price = 0

    for i in range(len(closes)):
        if pending_order is not None and i > 0:
            if pending_order['type'] == 'buy' and position == 0:
                position = 1
                entry_price = closes[i]
                long_trades += 1
            elif pending_order['type'] == 'sell' and position == 1:
                if closes[i] > entry_price:
                    long_wins += 1
                position = 0
            elif pending_order['type'] == 'short' and position == 0:
                position = -1
                entry_price = closes[i]
                short_trades += 1
            elif pending_order['type'] == 'cover' and position == -1:
                if closes[i] < entry_price:
                    short_wins += 1
                position = 0
            pending_order = None

        if i < 2 or len(returns_history) < window_size:
            positions.append(position)
            continue

        current_returns_idx = min(i - 2, len(returns_history) - 1)

        if current_returns_idx < window_size - 1:
            positions.append(position)
            continue

        recent_returns = returns_history[max(0, current_returns_idx - window_size + 1):current_returns_idx + 1]

        if len(recent_returns) < window_size:
            positions.append(position)
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
                mom_short_thresh = np.percentile(momentums, momentum_short_pct)
                mom_cover_thresh = np.percentile(momentums, momentum_cover_pct)
                vol_entry_thresh = np.percentile(volatilities, volatility_entry_pct)
                vol_exit_thresh = np.percentile(volatilities, volatility_exit_pct)
            else:
                positions.append(position)
                continue
        else:
            positions.append(position)
            continue

        if position == 1 and pending_order is None:
            if current_momentum < mom_sell_thresh or current_volatility > vol_exit_thresh:
                if i < len(closes) - 1:
                    pending_order = {'type': 'sell', 'bar': i}

        elif position == -1 and pending_order is None:
            if current_momentum > mom_cover_thresh or current_volatility > vol_exit_thresh:
                if i < len(closes) - 1:
                    pending_order = {'type': 'cover', 'bar': i}

        elif position == 0 and pending_order is None:
            if current_momentum > mom_buy_thresh and current_volatility < vol_entry_thresh:
                if i < len(closes) - 1:
                    pending_order = {'type': 'buy', 'bar': i}
            elif current_momentum < mom_short_thresh and current_volatility < vol_entry_thresh:
                if i < len(closes) - 1:
                    pending_order = {'type': 'short', 'bar': i}

        positions.append(position)

    positions_array = np.array(positions)

    num_trades = 0
    for i in range(1, len(positions)):
        if positions[i] != positions[i-1] and positions[i-1] != 0:
            num_trades += 1

    actual_returns = []
    for i in range(len(closes) - 1):
        actual_returns.append((closes[i+1] - closes[i]) / (closes[i] + EPSILON))
    actual_returns.append(0)

    actual_returns_array = np.array(actual_returns)

    strategy_returns = positions_array[:-1] * actual_returns_array[:-1]

    cost_adjustment = num_trades * trading_cost

    total_return = np.sum(strategy_returns) - cost_adjustment

    long_positions = np.sum(positions_array == 1)
    short_positions = np.sum(positions_array == -1)
    total_positions = long_positions + short_positions

    if total_positions > 0:
        winning_periods = np.sum(strategy_returns > 0)
        periods_in_position = np.sum(positions_array[:-1] != 0)
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
        win_rate = 0
        sharpe = 0

    long_win_rate = (long_wins / long_trades * 100) if long_trades > 0 else 0
    short_win_rate = (short_wins / short_trades * 100) if short_trades > 0 else 0

    y_true = np.zeros(len(actual_returns_array) - 1)
    y_true[actual_returns_array[:-1] > RETURN_THRESHOLD] = 1
    y_true[actual_returns_array[:-1] < -RETURN_THRESHOLD] = 2

    y_pred = positions_array[:-1].copy()
    y_pred[y_pred == -1] = 2

    data_dict = {'y_test': y_true}

    y_proba = np.zeros((len(y_pred), 3))
    for i, pred in enumerate(y_pred):
        y_proba[i, int(pred)] = 0.8
        for j in range(3):
            if j != int(pred):
                y_proba[i, j] = 0.1

    metrics = multiclass_metrics(data_dict, y_pred, y_proba)

    return {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'auc': metrics['auc'],
        '_preds': positions,
        'extras': {
            'total_return': round(total_return * 100, 2),
            'num_long_positions': int(long_positions),
            'num_short_positions': int(short_positions),
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_win_rate': round(long_win_rate, 1),
            'short_win_rate': round(short_win_rate, 1),
            'position_rate': round(total_positions / len(positions), 3) if len(positions) > 0 else 0,
            'sharpe_ratio': round(sharpe, 2),
            'num_trades': num_trades,
            'win_rate': round(win_rate * 100, 2),
            'momentum_buy_pct': momentum_buy_pct,
            'momentum_sell_pct': momentum_sell_pct,
            'momentum_short_pct': momentum_short_pct,
            'momentum_cover_pct': momentum_cover_pct,
            'volatility_entry_pct': volatility_entry_pct,
            'volatility_exit_pct': volatility_exit_pct,
            'lookback_window': lookback_window
        }
    }
