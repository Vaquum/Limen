import math
from limen.trading import Account

class BacktestSequential:
    
    def __init__(self, start_usdt=30000):

        self.fee_rate = 0.001

        self.account = Account(start_usdt)
        self.start_usdt = start_usdt
        self.trades = []
        self.equity_curve = []
        
    def run(self, actual, prediction, price_change, open_prices, close_prices):
        
        if not all(len(arr) == len(actual) for arr in [prediction, price_change, open_prices, close_prices]):
            raise ValueError('ERROR: Arrays must have same length')
        
        for i in range(len(actual)):
            pred = prediction[i]
            act = actual[i]
            open_price = open_prices[i]
            close_price = close_prices[i]
            
            if open_price <= 0 or close_price <= 0:
                continue
                
            current_usdt = self.account.account['total_usdt'][-1]
            
            if pred == 1:
                if current_usdt > 1:
                    buy_fee = current_usdt * self.fee_rate
                    usdt_after_buy_fee = current_usdt - buy_fee
                    if usdt_after_buy_fee <= 0:
                        continue
                    self.account.update_account('buy', usdt_after_buy_fee, open_price)
                    btc_held = self.account.long_position
                    gross_sell_amount = btc_held * close_price
                    sell_fee = gross_sell_amount * self.fee_rate
                    net_sell_amount = gross_sell_amount - sell_fee
                    self.account.update_account('sell', net_sell_amount, close_price)
                    final_usdt = self.account.account['total_usdt'][-1]
                    profit = final_usdt - current_usdt
                    self.trades.append({'type': 'long', 'hit': act == pred, 'pnl': profit, 'volume': usdt_after_buy_fee})
                    self.equity_curve.append(final_usdt)
            
        return self._calculate_metrics()
    
    def _calculate_metrics(self):
        
        if not self.trades:
            return {
                'PnL': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'expected_value': 0,
                'sharpe_ratio': 0,
                'net_long_volume': 0,
                'net_short_volume': 0,
                'net_trade_volume': 0
            }
            
        final_usdt = self.account.account['total_usdt'][-1]
        pnl = final_usdt - self.start_usdt
        
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total_trades = len(self.trades)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        max_drawdown = 0
        if self.equity_curve:
            peak = self.equity_curve[0]
            for equity in self.equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                
        expected_value = sum(t['pnl'] for t in self.trades) / total_trades if total_trades > 0 else 0
        
        net_long_volume = sum(t['volume'] for t in self.trades if t['type'] == 'long')
        net_short_volume = sum(t['volume'] for t in self.trades if t['type'] == 'short')
        net_trade_volume = net_long_volume + net_short_volume
        
        returns = []
        if len(self.equity_curve) > 1:
            prev_equity = self.start_usdt
            for equity in self.equity_curve:
                if prev_equity != 0:
                    returns.append((equity - prev_equity) / prev_equity)
                prev_equity = equity
        
        if len(returns) > 1:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            std_return = math.sqrt(variance)
            sharpe_ratio = mean_return / std_return if std_return != 0 else 0
        else:
            sharpe_ratio = 0
            
        # NOTE: These must all be rounded to 2 decimal places.
        return {
            'PnL': round(pnl, 2),
            'win_rate': round(win_rate, 2),
            'max_drawdown': round(max_drawdown, 2),
            'expected_value': round(expected_value, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'net_long_volume': round(net_long_volume, 2),
            'net_short_volume': round(net_short_volume, 2),
            'net_trade_volume': round(net_trade_volume, 2)
        }
