from datetime import datetime
import math

class Account:
        
    '''Account class is used to keep track of account information for both long and short positions'''
    
    def __init__(self, start_usdt):
        '''Initializes the account object.
        
        start_usdt | int | starting usdt balance
        '''
        self.position_id = 0
        self.TOLERANCE_BTC = 1e-14
        
        # PERFORMANCE FIX: Cached running totals for O(1) property access
        self._cached_long_position = 0.0
        self._cached_short_position = 0.0
        
        self.account = self._init_account(credit_usdt=start_usdt)
        
    def _init_account(self, credit_usdt):
        
        '''Initializes the account with the starting balance.
        
        credit_usdt | int | starting usdt balance
        '''
        account = {'position_id': [self.position_id],
                   'action': ['hold'],
                   'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                   'credit_usdt': [credit_usdt],
                   'debit_usdt': [0],
                   'amount_bought_btc': [0],
                   'amount_sold_btc': [0],
                   'amount_borrowed_btc': [0],        # Added field: tracks borrowed BTC for shorts
                   'amount_covered_btc': [0],         # Added field: tracks covered (repaid) BTC
                   'buy_price_usdt': [0],
                   'sell_price_usdt': [0],
                   'total_usdt': [credit_usdt],
                   'total_btc': [0]}
        
        return account
    
    def update_account(self,
                      action,
                      amount,
                      price_usdt):
        
        '''Updates the account information based on the action taken.
        
        action | str | 'buy', 'sell', 'short', 'cover', or 'hold'
        timestamp | datetime | current timestamp
        amount | int | amount in USDT (for buy/short) or BTC (for sell/cover)
        price_usdt | float | current price of the asset
        '''
        
        # Initialize values
        credit_usdt = 0
        debit_usdt = 0
        amount_bought_btc = 0
        amount_sold_btc = 0
        amount_borrowed_btc = 0
        amount_covered_btc = 0
        buy_price_usdt = 0
        sell_price_usdt = 0

        if action not in ['buy', 'sell', 'short', 'cover', 'hold']:
            raise ValueError('ERROR: ' + action + ' not supported.')
        if not (amount > 0 or action == 'hold'):
            raise ValueError('ERROR: amount must be positive for all actions except hold.')
        if price_usdt <= 0:
            raise ValueError('ERROR: price_usdt has to be positive.')
        
        # OVERFLOW PROTECTION: Check for numerical overflow conditions
        MAX_USDT = 1e12  # 1 trillion USDT limit
        MAX_BTC = 1e9    # 1 billion BTC limit
        
        current_total_usdt = self.account['total_usdt'][-1]
        current_long_btc = self._cached_long_position  # PERFORMANCE FIX: use cached value
        
        if current_total_usdt > MAX_USDT:
            raise ValueError('ERROR: Total USDT exceeds maximum limit')
        if current_long_btc > MAX_BTC:
            raise ValueError('ERROR: Long position exceeds maximum limit')
        if amount > MAX_USDT:
            raise ValueError('ERROR: Transaction amount exceeds maximum limit')
        
        if action == 'buy':

            if amount > self.account['total_usdt'][-1]:
                raise ValueError('ERROR: amount cannot be larger than total_usdt.')
                
            debit_usdt = amount
            amount_bought_btc = round(debit_usdt / price_usdt, 15)  # PRECISION FIX: 15 decimals instead of 7
            buy_price_usdt = price_usdt
        
        elif action == 'sell':
            
            current_long_position = self._cached_long_position  # PERFORMANCE FIX: use cached value
            btc_to_sell = round(amount / price_usdt, 15)
            if btc_to_sell > current_long_position + self.TOLERANCE_BTC:
                raise ValueError('ERROR: Trying to sell more BTC than available')

            credit_usdt = amount
            amount_sold_btc = btc_to_sell
            sell_price_usdt = price_usdt
            
        elif action == 'short':

            amount_borrowed_btc = round(amount / price_usdt, 15)  # PRECISION FIX: 15 decimals instead of 7
            credit_usdt = amount
            sell_price_usdt = price_usdt
            
        elif action == 'cover':

            net_borrowed_btc = self._cached_short_position  # PERFORMANCE FIX: use cached value
            
            if net_borrowed_btc == 0:
                raise ValueError('ERROR: No borrowed BTC to cover')
                
            btc_to_cover = round(amount / price_usdt, 15)  # PRECISION FIX: 15 decimals instead of 7
            if btc_to_cover > net_borrowed_btc + 1e-10:
                raise ValueError('ERROR: Trying to cover more BTC than borrowed')

            amount_covered_btc = btc_to_cover
            debit_usdt = amount
            buy_price_usdt = price_usdt

        # Update the account
        self.position_id += 1
        self.account['position_id'].append(self.position_id)
        self.account['action'].append(action)
        self.account['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.account['credit_usdt'].append(credit_usdt)
        self.account['debit_usdt'].append(debit_usdt)
        self.account['amount_bought_btc'].append(amount_bought_btc)
        self.account['amount_sold_btc'].append(amount_sold_btc)
        self.account['amount_borrowed_btc'].append(amount_borrowed_btc)
        self.account['amount_covered_btc'].append(amount_covered_btc)
        self.account['buy_price_usdt'].append(buy_price_usdt)
        self.account['sell_price_usdt'].append(sell_price_usdt)
    
        # PERFORMANCE FIX: Calculate sums once and cache for O(1) property access
        calculated_long = sum(self.account['amount_bought_btc']) - sum(self.account['amount_sold_btc'])
        calculated_short = sum(self.account['amount_borrowed_btc']) - sum(self.account['amount_covered_btc'])
        calculated_usdt = sum(self.account['credit_usdt']) - sum(self.account['debit_usdt'])
        
        # Update cached values with exact calculated values for perfect consistency
        self._cached_long_position = calculated_long
        self._cached_short_position = calculated_short
        
        # Calculate totals - total_btc represents actual BTC owned (long position only)
        total_btc = calculated_long
        total_usdt = calculated_usdt
        
        # OVERFLOW PROTECTION: Check calculated totals before storing
        if not math.isfinite(total_btc) or not math.isfinite(total_usdt):
            raise ValueError('ERROR: Calculated totals are not finite')
        if total_btc > MAX_BTC or total_usdt > MAX_USDT:
            raise ValueError('ERROR: Calculated totals exceed maximum limits')
        
        self.account['total_btc'].append(round(total_btc, 15))  # PRECISION FIX: 15 decimals instead of 7      
        self.account['total_usdt'].append(round(total_usdt, 2))
        
    @property
    def long_position(self):
        '''BTC owned from regular buys/sells'''
        return self._cached_long_position  # PERFORMANCE FIX: O(1) instead of O(n)
    
    @property
    def short_position(self):
        '''BTC owed from shorts'''
        return self._cached_short_position  # PERFORMANCE FIX: O(1) instead of O(n)
    
    @property
    def net_position(self):
        '''Net BTC exposure (positive=long, negative=short)'''
        return self._cached_long_position - self._cached_short_position  # PERFORMANCE FIX: O(1) instead of O(n)
