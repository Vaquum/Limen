import numpy as np
import polars as pl

from limen.utils.find_price_lines import find_price_lines

DEFAULT_LOOKAHEAD_HOURS = 48
DEFAULT_CONFIRMATION_HOURS = 24
PERCENTILE_UPPER_BOUND = 100


class TradelineLongBinaryTarget:

    '''Binary long target from a train-fitted line-height percentile threshold.'''

    def __init__(self,
                 train_data: pl.DataFrame,
                 target_name: str,
                 max_duration_hours: int,
                 min_height_pct: float,
                 long_threshold_percentile: float) -> None:

        '''
        Fit the breakout threshold from long lines detected on the training split.

        Args:
            train_data (pl.DataFrame): Training split with a 'close' column
            target_name (str): Name of the target column to create
            max_duration_hours (int): Exclusive upper bound on line duration in bars
            min_height_pct (float): Minimum absolute line height as a fraction of start price
            long_threshold_percentile (float): Percentile of long-line heights used
                as the breakout threshold, in [0, 100]

        Raises:
            ValueError: If 'close' is missing, the percentile is outside [0, 100],
                or no long lines exist in the training split
        '''

        if 'close' not in train_data.columns:
            raise ValueError('TradelineLongBinaryTarget train_data must contain a close column')

        if not 0 <= long_threshold_percentile <= PERCENTILE_UPPER_BOUND:
            raise ValueError('TradelineLongBinaryTarget long_threshold_percentile must be between 0 and 100')

        long_lines, _ = find_price_lines(
            train_data['close'].to_numpy(), max_duration_hours, min_height_pct
        )

        if not long_lines:
            raise ValueError(
                'TradelineLongBinaryTarget found no long price lines in the training split; '
                'loosen max_duration_hours or min_height_pct'
            )

        heights = np.abs(np.array([line['height_pct'] for line in long_lines]))

        self.target_name = target_name
        self.threshold: float = float(np.percentile(heights, long_threshold_percentile))

    def transform(self,
                  data: pl.DataFrame,
                  lookahead_hours: int = DEFAULT_LOOKAHEAD_HOURS,
                  confirmation_hours: int = DEFAULT_CONFIRMATION_HOURS) -> pl.DataFrame:

        '''
        Apply the confirmed-breakout label using the fitted threshold.

        Label 1 where the maximum close over [t, t + lookahead_hours] reaches
        the threshold AND the point return at +confirmation_hours or at
        +lookahead_hours also exceeds it; 0 otherwise. Rows without a full
        forward window become null.

        Args:
            data (pl.DataFrame): Split to transform (must have 'close' column)
            lookahead_hours (int): Forward window for the breakout check
            confirmation_hours (int): Forward offset for the point-confirmation check

        Returns:
            pl.DataFrame: Data with the target column added

        Raises:
            ValueError: If 'close' is missing or a window is below 1
        '''

        if 'close' not in data.columns:
            raise ValueError('TradelineLongBinaryTarget data must contain a close column')

        if lookahead_hours < 1:
            raise ValueError('TradelineLongBinaryTarget lookahead_hours must be at least 1')

        if confirmation_hours < 1:
            raise ValueError('TradelineLongBinaryTarget confirmation_hours must be at least 1')

        close = pl.col('close')

        forward_max = close.reverse().rolling_max(window_size=lookahead_hours + 1).reverse()
        breakout_return = forward_max / close - 1
        confirmation_a = close.shift(-confirmation_hours) / close - 1
        confirmation_b = close.shift(-lookahead_hours) / close - 1

        full_window = (
            breakout_return.is_not_null()
            & confirmation_a.is_not_null()
            & confirmation_b.is_not_null()
        )
        label = (
            (breakout_return >= self.threshold)
            & ((confirmation_a > self.threshold) | (confirmation_b > self.threshold))
        )

        return data.with_columns(
            pl.when(full_window)
            .then(label)
            .otherwise(None)
            .cast(pl.UInt8)
            .alias(self.target_name)
        )
