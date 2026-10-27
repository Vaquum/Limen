import polars as pl
import limen.targets._perturbation as _noise


class ForwardBreakoutTarget:

    '''Binary target for forward price breakouts above a percentage threshold.'''
    requires_permutation_id = True

    def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:

        '''
        Store the target column name; no fitting is performed.

        Args:
            train_data (pl.DataFrame): Training split (not used; kept for interface consistency)
            target_name (str): Name of the target column to create
        '''

        super().__init__()

        self.target_name = target_name
        self.perturbation_parameters: dict[str, object] = {}

    def transform(self,
                  data: pl.DataFrame,
                  forward_periods: int = 24,
                  threshold: float = 0.02,
                  shift: int = -1, *, horizon_jitter: int = 0, threshold_noise_std: float = 0.,
                  noise_seed: int = 0, label_smoothing: float = 0.,
                  permutation_id: str | None = None, training: bool = False) -> pl.DataFrame:

        '''
        Label 1 if price rises >= threshold over next forward_periods bars.

        Args:
            data (pl.DataFrame): Split to transform (must have 'close' column)
            forward_periods (int): Look-ahead window in bars
            threshold (float): Minimum return for positive label (0.02 = 2%)
            shift (int): Additional shift applied after labeling

        Returns:
            pl.DataFrame: Data with target column added
        '''

        forward_periods, (threshold,) = _noise.parameters(forward_periods, (threshold,), horizon_jitter=horizon_jitter,
            noise_std=threshold_noise_std, noise_seed=noise_seed, permutation_id=permutation_id)
        _noise.record_parameters(self, bool(horizon_jitter or threshold_noise_std), {'permutation_id': permutation_id, 'forward_periods': forward_periods, 'threshold': threshold})
        forward_return = (pl.col('close').shift(-forward_periods) - pl.col('close')) / pl.col('close')
        labelled = data.with_columns(
            (forward_return >= threshold).cast(pl.UInt8).shift(shift).alias(self.target_name)
        )
        return _noise.smooth_binary(labelled, self.target_name, label_smoothing, training=training)
