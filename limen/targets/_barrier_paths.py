import numpy as np
import numpy.typing as npt

__all__: list[str] = []


def first_touch(close: npt.NDArray[np.float64], volatility: npt.NDArray[np.float64],
                upper_multiple: float, lower_multiple: float, max_horizon: int) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.bool_]]:
    n = close.shape[0]
    upper = upper_multiple * volatility
    lower = lower_multiple * volatility
    label = np.zeros(n, dtype=np.int8)
    touched = np.zeros(n, dtype=bool)

    for k in range(1, max_horizon + 1):
        if k >= n:
            break
        prefix = n - k
        forward_return = close[k:] / close[:prefix] - 1.0
        open_bars = ~touched[:prefix]
        hit_upper = open_bars & (forward_return >= upper[:prefix])
        hit_lower = open_bars & (forward_return <= -lower[:prefix])
        label[:prefix][hit_upper] = 1
        label[:prefix][hit_lower] = -1
        touched[:prefix] |= hit_upper | hit_lower

    horizon_available = np.arange(n) <= (n - 1 - max_horizon)
    valid = (volatility > 0.0) & (touched | horizon_available)
    return label, valid
