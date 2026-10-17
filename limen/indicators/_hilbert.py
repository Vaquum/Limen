import math
from functools import lru_cache

__all__ = ['_phase_weights', 'do_hilbert_transform', 'init_hilbert_state']


def init_hilbert_state() -> list[float]:
    return [0.0] * 10


def do_hilbert_transform(
    state: list[float],
    input_value: float,
    adjusted_prev_period: float,
    hilbert_idx: int,
    is_even: bool,
) -> float:
    offset = 3 if is_even else 0
    previous = 7 if is_even else 6
    previous_input = 9 if is_even else 8
    hilbert_temp_real = 0.0962 * input_value
    var = -state[offset + hilbert_idx]
    state[offset + hilbert_idx] = hilbert_temp_real
    var += hilbert_temp_real
    var -= state[previous]
    state[previous] = 0.5769 * state[previous_input]
    var += state[previous]
    state[previous_input] = input_value
    return var * adjusted_prev_period


@lru_cache(maxsize=50)
def _phase_weights(period: int) -> tuple[tuple[float, float], ...]:
    circle = math.atan(1.0) * 8.0
    return tuple((math.sin((float(i) * circle) / float(period)),
                  math.cos((float(i) * circle) / float(period)))
                 for i in range(period))
