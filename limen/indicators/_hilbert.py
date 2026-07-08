from typing import cast


def _init_hilbert_state() -> dict[str, list[float] | float]:
    return {
        'odd': [0.0, 0.0, 0.0],
        'even': [0.0, 0.0, 0.0],
        'prev_odd': 0.0,
        'prev_even': 0.0,
        'prev_input_odd': 0.0,
        'prev_input_even': 0.0,
    }


def _do_hilbert_transform(
    state: dict[str, list[float] | float],
    input_value: float,
    adjusted_prev_period: float,
    hilbert_idx: int,
    is_even: bool,
) -> float:
    side = 'even' if is_even else 'odd'
    prev_side = 'prev_even' if is_even else 'prev_odd'
    prev_input_side = 'prev_input_even' if is_even else 'prev_input_odd'

    arr = state[side]
    if not isinstance(arr, list):
        raise TypeError(f"Expected list for state['{side}'], got {type(arr).__name__}")

    hilbert_temp_real = 0.0962 * input_value
    var = -arr[hilbert_idx]
    arr[hilbert_idx] = hilbert_temp_real
    var += hilbert_temp_real

    prev_val = cast(float, state[prev_side])
    var -= prev_val
    prev_val = 0.5769 * cast(float, state[prev_input_side])
    state[prev_side] = prev_val
    var += prev_val

    state[prev_input_side] = input_value
    var *= adjusted_prev_period
    return var
