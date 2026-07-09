import numpy as np
import numpy.typing as npt


def stddev_using_precalc_ma(
    values: npt.NDArray[np.float64],
    movavg: npt.NDArray[np.float64],
    movavg_beg_idx: int,
    movavg_nb_element: int,
    period: int,
) -> npt.NDArray[np.float64]:
    out = np.empty(movavg_nb_element, dtype=float)

    start_sum = 1 + movavg_beg_idx - period
    end_sum = movavg_beg_idx

    period_total2 = 0.0
    out_idx = start_sum
    while out_idx < end_sum:
        temp = values[out_idx]
        period_total2 += temp * temp
        out_idx += 1

    out_idx = 0
    while out_idx < movavg_nb_element:
        temp = values[end_sum]
        period_total2 += temp * temp
        mean_value2 = period_total2 / period

        temp = values[start_sum]
        period_total2 -= temp * temp

        temp = movavg[movavg_beg_idx + out_idx]
        mean_value2 -= temp * temp

        out[out_idx] = np.sqrt(mean_value2) if mean_value2 > 0.0 else 0.0

        out_idx += 1
        start_sum += 1
        end_sum += 1

    return out


def stddev_from_var(
    values: npt.NDArray[np.float64],
    start_idx: int,
    end_idx: int,
    period: int,
) -> tuple[int, npt.NDArray[np.float64]]:
    nb_initial = period - 1
    start = max(start_idx, nb_initial)
    if start > end_idx:
        return start, np.empty(0, dtype=float)

    period_total1 = 0.0
    period_total2 = 0.0
    trailing_idx = start - nb_initial

    i = trailing_idx
    while i < start:
        temp = values[i]
        period_total1 += temp
        period_total2 += temp * temp
        i += 1

    out = []
    while i <= end_idx:
        temp = values[i]
        i += 1

        period_total1 += temp
        period_total2 += temp * temp

        mean1 = period_total1 / period
        mean2 = period_total2 / period

        temp = values[trailing_idx]
        trailing_idx += 1
        period_total1 -= temp
        period_total2 -= temp * temp

        var_value = mean2 - (mean1 * mean1)
        out.append(np.sqrt(var_value) if var_value > 0.0 else 0.0)

    return start, np.asarray(out, dtype=float)
