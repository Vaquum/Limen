import polars as pl


def _train_segments(lo: int,
                    hi: int,
                    zones: list[tuple[int, int]]) -> list[tuple[int, int]]:

    '''
    Compute the kept sub-intervals of `[lo, hi)` after removing zones.

    Args:
        lo (int): Inclusive start of the candidate train interval
        hi (int): Exclusive end of the candidate train interval
        zones (list): Excluded `[start, end)` intervals sorted by start

    Returns:
        list[tuple[int, int]]: Kept `[start, end)` intervals in order
    '''

    segments: list[tuple[int, int]] = []
    cursor = lo

    for zone_lo, zone_hi in zones:
        if zone_hi <= cursor or zone_lo >= hi:
            continue
        if zone_lo > cursor:
            segments.append((cursor, zone_lo))
        cursor = max(cursor, zone_hi)
        if cursor >= hi:
            break

    if cursor < hi:
        segments.append((cursor, hi))

    return segments


def split_walk_forward(data: pl.DataFrame,
                       *,
                       n_folds: int,
                       test_bars: int,
                       purge_bars: int,
                       embargo_bars: int,
                       anchored: bool) -> list[tuple[pl.DataFrame, pl.DataFrame]]:

    '''
    Compute purged and embargoed walk-forward folds over a time-ordered frame.

    The last `n_folds * test_bars` rows are partitioned into contiguous,
    non-overlapping test windows rolling forward. Each fold's train window
    is every row before its test start minus the trailing `purge_bars`
    rows (label-horizon purge), minus the `embargo_bars` rows following
    every earlier test window (embargo). With `anchored=True` the train
    window grows from row zero; with `anchored=False` it keeps a fixed
    width equal to the first fold's span.

    Args:
        data (pl.DataFrame): Time-ordered Polars DataFrame to fold
        n_folds (int): Number of walk-forward folds, at least 1
        test_bars (int): Rows per test window, at least 1
        purge_bars (int): Rows purged before each test start, at least 0
        embargo_bars (int): Rows embargoed after each earlier test window, at least 0
        anchored (bool): Grow train from row zero when True, fixed width when False

    Returns:
        list[tuple[pl.DataFrame, pl.DataFrame]]: Ordered (train, test) pairs

    Raises:
        ValueError: If a parameter is out of range or any fold's train or
            test window is empty under the given geometry
    '''

    if n_folds < 1:
        raise ValueError(f'split_walk_forward n_folds must be at least 1, got {n_folds}')
    if test_bars < 1:
        raise ValueError(f'split_walk_forward test_bars must be at least 1, got {test_bars}')
    if purge_bars < 0:
        raise ValueError(f'split_walk_forward purge_bars must be at least 0, got {purge_bars}')
    if embargo_bars < 0:
        raise ValueError(f'split_walk_forward embargo_bars must be at least 0, got {embargo_bars}')

    total = data.height
    first_test_start = total - n_folds * test_bars
    train_width = first_test_start - purge_bars

    if train_width < 1:
        raise ValueError(
            f'split_walk_forward geometry leaves no train rows: height {total} with n_folds {n_folds}, test_bars {test_bars}, purge_bars {purge_bars}'
        )

    folds: list[tuple[pl.DataFrame, pl.DataFrame]] = []

    for fold in range(n_folds):
        test_start = first_test_start + fold * test_bars
        train_hi = test_start - purge_bars
        train_lo = 0 if anchored else fold * test_bars

        zones = [
            (first_test_start + (earlier + 1) * test_bars,
             first_test_start + (earlier + 1) * test_bars + embargo_bars)
            for earlier in range(fold)
        ]
        segments = _train_segments(train_lo, train_hi, zones)

        if not segments:
            raise ValueError(
                f'split_walk_forward fold {fold} train window is empty after purge and embargo'
            )

        if len(segments) == 1:
            start, end = segments[0]
            train = data.slice(start, end - start)
        else:
            train = pl.concat([data.slice(start, end - start) for start, end in segments])
        test = data.slice(test_start, test_bars)
        folds.append((train, test))

    return folds
