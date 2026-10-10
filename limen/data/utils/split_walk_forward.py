import polars as pl

__all__ = ['split_walk_forward']


def _train_segments(lo: int,
                    hi: int,
                    zones: list[tuple[int, int]]) -> list[tuple[int, int]]:
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
    """Split an already time-ordered frame by row position.

    The final ``n_folds * test_bars`` rows form contiguous test windows.
    Each candidate train ends ``purge_bars`` before its test start and
    excludes ``embargo_bars`` rows following every earlier test window.
    Anchored trains start at row zero; rolling candidate trains retain
    the first fold's width before embargo removal.
    """
    if n_folds < 1:
        raise ValueError(f'split_walk_forward n_folds must be at least 1, got {n_folds}')
    if test_bars < 1:
        raise ValueError(f'split_walk_forward test_bars must be at least 1, got {test_bars}')
    if purge_bars < 0:
        raise ValueError(f'split_walk_forward purge_bars must be at least 0, got {purge_bars}')
    if embargo_bars < 0:
        raise ValueError(f'split_walk_forward embargo_bars must be at least 0, got {embargo_bars}')

    first_test_start = data.height - n_folds * test_bars
    if first_test_start - purge_bars < 1:
        raise ValueError('split_walk_forward geometry leaves no train rows')

    folds: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    for fold in range(n_folds):
        test_start = first_test_start + fold * test_bars
        train_lo = 0 if anchored else fold * test_bars
        train_hi = test_start - purge_bars
        zones = [
            (first_test_start + (earlier + 1) * test_bars,
             first_test_start + (earlier + 1) * test_bars + embargo_bars)
            for earlier in range(fold)
        ] if embargo_bars else []
        segments = _train_segments(train_lo, train_hi, zones)
        if not segments:
            raise ValueError(f'split_walk_forward fold {fold} train window is empty after embargo')
        parts = [data.slice(start, end - start) for start, end in segments]
        train = parts[0] if len(parts) == 1 else pl.concat(parts)
        folds.append((train, data.slice(test_start, test_bars)))
    return folds
