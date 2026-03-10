import time

import polars as pl

from limen.experiment.reducer.budget_reducer import BudgetReducer
from tests.stubs.stubs import make_msq


def test_constructor_validation():

    try:
        BudgetReducer(trim_strategy='worst_first')
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'metric is required' in str(e)

    try:
        BudgetReducer(trim_strategy='unknown')
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'Unknown trim_strategy' in str(e)


def test_early_returns():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    df = pl.DataFrame({'a': [1], 'b': ['x'], 'score': [0.5]})

    inactive = BudgetReducer(max_permutations=10, active=False)
    assert inactive.analyze_and_intervene(df, msq) == []

    reducer = BudgetReducer(max_permutations=10)
    list(msq)
    assert reducer.analyze_and_intervene(df, msq) == []


def test_trimmed_flag_prevents_retrim():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    next(msq)
    next(msq)
    next(msq)

    reducer = BudgetReducer(
        max_permutations=4,
        check_after_pct=0.0,
    )
    df = pl.DataFrame({'a': [1, 2, 3], 'b': ['x', 'x', 'x'], 'score': [0.5, 0.6, 0.7]})

    result1 = reducer.analyze_and_intervene(df, msq)
    assert len(result1) > 0
    assert reducer._trimmed is True

    result2 = reducer.analyze_and_intervene(df, msq)
    assert result2 == []


def test_permutation_budget_trim_random():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    for _ in range(4):
        next(msq)

    reducer = BudgetReducer(
        max_permutations=5,
        check_after_pct=0.0,
    )
    df = pl.DataFrame({
        'a': [1, 2, 3, 1],
        'b': ['x', 'x', 'x', 'y'],
        'score': [0.5, 0.6, 0.7, 0.8],
    })

    result = reducer.analyze_and_intervene(df, msq)
    assert len(result) == 1
    assert result[0]['op'] == 'trim'
    assert result[0]['target_count'] == 5


def test_permutation_budget_check_after_pct_gate():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    next(msq)

    reducer = BudgetReducer(
        max_permutations=100,
        check_after_pct=0.5,
    )
    df = pl.DataFrame({'a': [1], 'b': ['x'], 'score': [0.5]})

    result = reducer.analyze_and_intervene(df, msq)
    assert result == []


def test_walltime_budget_trim():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    for _ in range(4):
        next(msq)

    start = time.time() - 3600.0
    reducer = BudgetReducer(
        start_time=start,
        max_walltime_hours=1.2,
        check_after_pct=0.0,
    )
    df = pl.DataFrame({
        'a': [1, 2, 3, 1],
        'b': ['x', 'x', 'x', 'y'],
        'score': [0.5, 0.6, 0.7, 0.8],
    })

    result = reducer.analyze_and_intervene(df, msq)
    assert len(result) == 1
    assert result[0]['op'] == 'trim'


def test_walltime_exceeded():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    for _ in range(3):
        next(msq)

    start = time.time() - 7200.0
    reducer = BudgetReducer(
        start_time=start,
        max_walltime_hours=1.0,
        check_after_pct=0.0,
    )
    df = pl.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'x', 'x'],
        'score': [0.5, 0.6, 0.7],
    })

    result = reducer.analyze_and_intervene(df, msq)
    assert len(result) == 1
    assert result[0]['op'] == 'trim'
    assert result[0]['target_count'] == 3


def test_no_trim_when_within_budget():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    next(msq)

    reducer = BudgetReducer(
        max_permutations=100,
        check_after_pct=0.0,
    )
    df = pl.DataFrame({'a': [1], 'b': ['x'], 'score': [0.5]})

    result = reducer.analyze_and_intervene(df, msq)
    assert result == []


def test_worst_first_trim():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    for _ in range(4):
        next(msq)

    reducer = BudgetReducer(
        max_permutations=5,
        trim_strategy='worst_first',
        metric='score',
        maximize=True,
        check_after_pct=0.0,
    )

    df = pl.DataFrame({
        'a': [1, 1, 2, 2, 3, 3],
        'b': ['x', 'y', 'x', 'y', 'x', 'y'],
        'score': [0.1, 0.2, 0.5, 0.6, 0.9, 0.95],
    })

    result = reducer.analyze_and_intervene(df, msq)
    assert len(result) > 0

    remove_ops = [r for r in result if r['op'] == 'remove_is']
    assert len(remove_ops) >= 1
    assert any(r['reason'].startswith('budget worst_first') for r in remove_ops)


def test_worst_first_fallback_to_random():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    for _ in range(4):
        next(msq)

    reducer = BudgetReducer(
        max_permutations=5,
        trim_strategy='worst_first',
        metric='missing_metric',
        check_after_pct=0.0,
    )

    df = pl.DataFrame({
        'a': [1, 2, 3, 1],
        'b': ['x', 'x', 'x', 'y'],
        'score': [0.5, 0.6, 0.7, 0.8],
    })

    result = reducer.analyze_and_intervene(df, msq)
    assert len(result) == 1
    assert result[0]['op'] == 'trim'


def test_both_budgets_takes_min():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    for _ in range(4):
        next(msq)

    start = time.time() - 3600.0
    reducer = BudgetReducer(
        start_time=start,
        max_walltime_hours=10.0,
        max_permutations=5,
        check_after_pct=0.0,
    )
    df = pl.DataFrame({
        'a': [1, 2, 3, 1],
        'b': ['x', 'x', 'x', 'y'],
        'score': [0.5, 0.6, 0.7, 0.8],
    })

    result = reducer.analyze_and_intervene(df, msq)
    assert len(result) == 1
    assert result[0]['op'] == 'trim'
    assert result[0]['target_count'] == 5


def test_no_budget_configured():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    for _ in range(3):
        next(msq)

    reducer = BudgetReducer()
    df = pl.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'x', 'x'],
        'score': [0.5, 0.6, 0.7],
    })

    result = reducer.analyze_and_intervene(df, msq)
    assert result == []


def test_state_roundtrip():

    reducer = BudgetReducer(max_permutations=100)
    reducer._trimmed = True
    reducer._start_time = 12345.0

    state = reducer.get_state()
    assert state['trimmed'] is True
    assert state['start_time'] == 12345.0

    reducer2 = BudgetReducer(max_permutations=100)
    reducer2.set_state(state)
    assert reducer2._trimmed is True
    assert reducer2._start_time == 12345.0


def test_set_state_missing_key():

    reducer = BudgetReducer(max_permutations=100)

    for key in ('start_time', 'trimmed'):
        state = reducer.get_state()
        del state[key]
        try:
            reducer.set_state(state)
            assert False, f"Should have raised ValueError for missing {key!r}"
        except ValueError as e:
            assert key in str(e)


def test_remaining_none_with_permutation_budget():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    for _ in range(4):
        next(msq)

    reducer = BudgetReducer(
        max_permutations=5,
        check_after_pct=0.0,
    )
    df = pl.DataFrame({
        'a': [1, 2, 3, 1],
        'b': ['x', 'x', 'x', 'y'],
        'score': [0.5, 0.6, 0.7, 0.8],
    })

    original_remaining = msq.remaining_count
    msq.remaining_count = lambda: None

    result = reducer.analyze_and_intervene(df, msq)
    assert len(result) == 1
    assert result[0]['op'] == 'trim'

    msq.remaining_count = original_remaining


def test_walltime_zero_elapsed():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    for _ in range(3):
        next(msq)

    reducer = BudgetReducer(
        start_time=time.time(),
        max_walltime_hours=1.0,
        check_after_pct=0.0,
    )
    df = pl.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'x', 'x'],
        'score': [0.5, 0.6, 0.7],
    })

    result = reducer.analyze_and_intervene(df, msq)
    assert result == []


def test_worst_first_includes_trim_fallback():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    for _ in range(4):
        next(msq)

    reducer = BudgetReducer(
        max_permutations=5,
        trim_strategy='worst_first',
        metric='score',
        maximize=True,
        check_after_pct=0.0,
    )

    df = pl.DataFrame({
        'a': [1, 1, 2, 2, 3, 3],
        'b': ['x', 'y', 'x', 'y', 'x', 'y'],
        'score': [0.1, 0.2, 0.5, 0.6, 0.9, 0.95],
    })

    result = reducer.analyze_and_intervene(df, msq)
    remove_ops = [r for r in result if r['op'] == 'remove_is']
    trim_ops = [r for r in result if r['op'] == 'trim']
    assert len(remove_ops) >= 1
    assert len(trim_ops) == 1


def test_worst_first_maximize_false():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})
    for _ in range(4):
        next(msq)

    reducer = BudgetReducer(
        max_permutations=5,
        trim_strategy='worst_first',
        metric='loss',
        maximize=False,
        check_after_pct=0.0,
    )

    df = pl.DataFrame({
        'a': [1, 1, 2, 2, 3, 3],
        'b': ['x', 'y', 'x', 'y', 'x', 'y'],
        'loss': [0.1, 0.2, 0.5, 0.6, 0.9, 0.95],
    })

    result = reducer.analyze_and_intervene(df, msq)
    remove_ops = [r for r in result if r['op'] == 'remove_is']
    assert len(remove_ops) >= 1
