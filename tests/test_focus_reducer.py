import polars as pl

from limen.experiment.reducer.filter_types import FILTER_KEEP_BETWEEN
from limen.experiment.reducer.filter_types import FILTER_KEEP_VALUES
from limen.experiment.reducer.focus_reducer import FocusReducer
from tests.stubs.stubs import make_msq


def test_breakthrough_activates_focus():

    n = 20
    df = pl.DataFrame({
        'a': [1] * n + [2] * n,
        'b': ['x'] * n + ['y'] * n,
        'score': [0.5] * (n * 2 - 1) + [0.95],
    })
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        focus_range_pct=0.2,
        min_observations=5,
    )
    result = reducer.analyze_and_intervene(df, msq)

    set_filters = [r for r in result if r['op'] == 'set_filter']
    inject_values = [r for r in result if r['op'] == 'inject_value']

    assert len(set_filters) == 2
    assert len(inject_values) >= 1
    assert reducer._focus_active is True
    assert reducer._best_metric == 0.95


def test_no_breakthrough():

    n = 20
    df = pl.DataFrame({
        'a': [1] * n,
        'b': ['x'] * n,
        'score': [0.5] * n,
    })
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        min_observations=5,
    )
    result = reducer.analyze_and_intervene(df, msq)

    assert result == []
    assert reducer._focus_active is False


def test_maximize_false():

    n = 20
    df = pl.DataFrame({
        'a': [1] * n,
        'b': ['x'] * n,
        'loss': [0.5] * (n - 1) + [0.01],
    })
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='loss',
        breakthrough_threshold=0.05,
        maximize=False,
        min_observations=5,
    )
    result = reducer.analyze_and_intervene(df, msq)

    assert len(result) > 0
    assert reducer._focus_active is True
    assert reducer._best_metric == 0.01


def test_early_returns():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    inactive = FocusReducer(
        metric='score', breakthrough_threshold=0.9, active=False,
    )
    assert inactive.analyze_and_intervene(pl.DataFrame(), msq) == []

    reducer = FocusReducer(
        metric='score', breakthrough_threshold=0.9, min_observations=10,
    )

    df_empty = pl.DataFrame(
        {'a': [], 'b': [], 'score': []},
        schema={'a': pl.Int64, 'b': pl.Utf8, 'score': pl.Float64},
    )
    assert reducer.analyze_and_intervene(df_empty, msq) == []

    df_no_metric = pl.DataFrame({'a': [1] * 20, 'b': ['x'] * 20, 'other': [1.0] * 20})
    assert reducer.analyze_and_intervene(df_no_metric, msq) == []

    df_few = pl.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'x'], 'score': [0.95, 0.95, 0.95]})
    assert reducer.analyze_and_intervene(df_few, msq) == []


def test_snap_back_after_timeout():

    n = 20
    df_breakthrough = pl.DataFrame({
        'a': [1] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.95],
    })
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        focus_timeout=3,
        min_observations=5,
    )

    result1 = reducer.analyze_and_intervene(df_breakthrough, msq)
    assert reducer._focus_active is True
    set_filter_keys = {r['key'] for r in result1 if r['op'] == 'set_filter'}

    df_no_improvement = df_breakthrough
    reducer.analyze_and_intervene(df_no_improvement, msq)
    assert reducer._rounds_since_improvement == 1

    reducer.analyze_and_intervene(df_no_improvement, msq)
    assert reducer._rounds_since_improvement == 2

    result_snap = reducer.analyze_and_intervene(df_no_improvement, msq)
    assert reducer._focus_active is False

    clear_keys = {r['key'] for r in result_snap if r['op'] == 'clear_filter'}
    assert clear_keys == set_filter_keys


def test_improvement_resets_timeout():

    n = 20
    df1 = pl.DataFrame({
        'a': [1] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.95],
    })
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        focus_timeout=3,
        min_observations=5,
    )

    reducer.analyze_and_intervene(df1, msq)
    assert reducer._focus_active is True

    reducer.analyze_and_intervene(df1, msq)
    assert reducer._rounds_since_improvement == 1

    # Improvement resets counter
    df2 = pl.DataFrame({
        'a': [1] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.98],
    })
    result = reducer.analyze_and_intervene(df2, msq)
    assert reducer._rounds_since_improvement == 0
    assert reducer._best_metric == 0.98

    # set_filter re-emitted on improvement
    set_filters = [r for r in result if r['op'] == 'set_filter']
    assert len(set_filters) >= 1


def test_numeric_narrowing_range():

    n = 20
    df = pl.DataFrame({
        'a': [10] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.95],
    })
    msq, _, _ = make_msq(params={'a': [5, 10, 15], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        focus_range_pct=0.2,
        min_observations=5,
    )
    result = reducer.analyze_and_intervene(df, msq)

    a_filter = [r for r in result if r['op'] == 'set_filter' and r['filter_params'].get('param') == 'a']
    assert len(a_filter) == 1
    assert a_filter[0]['filter_type'] == FILTER_KEEP_BETWEEN
    assert a_filter[0]['filter_params']['lower'] == 8
    assert a_filter[0]['filter_params']['upper'] == 12


def test_categorical_narrowing():

    n = 20
    df = pl.DataFrame({
        'a': [1] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.95],
    })
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        min_observations=5,
    )
    result = reducer.analyze_and_intervene(df, msq)

    b_filter = [r for r in result if r['op'] == 'set_filter' and r['filter_params'].get('param') == 'b']
    assert len(b_filter) == 1
    assert b_filter[0]['filter_type'] == FILTER_KEEP_VALUES
    assert b_filter[0]['filter_params']['values'] == ['x']


def test_zero_center_narrowing():

    n = 20
    df = pl.DataFrame({
        'a': [0] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.95],
    })
    msq, _, _ = make_msq(params={'a': [-1, 0, 1], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        focus_range_pct=0.2,
        min_observations=5,
    )
    result = reducer.analyze_and_intervene(df, msq)

    a_filter = [r for r in result if r['op'] == 'set_filter' and r['filter_params'].get('param') == 'a']
    assert len(a_filter) == 1
    # center=0 (int): half_width=0.2, floor(-0.2)=-1, ceil(0.2)=1
    assert a_filter[0]['filter_params']['lower'] == -1
    assert a_filter[0]['filter_params']['upper'] == 1


def test_variation_injection_count():

    n = 20
    df = pl.DataFrame({
        'a': [10.0] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.95],
    })
    msq, _, _ = make_msq(params={'a': [5.0, 10.0, 15.0], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        variation_count=7,
        min_observations=5,
    )
    result = reducer.analyze_and_intervene(df, msq)

    # Only numeric param 'a' gets injections, not categorical 'b'
    inject_a = [r for r in result if r['op'] == 'inject_value' and r['param'] == 'a']
    inject_b = [r for r in result if r['op'] == 'inject_value' and r['param'] == 'b']
    assert len(inject_a) == 7
    assert len(inject_b) == 0


def test_variation_injection_linspace():

    n = 20
    df = pl.DataFrame({
        'a': [10.0] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.95],
    })
    msq, _, _ = make_msq(params={'a': [5.0, 10.0, 15.0], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        focus_range_pct=0.2,
        variation_count=5,
        min_observations=5,
    )
    result = reducer.analyze_and_intervene(df, msq)

    inject_values = sorted(r['value'] for r in result if r['op'] == 'inject_value')
    assert len(inject_values) == 5
    assert abs(inject_values[0] - 8.0) < 1e-9
    assert abs(inject_values[-1] - 12.0) < 1e-9

    # Check even spacing
    for i in range(1, len(inject_values)):
        gap = inject_values[i] - inject_values[i - 1]
        assert abs(gap - 1.0) < 1e-9


def test_no_injection_on_improvement():

    n = 20
    df1 = pl.DataFrame({
        'a': [10] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.95],
    })
    msq, _, _ = make_msq(params={'a': [5, 10, 15], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        variation_count=5,
        min_observations=5,
    )

    result1 = reducer.analyze_and_intervene(df1, msq)
    inject1 = [r for r in result1 if r['op'] == 'inject_value']
    assert len(inject1) > 0

    # Improvement: only filters, no new injections
    df2 = pl.DataFrame({
        'a': [10] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.99],
    })
    result2 = reducer.analyze_and_intervene(df2, msq)
    inject2 = [r for r in result2 if r['op'] == 'inject_value']
    assert len(inject2) == 0

    set_filters2 = [r for r in result2 if r['op'] == 'set_filter']
    assert len(set_filters2) >= 1


def test_mixed_param_types():

    n = 20
    df = pl.DataFrame({
        'a': [10] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.95],
    })
    msq, _, _ = make_msq(params={'a': [5, 10, 15], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        min_observations=5,
    )
    result = reducer.analyze_and_intervene(df, msq)

    a_filter = [r for r in result if r['op'] == 'set_filter' and r['filter_params'].get('param') == 'a']
    b_filter = [r for r in result if r['op'] == 'set_filter' and r['filter_params'].get('param') == 'b']

    assert a_filter[0]['filter_type'] == FILTER_KEEP_BETWEEN
    assert b_filter[0]['filter_type'] == FILTER_KEEP_VALUES

    inject_params = {r['param'] for r in result if r['op'] == 'inject_value'}
    assert 'a' in inject_params
    assert 'b' not in inject_params


def test_improvement_updates_combo():

    n = 20
    df1 = pl.DataFrame({
        'a': [10] * n,
        'b': ['x'] * n,
        'score': [0.5] * (n - 1) + [0.95],
    })
    msq, _, _ = make_msq(params={'a': [5, 10, 15], 'b': ['x', 'y']})

    reducer = FocusReducer(
        metric='score',
        breakthrough_threshold=0.9,
        focus_range_pct=0.2,
        min_observations=5,
    )
    reducer.analyze_and_intervene(df1, msq)
    assert reducer._breakthrough_combo['a'] == 10

    df2 = pl.DataFrame({
        'a': [10] * n + [15] * n,
        'b': ['x'] * n + ['y'] * n,
        'score': [0.5] * (n * 2 - 1) + [0.99],
    })
    result2 = reducer.analyze_and_intervene(df2, msq)

    assert reducer._breakthrough_combo['a'] == 15
    assert reducer._best_metric == 0.99

    a_filter = [r for r in result2 if r['op'] == 'set_filter' and r['filter_params'].get('param') == 'a']
    assert a_filter[0]['filter_params']['lower'] == 12
    assert a_filter[0]['filter_params']['upper'] == 18


def test_state_roundtrip():

    reducer = FocusReducer(metric='score', breakthrough_threshold=0.9)
    reducer._focus_active = True
    reducer._breakthrough_combo = {'a': 10, 'b': 'x'}
    reducer._best_metric = 0.95
    reducer._rounds_since_improvement = 2
    reducer._focused_params = {'focus_score_a', 'focus_score_b'}

    state = reducer.get_state()
    assert state['focus_active'] is True
    assert state['best_metric'] == 0.95

    reducer2 = FocusReducer(metric='score', breakthrough_threshold=0.9)
    reducer2.set_state(state)

    assert reducer2._focus_active is True
    assert reducer2._breakthrough_combo == {'a': 10, 'b': 'x'}
    assert reducer2._best_metric == 0.95
    assert reducer2._rounds_since_improvement == 2
    assert reducer2._focused_params == {'focus_score_a', 'focus_score_b'}


def test_set_state_missing_key():

    reducer = FocusReducer(metric='score', breakthrough_threshold=0.9)

    for key in ('focus_active', 'breakthrough_combo', 'best_metric',
                'rounds_since_improvement', 'focused_params'):
        state = reducer.get_state()
        del state[key]
        try:
            reducer.set_state(state)
            assert False, f"Should have raised ValueError for missing {key!r}"
        except ValueError as e:
            assert key in str(e)
