import polars as pl

from limen.experiment.reducer.filter_types import FILTER_SAMPLE
from limen.experiment.reducer.saturation_reducer import SaturationReducer
from tests.stubs.stubs import make_msq


def test_saturated_value_filter():

    n = 40
    df = pl.DataFrame({
        'a': [0] * n + [1] * n,
        'b': [10, 20] * n,
        'score': [5.0] * n + [3.0 + (i % 5) * 0.5 for i in range(n)],
    })
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    reducer = SaturationReducer(
        metric='score',
        cv_threshold=0.01,
        min_samples_per_value=10,
        retain_fraction=0.1,
    )
    result = reducer.analyze_and_intervene(df, msq)

    set_filters = [r for r in result if r['op'] == 'set_filter']
    assert len(set_filters) >= 1

    a0_filter = [r for r in set_filters if r['filter_params']['param'] == 'a' and r['filter_params']['value'] == 0]
    assert len(a0_filter) == 1
    assert a0_filter[0]['filter_type'] == FILTER_SAMPLE
    assert a0_filter[0]['filter_params']['fraction'] == 0.1
    assert 'saturated' in a0_filter[0]['reason']


def test_no_saturation():

    n = 40
    df = pl.DataFrame({
        'a': [0] * n + [1] * n,
        'b': [10, 20] * n,
        'score': [float(i) for i in range(n)] + [float(i * 2) for i in range(n)],
    })
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    reducer = SaturationReducer(
        metric='score',
        cv_threshold=0.01,
        min_samples_per_value=10,
    )
    result = reducer.analyze_and_intervene(df, msq)

    assert result == []


def test_early_returns():

    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})
    reducer = SaturationReducer(metric='score', min_samples_per_value=10)

    inactive = SaturationReducer(metric='score', active=False)
    assert inactive.analyze_and_intervene(pl.DataFrame(), msq) == []

    df_empty = pl.DataFrame(
        {'a': [], 'b': [], 'score': []},
        schema={'a': pl.Int64, 'b': pl.Int64, 'score': pl.Float64},
    )
    assert reducer.analyze_and_intervene(df_empty, msq) == []

    df_no_metric = pl.DataFrame({'a': [0, 1], 'b': [10, 20], 'other': [1.0, 2.0]})
    assert reducer.analyze_and_intervene(df_no_metric, msq) == []

    df_few = pl.DataFrame({
        'a': [0, 0, 0],
        'b': [10, 10, 10],
        'score': [5.0, 5.0, 5.0],
    })
    assert reducer.analyze_and_intervene(df_few, msq) == []


def test_window_size():

    old_scores = [float(i) for i in range(50)]
    new_scores = [5.0] * 30
    n = len(old_scores) + len(new_scores)

    df = pl.DataFrame({
        'a': [0] * n,
        'b': [10] * n,
        'score': old_scores + new_scores,
    })
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    reducer = SaturationReducer(
        metric='score',
        cv_threshold=0.01,
        window_size=30,
        min_samples_per_value=10,
    )
    result = reducer.analyze_and_intervene(df, msq)

    set_filters = [r for r in result if r['op'] == 'set_filter']
    assert len(set_filters) >= 1


def test_zero_mean_skipped():

    n = 40
    df = pl.DataFrame({
        'a': [0] * n,
        'b': [10] * n,
        'score': [0.0] * n,
    })
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    reducer = SaturationReducer(
        metric='score',
        cv_threshold=0.01,
        min_samples_per_value=10,
    )
    result = reducer.analyze_and_intervene(df, msq)

    assert result == []


def test_unsaturation_clears_filter():

    n = 40
    df_saturated = pl.DataFrame({
        'a': [0] * n,
        'b': [10] * n,
        'score': [5.0] * n,
    })
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    reducer = SaturationReducer(
        metric='score',
        cv_threshold=0.01,
        min_samples_per_value=10,
    )

    result1 = reducer.analyze_and_intervene(df_saturated, msq)
    set_filters = [r for r in result1 if r['op'] == 'set_filter']
    assert len(set_filters) >= 1

    df_volatile = pl.DataFrame({
        'a': [0] * n,
        'b': [10] * n,
        'score': [float(i) for i in range(n)],
    })

    result2 = reducer.analyze_and_intervene(df_volatile, msq)
    clear_filters = [r for r in result2 if r['op'] == 'clear_filter']
    assert len(clear_filters) >= 1


def test_state_roundtrip():

    reducer = SaturationReducer(metric='score')
    reducer._saturated = {('a', 0), ('b', 10)}

    state = reducer.get_state()
    assert 'saturated' in state

    reducer2 = SaturationReducer(metric='score')
    reducer2.set_state(state)

    assert reducer2._saturated == {('a', 0), ('b', 10)}


def test_set_state_missing_key():

    reducer = SaturationReducer(metric='score')

    try:
        reducer.set_state({})
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'saturated' in str(e)
