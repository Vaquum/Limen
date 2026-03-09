import polars as pl

from limen.experiment.reducer.sanity_reducer import SanityReducer
from tests.stubs.stubs import make_msq


def test_nan_above_threshold():

    df = pl.DataFrame({
        'a': [1, 1, 1, 2, 2, 2],
        'b': ['x', 'x', 'y', 'x', 'x', 'y'],
        'score': [float('nan'), float('nan'), float('nan'), 3.0, 4.0, 5.0],
    })
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', nan_threshold=0.5)
    result = reducer.analyze_and_intervene(df, msq)

    assert len(result) == 1
    assert result[0] == {'op': 'remove_is', 'param': 'a', 'value': 1}


def test_min_observations_gate():

    df = pl.DataFrame({
        'a': [1],
        'b': ['x'],
        'score': [float('nan')],
    })
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', nan_threshold=0.1, min_observations=5)
    result = reducer.analyze_and_intervene(df, msq)

    assert result == []


def test_early_returns():

    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    # Empty log
    df_empty = pl.DataFrame({'a': [], 'b': [], 'score': []},
                            schema={'a': pl.Int64, 'b': pl.Utf8, 'score': pl.Float64})
    assert SanityReducer(metric='score').analyze_and_intervene(df_empty, msq) == []

    # Missing metric column
    df_no_metric = pl.DataFrame({'a': [1, 2], 'b': ['x', 'y'], 'other': [1.0, 2.0]})
    assert SanityReducer(metric='score').analyze_and_intervene(df_no_metric, msq) == []

    # Inactive reducer
    df_nan = pl.DataFrame({'a': [1, 1], 'b': ['x', 'x'], 'score': [float('nan'), float('nan')]})
    assert SanityReducer(metric='score', active=False).analyze_and_intervene(df_nan, msq) == []


def test_multiple_params_pruned():

    df = pl.DataFrame({
        'a': [1, 1, 2, 2],
        'b': ['x', 'x', 'y', 'y'],
        'score': [float('nan'), float('nan'), float('nan'), float('nan')],
    })
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', nan_threshold=0.5)
    result = reducer.analyze_and_intervene(df, msq)

    pruned = {(r['param'], r['value']) for r in result}
    assert ('a', 1) in pruned
    assert ('a', 2) in pruned
    assert ('b', 'x') in pruned
    assert ('b', 'y') in pruned


def test_dedup_across_triggers():

    df = pl.DataFrame({
        'a': [1, 1],
        'b': ['x', 'x'],
        'score': [float('nan'), float('nan')],
    })
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', nan_threshold=0.1)
    result1 = reducer.analyze_and_intervene(df, msq)
    result2 = reducer.analyze_and_intervene(df, msq)

    assert len(result1) > 0
    assert result2 == []


def test_boundary_exact_threshold():

    # 1 NaN out of 5 = 0.2 rate, threshold = 0.2 → NOT pruned (strict >)
    df = pl.DataFrame({
        'a': [1, 1, 1, 1, 1],
        'b': ['x', 'x', 'x', 'x', 'x'],
        'score': [float('nan'), 2.0, 3.0, 4.0, 5.0],
    })
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', nan_threshold=0.2)
    result = reducer.analyze_and_intervene(df, msq)

    assert result == []


def test_zero_metric_suggestion():

    df = pl.DataFrame({
        'a': [1, 1, 1, 2, 2, 2],
        'b': ['x', 'x', 'x', 'y', 'y', 'y'],
        'score': [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
    })
    msq, _, _ = make_msq(params={'a': [1, 2], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', zero_metric_threshold=0.5)
    result = reducer.analyze_and_intervene(df, msq)

    suggestions = [r for r in result if r.get('action') == 'suggest']
    assert len(suggestions) >= 1
    assert suggestions[0]['param'] == 'a'
    assert suggestions[0]['value'] == 1
    assert 'zero_metric' in suggestions[0]['reason']


def test_zero_metric_disabled_by_default():

    df = pl.DataFrame({
        'a': [1, 1, 1],
        'b': ['x', 'x', 'x'],
        'score': [0.0, 0.0, 0.0],
    })
    msq, _, _ = make_msq(params={'a': [1, 2], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score')
    result = reducer.analyze_and_intervene(df, msq)

    suggestions = [r for r in result if r.get('action') == 'suggest']
    assert suggestions == []


def test_execution_timeout_suggestion():

    df = pl.DataFrame({
        'a': [1, 1, 1, 2, 2, 2],
        'b': ['x', 'x', 'x', 'y', 'y', 'y'],
        'score': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'execution_time': [100.0, 200.0, 150.0, 1.0, 2.0, 3.0],
    })
    msq, _, _ = make_msq(params={'a': [1, 2], 'b': ['x', 'y']})

    reducer = SanityReducer(
        metric='score',
        execution_time_threshold=50.0,
    )
    result = reducer.analyze_and_intervene(df, msq)

    suggestions = [r for r in result if r.get('action') == 'suggest']
    assert len(suggestions) >= 1
    timeout_a = [a for a in suggestions if a['param'] == 'a' and a['value'] == 1]
    assert len(timeout_a) == 1
    assert 'execution_timeout' in timeout_a[0]['reason']


def test_warning_suggestion():

    df = pl.DataFrame({
        'a': [1, 1, 1, 2, 2, 2],
        'b': ['x', 'x', 'x', 'y', 'y', 'y'],
        'score': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        '_warnings': ['["RuntimeWarning"]', '["ConvergenceWarning"]', '["FutureWarning"]', '[]', '[]', '[]'],
    })
    msq, _, _ = make_msq(params={'a': [1, 2], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', warning_threshold=0.5)
    result = reducer.analyze_and_intervene(df, msq)

    suggestions = [r for r in result if r.get('action') == 'suggest']
    assert len(suggestions) >= 1
    warn_a = [a for a in suggestions if a['param'] == 'a' and a['value'] == 1]
    assert len(warn_a) == 1
    assert 'warning rate' in warn_a[0]['reason']


def test_warning_no_column():

    df = pl.DataFrame({
        'a': [1, 1],
        'b': ['x', 'x'],
        'score': [1.0, 2.0],
    })
    msq, _, _ = make_msq(params={'a': [1, 2], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', warning_threshold=0.1)
    result = reducer.analyze_and_intervene(df, msq)

    suggestions = [r for r in result if r.get('action') == 'suggest']
    assert suggestions == []


def test_suggestion_dedup():

    df = pl.DataFrame({
        'a': [1, 1, 1],
        'b': ['x', 'x', 'x'],
        'score': [0.0, 0.0, 0.0],
    })
    msq, _, _ = make_msq(params={'a': [1, 2], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', zero_metric_threshold=0.5)
    result1 = reducer.analyze_and_intervene(df, msq)
    result2 = reducer.analyze_and_intervene(df, msq)

    suggestions1 = [r for r in result1 if r.get('action') == 'suggest']
    suggestions2 = [r for r in result2 if r.get('action') == 'suggest']
    assert len(suggestions1) > 0
    assert suggestions2 == []


def test_suggestion_checkpoint_roundtrip():

    reducer = SanityReducer(metric='score')
    reducer._removed = {('a', 1)}
    reducer._suggested = {('b', 'x', 'zero_metric')}

    state = reducer.get_state()
    assert 'removed' in state
    assert 'suggested' in state

    reducer2 = SanityReducer(metric='score')
    reducer2.set_state(state)

    assert reducer2._removed == {('a', 1)}
    assert reducer2._suggested == {('b', 'x', 'zero_metric')}


