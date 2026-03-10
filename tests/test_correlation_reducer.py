import random

import polars as pl

from limen.experiment.reducer.correlation_reducer import CorrelationReducer
from tests.stubs.stubs import make_msq


def _make_correlated_df(n: int = 60) -> pl.DataFrame:

    '''Create a DataFrame where a=0 strongly outperforms a=1.'''

    scores_a0 = [8.0 + (i % 5) * 0.1 for i in range(n // 2)]
    scores_a1 = [2.0 + (i % 5) * 0.1 for i in range(n // 2)]

    return pl.DataFrame({
        'a': [0] * (n // 2) + [1] * (n // 2),
        'b': [10, 20] * (n // 2),
        'score': scores_a0 + scores_a1,
    })


def test_wrong_direction_removal():

    df = _make_correlated_df()
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    reducer = CorrelationReducer(
        metric='score',
        min_observations=10,
        n_boot=50,
        negative_correlation_threshold=-0.2,
    )
    result = reducer.analyze_and_intervene(df, msq)

    removals = [r for r in result if r.get('action') is None]
    assert len(removals) >= 1
    a_removal = [r for r in removals if r['param'] == 'a']
    assert len(a_removal) == 1
    assert a_removal[0]['value'] == 1
    assert 'wrong-direction' in a_removal[0]['reason']


def test_low_impact_suggestion():

    n = 60
    df = pl.DataFrame({
        'a': [0, 1] * (n // 2),
        'b': [10] * (n // 2) + [20] * (n // 2),
        'score': [5.0 + (i % 3) * 0.01 for i in range(n)],
    })
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    reducer = CorrelationReducer(
        metric='score',
        min_observations=10,
        n_boot=50,
        prune_threshold=0.1,
        sign_stability_threshold=0.5,
    )
    result = reducer.analyze_and_intervene(df, msq)

    suggestions = [r for r in result if r.get('action') == 'suggest']
    assert len(suggestions) >= 1
    assert suggestions[0]['op'] == 'keep_is'
    assert 'low-impact' in suggestions[0]['reason']


def test_dedup_across_triggers():

    df = _make_correlated_df()
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    reducer = CorrelationReducer(
        metric='score',
        min_observations=10,
        n_boot=50,
        negative_correlation_threshold=-0.2,
    )
    result1 = reducer.analyze_and_intervene(df, msq)
    result2 = reducer.analyze_and_intervene(df, msq)

    assert len(result1) > 0
    a_removals_1 = [r for r in result1 if r['param'] == 'a' and r.get('action') is None]
    a_removals_2 = [r for r in result2 if r['param'] == 'a' and r.get('action') is None]
    assert len(a_removals_1) == 1
    assert len(a_removals_2) == 0


def test_early_returns():

    df = _make_correlated_df()
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    inactive = CorrelationReducer(metric='score', active=False, min_observations=10, n_boot=50)
    assert inactive.analyze_and_intervene(df, msq) == []

    reducer = CorrelationReducer(metric='score', min_observations=10, n_boot=50)

    df_empty = pl.DataFrame(
        {'a': [], 'b': [], 'score': []},
        schema={'a': pl.Int64, 'b': pl.Int64, 'score': pl.Float64},
    )
    assert reducer.analyze_and_intervene(df_empty, msq) == []

    df_no_metric = pl.DataFrame({'a': [0, 1], 'b': [10, 20], 'other': [1.0, 2.0]})
    assert reducer.analyze_and_intervene(df_no_metric, msq) == []

    df_few = pl.DataFrame({
        'a': [0, 1, 0, 1, 0],
        'b': [10, 20, 10, 20, 10],
        'score': [8.0, 2.0, 7.0, 3.0, 9.0],
    })
    few_reducer = CorrelationReducer(metric='score', min_observations=100, n_boot=50)
    assert few_reducer.analyze_and_intervene(df_few, msq) == []


def test_insufficient_data_for_correlation():

    df = pl.DataFrame({
        'a': [0] * 20,
        'b': [10] * 20,
        'score': [5.0] * 20,
    })
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    reducer = CorrelationReducer(
        metric='score',
        min_observations=10,
        n_boot=50,
    )
    assert reducer.analyze_and_intervene(df, msq) == []


def test_state_roundtrip():

    reducer = CorrelationReducer(metric='score')
    reducer._applied = {('a', 1)}
    reducer._suggested = {'b'}

    state = reducer.get_state()
    assert 'applied' in state
    assert 'suggested' in state

    reducer2 = CorrelationReducer(metric='score')
    reducer2.set_state(state)

    assert reducer2._applied == {('a', 1)}
    assert reducer2._suggested == {'b'}


def test_set_state_missing_key():

    reducer = CorrelationReducer(metric='score')

    try:
        reducer.set_state({})
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'applied' in str(e)


def test_maximize_false():

    n = 60
    scores_a0 = [2.0 + (i % 5) * 0.1 for i in range(n // 2)]
    scores_a1 = [8.0 + (i % 5) * 0.1 for i in range(n // 2)]

    df = pl.DataFrame({
        'a': [0] * (n // 2) + [1] * (n // 2),
        'b': [10, 20] * (n // 2),
        'score': scores_a0 + scores_a1,
    })
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    reducer = CorrelationReducer(
        metric='score',
        min_observations=10,
        n_boot=50,
        negative_correlation_threshold=-0.2,
        maximize=False,
    )
    result = reducer.analyze_and_intervene(df, msq)

    removals = [r for r in result if r.get('action') is None and r['param'] == 'a']
    assert len(removals) == 1
    assert removals[0]['value'] == 1


def test_sign_stability_below_threshold():

    n = 60
    random.seed(42)
    scores = [random.gauss(5.0, 2.0) for _ in range(n)]

    df = pl.DataFrame({
        'a': [0, 1] * (n // 2),
        'b': [10, 20] * (n // 2),
        'score': scores,
    })
    msq, _, _ = make_msq(params={'a': [0, 1], 'b': [10, 20]})

    reducer = CorrelationReducer(
        metric='score',
        min_observations=10,
        n_boot=50,
        sign_stability_threshold=0.99,
        prune_threshold=0.5,
        negative_correlation_threshold=-0.01,
    )
    result = reducer.analyze_and_intervene(df, msq)

    assert result == []
