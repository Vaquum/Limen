import pandas as pd

from limen.experiment.reducer.sanity_reducer import SanityReducer
from tests.stubs.stubs import make_msq


class FakeLog:

    def __init__(self, df):
        self.experiment_log = df


def test_nan_above_threshold():

    df = pd.DataFrame({
        'a': [1, 1, 1, 2, 2, 2],
        'b': ['x', 'x', 'y', 'x', 'x', 'y'],
        'score': [float('nan'), float('nan'), float('nan'), 3.0, 4.0, 5.0],
    })
    log = FakeLog(df)
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', nan_threshold=0.5)
    result = reducer.analyze_and_intervene(log, msq)

    assert len(result) == 1
    assert result[0] == {'op': 'remove_is', 'param': 'a', 'value': 1}


def test_nan_below_threshold():

    df = pd.DataFrame({
        'a': [1, 1, 1, 1, 1],
        'b': ['x', 'x', 'x', 'y', 'y'],
        'score': [1.0, float('nan'), 3.0, 4.0, 5.0],
    })
    log = FakeLog(df)
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', nan_threshold=0.5)
    result = reducer.analyze_and_intervene(log, msq)

    assert result == []


def test_min_observations_gate():

    df = pd.DataFrame({
        'a': [1],
        'b': ['x'],
        'score': [float('nan')],
    })
    log = FakeLog(df)
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', nan_threshold=0.1, min_observations=5)
    result = reducer.analyze_and_intervene(log, msq)

    assert result == []


def test_empty_log():

    df = pd.DataFrame(columns=['a', 'b', 'score'])
    log = FakeLog(df)
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score')
    result = reducer.analyze_and_intervene(log, msq)

    assert result == []


def test_missing_metric_column():

    df = pd.DataFrame({
        'a': [1, 2],
        'b': ['x', 'y'],
        'other': [1.0, 2.0],
    })
    log = FakeLog(df)
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score')
    result = reducer.analyze_and_intervene(log, msq)

    assert result == []


def test_inactive_returns_empty():

    df = pd.DataFrame({
        'a': [1, 1],
        'b': ['x', 'x'],
        'score': [float('nan'), float('nan')],
    })
    log = FakeLog(df)
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', active=False)
    result = reducer.analyze_and_intervene(log, msq)

    assert result == []


def test_multiple_params_pruned():

    df = pd.DataFrame({
        'a': [1, 1, 2, 2],
        'b': ['x', 'x', 'y', 'y'],
        'score': [float('nan'), float('nan'), float('nan'), float('nan')],
    })
    log = FakeLog(df)
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', nan_threshold=0.5)
    result = reducer.analyze_and_intervene(log, msq)

    pruned = {(r['param'], r['value']) for r in result}
    assert ('a', 1) in pruned
    assert ('a', 2) in pruned
    assert ('b', 'x') in pruned
    assert ('b', 'y') in pruned


def test_dedup_across_triggers():

    df = pd.DataFrame({
        'a': [1, 1],
        'b': ['x', 'x'],
        'score': [float('nan'), float('nan')],
    })
    log = FakeLog(df)
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', nan_threshold=0.1)
    result1 = reducer.analyze_and_intervene(log, msq)
    result2 = reducer.analyze_and_intervene(log, msq)

    assert len(result1) > 0
    assert result2 == []


def test_checkpoint_roundtrip():

    reducer = SanityReducer(metric='score')
    reducer._removed = {('a', 1), ('b', 'x')}

    state = reducer.get_state()
    assert 'removed' in state

    reducer2 = SanityReducer(metric='score')
    reducer2.set_state(state)

    assert reducer2._removed == {('a', 1), ('b', 'x')}


def test_boundary_exact_threshold():

    # 1 NaN out of 5 = 0.2 rate, threshold = 0.2 → NOT pruned (strict >)
    df = pd.DataFrame({
        'a': [1, 1, 1, 1, 1],
        'b': ['x', 'x', 'x', 'x', 'x'],
        'score': [float('nan'), 2.0, 3.0, 4.0, 5.0],
    })
    log = FakeLog(df)
    msq, _, _ = make_msq(params={'a': [1, 2, 3], 'b': ['x', 'y']})

    reducer = SanityReducer(metric='score', nan_threshold=0.2)
    result = reducer.analyze_and_intervene(log, msq)

    assert result == []
