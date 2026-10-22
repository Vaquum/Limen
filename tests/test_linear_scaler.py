from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars.testing import assert_series_equal

from limen.indicators import avgprice
from limen.indicators import roc
from limen.scalers.linear_scaler import DEFAULT_SCALING_RULES
from limen.scalers.linear_scaler import LinearScaler
from limen.scalers.linear_scaler import build_rules
from limen.scalers.linear_scaler import get_scaling_rule
from limen.scalers.linear_scaler import inverse_transform


@pytest.fixture(scope='module')
def recorded():
    path = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'
    raw = pl.read_parquet(path, n_rows=512)
    return avgprice(roc(roc(raw, period=1), period=2))


def test_explicit_overrides_select_forward_scaling(recorded):
    train = recorded.select('roc_1', 'roc_2', 'volume', 'avgprice')
    defaults = list(DEFAULT_SCALING_RULES.items())
    rules = build_rules({'none': ['roc_1', 'volume'], 'standard': ['avgprice']})
    assert list(DEFAULT_SCALING_RULES.items()) == defaults
    assert rules is not DEFAULT_SCALING_RULES
    scaler = LinearScaler(train, rules=rules)
    actual = scaler.transform(train)
    restored = inverse_transform(actual, scaler)
    assert train['roc_1'][1] == pytest.approx(0.11288096524970648)
    for col in ('roc_1', 'volume'):
        assert get_scaling_rule(col, rules) == 'none'
        assert col not in scaler.means and col not in scaler.stds
        assert_series_equal(actual[col], train[col], check_exact=True)
        assert_series_equal(restored[col], train[col], check_exact=True)
    for col in ('avgprice', 'roc_2'):
        assert get_scaling_rule(col, rules) == 'standard'
        expected = (train[col] - train[col].mean()) / train[col].std(ddof=0)
        assert_series_equal(actual[col], expected, rel_tol=1e-12, abs_tol=1e-12)
    assert_frame_equal(restored, train, rel_tol=1e-12, abs_tol=1e-12)


def test_existing_key_moves_before_generic_without_mutation(recorded):
    base = {r'^roc': 'standard', r'^roc_1$': 'none', r'.*': 'none'}
    overrides = {'divide_100': ['roc_1'], 'none': ['volume']}
    before_base, before_overrides = dict(base), deepcopy(overrides)
    rules = build_rules(overrides, base_rules=base)
    assert list(rules.items()) == [
        (r'^roc_1$', 'divide_100'), (r'^volume$', 'none'),
        (r'^roc', 'standard'), (r'.*', 'none'),
    ]
    assert list(base.items()) == list(before_base.items())
    assert overrides == before_overrides
    assert rules is not base and rules is not overrides
    assert get_scaling_rule('roc_1', rules) == 'divide_100'
    assert get_scaling_rule('roc_2', rules) == 'standard'
    train = recorded.select('roc_1')
    scaler = LinearScaler(train, rules=rules)
    actual = scaler.transform(train)
    assert scaler.means == {} and scaler.stds == {}
    assert_series_equal(actual['roc_1'], train['roc_1'] / 100, check_exact=True)
    assert_frame_equal(inverse_transform(actual, scaler), train, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize('entry, expected', [
    ('roc', ['none', 'standard']),
    (r'roc(_\d+)?', ['none', 'none']),
])
def test_override_entries_keep_raw_regex_matching(entry, expected):
    rules = build_rules({'none': [entry]})
    assert next(iter(rules)) == fr'^{entry}$'
    assert [get_scaling_rule(col, rules) for col in ('roc', 'roc_1')] == expected


def test_duplicate_override_keeps_last_value_and_first_position():
    rules = build_rules({'none': ['roc_1', 'volume'], 'divide_100': ['roc_1']})
    assert list(rules.items())[:2] == [(r'^roc_1$', 'divide_100'), (r'^volume$', 'none')]


@pytest.mark.parametrize('overrides', [None, {}])
@pytest.mark.parametrize('base', [None, {}])
def test_empty_inputs_keep_fresh_default_rules(overrides, base):
    rules = build_rules(overrides, base_rules=base)
    assert list(rules.items()) == list(DEFAULT_SCALING_RULES.items())
    assert rules is not DEFAULT_SCALING_RULES and rules is not base


def test_custom_base_preserves_lookup_and_scaler_defaults(recorded):
    rules = build_rules(base_rules={r'^volume$': 'none'})
    assert list(rules.items()) == [(r'^volume$', 'none')]
    assert get_scaling_rule('avgprice', rules) == 'none'
    train = recorded.select('avgprice')
    scaler = LinearScaler(train, rules=rules)
    assert scaler.default == 'standard'
    expected = (train['avgprice'] - train['avgprice'].mean()) / train['avgprice'].std(ddof=0)
    assert_series_equal(scaler.transform(train)['avgprice'], expected, rel_tol=1e-12, abs_tol=1e-12)
