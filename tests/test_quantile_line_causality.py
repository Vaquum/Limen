"""Quantile end events must depend only on the recorded prefix."""
import numpy as np
import pytest
from polars.testing import assert_frame_equal

from limen.features.quantile_price_lines import quantile_price_lines
from tests.test_line_performance import BASELINES, DATA


@pytest.mark.parametrize('quantile', [.7, .8])
@pytest.mark.parametrize('research', [True, False])
def test_quantile_end_features_are_prefix_causal(quantile, research):
    kwargs = {'max_duration_hours': 48, 'min_height_pct': .002,
              'quantile_threshold': quantile, 'include_research_only': research}
    prefix = quantile_price_lines(DATA.head(192), **kwargs)
    extended = quantile_price_lines(DATA.head(384), **kwargs)
    columns = [name for name in prefix.columns if name not in ('close', 'active_quantile_count')]
    assert len(columns) == 5
    assert_frame_equal(prefix.select(columns), extended.head(192).select(columns),
                       rel_tol=1e-12, abs_tol=1e-12)
    other = quantile_price_lines(DATA.head(384), **{**kwargs, 'include_research_only': not research})
    assert_frame_equal(extended.select(columns), other.select(columns), check_exact=True)
    if research:
        expected = BASELINES['quantile_price_lines'].quantile_price_lines(DATA.head(384), **kwargs)
        assert extended['active_quantile_count'].equals(expected['active_quantile_count'])


@pytest.mark.parametrize('quantile', [0., .7, .8, 1.])
@pytest.mark.parametrize('research', [True, False])
@pytest.mark.parametrize('windows', [(48, 6, 24), (3, 1, 2)])
def test_quantile_rows_match_frozen_prefix_oracle(quantile, research, windows):
    kwargs = dict(zip(('density_lookback_hours', 'momentum_lookback_hours', 'height_lookback_hours'), windows, strict=True))
    kwargs.update(max_duration_hours=48, min_height_pct=.002,
                  quantile_threshold=quantile, include_research_only=research)
    result = quantile_price_lines(DATA.head(384), **kwargs)
    columns = [name for name in result.columns if name != 'active_quantile_count']
    for cutoff in (1, 2, 17, 48, 96, 192, 384):
        expected = BASELINES['quantile_price_lines'].quantile_price_lines(DATA.head(cutoff), **kwargs)
        assert_frame_equal(result.slice(cutoff - 1, 1).select(columns), expected.tail(1).select(columns),
                           rel_tol=1e-12, abs_tol=1e-12)
    assert result.schema == expected.schema
    np.testing.assert_array_equal(result['close'].to_numpy(), DATA['close'].head(384).to_numpy())
