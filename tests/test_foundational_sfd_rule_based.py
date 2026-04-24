from limen.sfd.foundational_sfd import rule_based as rb_module


def test_params_returns_expected_keys() -> None:
    p = rb_module.params()
    assert 'rsi_period' in p
    assert 'rsi_threshold' in p
    assert 'ema_period' in p
    assert 'sharpe_std_threshold' in p
    assert 'sharpe_degradation_threshold' in p


def test_params_all_values_are_lists() -> None:
    for key, val in rb_module.params().items():
        assert isinstance(val, list), f'{key} should be a list'


def test_manifest_builds_with_rule_based_sentinel() -> None:
    m = rb_module.manifest()
    assert m._strategy is not None
    assert m._strategy.entry == 'entry'


def test_manifest_conditions_reference_parameterised_columns() -> None:
    m = rb_module.manifest()
    leaf_conditions = [c for c in m._strategy.conditions if 'type' in c]
    all_col_refs = [
        v for c in leaf_conditions for k, v in c.items() if isinstance(v, str)
    ]
    assert any('{rsi_period}' in ref for ref in all_col_refs)
    assert any('{ema_period}' in ref for ref in all_col_refs)
