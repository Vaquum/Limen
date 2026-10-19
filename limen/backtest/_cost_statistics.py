from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from typing import cast

from limen.backtest import _cost_history as history
from limen.backtest.trade_contract import JsonValue

_ZERO = Decimal(0)
_BPS = Decimal(10_000)
_MIN_JACKKNIFE_DAYS = 2
_MIN_OBSERVED_PARENTS = 20
_MIN_FILL_DAYS = 3


def _day(fill: Mapping[str, JsonValue]) -> str:
    return datetime.fromtimestamp(cast(int, fill['time_ns']) // 1_000_000_000, timezone.utc).date().isoformat()


def _estimate(fills: Sequence[Mapping[str, JsonValue]], cutoff_ns: int, currency: str) -> dict[str, JsonValue]:
    reference_total, quote_total, cost, fees, builder = (_ZERO for _ in range(5))
    parents: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    parent_costs: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    accounts: dict[str, Decimal] = defaultdict(Decimal)
    account_fees: dict[str, Decimal] = defaultdict(Decimal)
    price_complete = fee_complete = True
    missing_parents = builder_count = 0
    quantities: list[Decimal] = []
    for fill in fills:
        quantity, quote = (history.decimal(fill[name], name) for name in ('quantity', 'quote_notional'))
        quantities.append(quantity)
        quote_total += quote
        account = cast(str, fill['account'])
        accounts[account] += quote
        if fill['reference_price'] is None:
            price_complete = False
        else:
            referenced = quantity * history.decimal(fill['reference_price'], 'reference price')
            reference_total += referenced
            signed = (quote - referenced) * (1 if fill['side'] == 'buy' else -1)
            cost += signed
            if fill['parent_id'] is None:
                missing_parents += 1
            else:
                parent = (account, cast(str, fill['parent_id']))
                parents[parent] += referenced
                parent_costs[parent] += signed
        if (fill['fee'] is None or fill['fee_currency'] != currency or fill['fee_received_at_ns'] is None
                or cast(int, fill['fee_received_at_ns']) > cutoff_ns):
            fee_complete = False
        else:
            fees += history.decimal(fill['fee'], 'fee')
            account_fees[account] += history.decimal(fill['fee'], 'fee')
        if (fill['builder_fee'] is not None and fill['fee_currency'] == currency and fill['fee_received_at_ns'] is not None
                and cast(int, fill['fee_received_at_ns']) <= cutoff_ns):
            builder += history.decimal(fill['builder_fee'], 'builder fee')
            builder_count += 1
    weights = list(parents.values())
    complete_weights = bool(weights) and price_complete and not missing_parents
    quantiles = sorted(_BPS * parent_costs[key] / value for key, value in parents.items())
    squared = sum((value * value for value in weights), _ZERO)
    return history.mapping({
        'price_cost_bps': str(_BPS * cost / reference_total) if price_complete and reference_total else None,
        'fee_bps': str(_BPS * fees / quote_total) if fee_complete and quote_total else None,
        'legs': len(fills), 'parents': len(parents), 'days': len({_day(fill) for fill in fills}),
        'missing_parent_legs': missing_parents,
        'parent_effective_n': str(reference_total * reference_total / squared) if complete_weights else None,
        'max_parent_weight': str(max(weights) / reference_total) if complete_weights else None,
        'parent_quantiles_bps': {str(p): str(quantiles[max(0, (len(quantiles) * p + 99) // 100 - 1)])
                                for p in (25, 50, 75, 95)} if price_complete and quantiles else {},
        'parent_quantile_method': 'unweighted_empirical_left_continuous',
        'quantity_min': str(min(quantities)) if quantities else None,
        'quantity_max': str(max(quantities)) if quantities else None,
        'quote_notional': str(quote_total), 'accounts': len(accounts),
        'max_account_quote_weight': str(max(accounts.values()) / quote_total) if accounts else None,
        'top_three_account_quote_weight': str(sum(sorted(accounts.values())[-3:]) / quote_total) if accounts else None,
        'account_fee_bps': {key: str(_BPS * account_fees[key] / amount) for key, amount in sorted(accounts.items())}
                          if fee_complete else {},
        'known_builder_fee': str(builder), 'builder_fee_present_legs': builder_count,
        'builder_fee_unknown_legs': len(fills) - builder_count,
        'twap_legs': sum(fill['twap_id'] is not None for fill in fills),
        'fee_transfer': 'unavailable_account_specific',
    })


def _sensitivity(fills: Sequence[Mapping[str, JsonValue]], cutoff_ns: int, currency: str) -> dict[str, JsonValue]:
    days = sorted({_day(fill) for fill in fills})
    deleted = {day: _estimate([fill for fill in fills if _day(fill) != day], cutoff_ns, currency) for day in days}
    result: dict[str, JsonValue] = {'leave_one_day_out': cast(dict[str, JsonValue], deleted),
                                   'day_cluster_count': len(days), 'coverage_guarantee': False}
    for name in ('price_cost_bps', 'fee_bps'):
        values = [item[name] for item in deleted.values()]
        if len(values) >= _MIN_JACKKNIFE_DAYS and all(isinstance(value, str) for value in values):
            numeric = [Decimal(cast(str, value)) for value in values]
            mean = sum(numeric, _ZERO) / len(numeric)
            variance = Decimal(len(numeric) - 1) / len(numeric) * sum(((value - mean) ** 2 for value in numeric), _ZERO)
            result[name] = {'min': str(min(numeric)), 'max': str(max(numeric)), 'jackknife_se': str(variance.sqrt())}
        else:
            result[name] = None
    return result


def report(value: Mapping[str, JsonValue], cutoff_ns: int) -> dict[str, JsonValue]:
    source = history.mapping(value['source'])
    product = history.mapping(source['product'])
    currency = cast(str, product['quote_currency'])
    fills = history.rows(value['fills'], 'fills')
    selected = [fill for fill in fills if history.selected(fill)]
    admitted: list[dict[str, JsonValue]] = []
    reasons: list[str] = []
    receipts: list[int] = []
    exclusions: dict[str, int] = defaultdict(int)
    for fill in selected:
        reason = None
        if fill['reference_price'] is None or fill['reference_time_ns'] is None:
            reason = 'missing_selected_reference'
        elif fill['reference_received_at_ns'] is None:
            reason = 'missing_reference_receipt'
        elif cast(int, fill['received_at_ns']) > cutoff_ns or cast(int, fill['reference_received_at_ns']) > cutoff_ns:
            reason = 'required_evidence_after_cutoff'
        if reason is not None:
            reasons.append(reason)
            exclusions[reason] += 1
        else:
            admitted.append(fill)
            receipts.extend((cast(int, fill['received_at_ns']), cast(int, fill['reference_received_at_ns'])))
            fee_receipt = fill['fee_received_at_ns']
            if fee_receipt is not None and cast(int, fee_receipt) <= cutoff_ns:
                receipts.append(cast(int, fee_receipt))
    if any(fill['role'] == 'unknown' for fill in fills):
        reasons.append('unknown_matching_role')
    estimates: dict[str, JsonValue] = {}
    sensitivities: dict[str, JsonValue] = {}
    day_side: dict[str, JsonValue] = {}
    thin_effective: list[str] = []
    for side in ('buy', 'sell'):
        side_fills = [fill for fill in admitted if fill['side'] == side]
        estimate = _estimate(side_fills, cutoff_ns, currency)
        estimates[side] = estimate
        sensitivities[side] = _sensitivity(side_fills, cutoff_ns, currency)
        if cast(int, estimate['parents']) < _MIN_OBSERVED_PARENTS or cast(int, estimate['days']) < _MIN_FILL_DAYS:
            reasons.append(f'thin_observed_support:{side}')
        if estimate['missing_parent_legs']:
            reasons.append(f'unknown_parent_identity:{side}')
        price = estimate['price_cost_bps']
        if price is None:
            reasons.append(f'unavailable_price_estimate:{side}')
        elif Decimal(cast(str, price)) < 0:
            reasons.append(f'negative_price_constant:{side}')
        effective = estimate['parent_effective_n']
        if effective is not None and Decimal(cast(str, effective)) < _MIN_OBSERVED_PARENTS:
            thin_effective.append(side)
        for day in sorted({_day(fill) for fill in selected}):
            group = [fill for fill in side_fills if _day(fill) == day]
            observed = [fill for fill in selected if _day(fill) == day and fill['side'] == side]
            day_side[day + ':' + side] = {'selected': len(observed), 'admitted': len(group),
                                         'excluded': len(observed) - len(group),
                                         'admitted_estimate': _estimate(group, cutoff_ns, currency)}
    slip = None
    if not reasons:
        slip = str(sum((Decimal(cast(str, history.mapping(estimates[side])['price_cost_bps'])) for side in ('buy', 'sell')), _ZERO) / 2)
    disposition = ('unavailable' if reasons else
                   'retrospective_low_effective_support' if thin_effective else 'retrospective_cost_scenario')
    estimates['slip_bps'] = slip
    makers = [fill for fill in fills if fill['role'] == 'maker' and cast(int, fill['received_at_ns']) <= cutoff_ns]
    known_makers = [fill for fill in makers if fill['fee'] is not None and fill['fee_currency'] == currency
                    and fill['fee_received_at_ns'] is not None and cast(int, fill['fee_received_at_ns']) <= cutoff_ns]
    maker_quote = sum((history.decimal(fill['quote_notional'], 'maker quote') for fill in known_makers), _ZERO)
    maker_fee = sum((history.decimal(fill['fee'], 'maker fee') for fill in known_makers), _ZERO)
    estimates['provided_maker_sample'] = {
        'legs': len(makers), 'known_fee_legs': len(known_makers),
        'fee_bps': str(_BPS * maker_fee / maker_quote) if maker_quote and len(makers) == len(known_makers) else None,
        'transfer': 'unavailable_matching_role',
    }
    return history.mapping({
        'method': history.METHOD, 'source': copy.deepcopy(source),
        'clocks': {'cutoff_ns': cutoff_ns, 'available_at_ns': max(receipts) if receipts else None,
                   'clock': 'assumed_exchange_label', 'historical_delivery': 'unknown'},
        'estimates': estimates,
        'support': {'disposition': disposition, 'reasons': sorted(set(reasons)),
                    'selected_legs': len(selected), 'admitted_legs': len(admitted),
                    'low_effective_support_sides': thin_effective, 'day_side': day_side,
                    'parent_completeness': 'requested_and_terminal_unknown'},
        'sensitivity': {'computed_selected_population': sensitivities,
                        'source_reported_diagnostics': copy.deepcopy(value['source_diagnostics'])},
        'exclusions': {'computed_selected_population': dict(exclusions),
                       'source_reported_window_summary': copy.deepcopy(value['source_window_summary'])},
        'limitations': ['retrospective_software_stress', 'historical_delivery_assumed',
                        'minute_label_snapshot_inferred', 'no_representative_cost_survival_claim',
                        'no_fee_transfer', 'unknown_whole_order_completeness',
                        'three_day_jackknife_has_no_coverage_guarantee'],
    })


__all__: list[str] = []
