# Triple-Barrier Method

The triple-barrier method labels each bar by *what happens next*, not by a fixed-horizon return. From each bar it opens three barriers — an upper profit-taking level, a lower stop-loss level, and a vertical time limit — and records which one the price path touches first. This is path-dependent labeling from Lopez de Prado's *Advances in Financial Machine Learning* (AFML), Chapter 3, and it produces targets that reflect how a position would actually resolve.

This guide explains the three barriers, how to size them, and how to attach the target to an experiment. For the complete parameter table and null-handling reference, see the `TripleBarrierTarget` section of [Targets](Targets.md).

## What This Guide Covers

- why path-dependent labeling differs from a fixed-horizon return label
- how the profit, stop, and time barriers are defined and computed
- how the profit-to-stop ratio is set through the two barrier multiples
- attaching the target to a manifest and a YAML experiment

## Prerequisites

- A working manifest or YAML experiment. Start with [Experiment Manifest](Experiment-Manifest.md) if you do not have one.
- The target reads a close-price column (`close` by default), so the data source must provide it.

## Current Scope

The shipping `TripleBarrierTarget` produces a single **ternary** label: `1` when the upper (profit) barrier is touched first, `-1` when the lower (stop) barrier is touched first, and `0` when neither is touched before the time barrier. There is one labeling mode; the target does not emit meta-labeling or companion columns. Bars are `null` during the volatility warmup and when the time barrier would extend past the available data — the manifest drops those rows.

## Why Path-Dependent Labeling

A fixed-horizon label asks "what is the return exactly `H` bars from now?" That ignores the path: a position that ran to a profit target on bar 3 and a position that bled to a stop on bar 3 can show the same sign at bar `H`. The triple-barrier label instead asks "which outcome came first?" — profit, stop, or timeout — which is the question a real position answers. Training on that label teaches the model to separate setups that resolve favorably from those that get stopped out, rather than setups that merely drift the right way by an arbitrary deadline.

## The Three Barriers

**Upper and lower barriers are volatility-scaled.** At each bar, Limen estimates volatility as the EWMA standard deviation of close-to-close returns (controlled by `span` and `min_periods`) and places the barriers at a multiple of that estimate:

- upper (profit) barrier at `+upper_multiple × volatility`
- lower (stop) barrier at `-lower_multiple × volatility`

The width is fixed using the **entry bar's** volatility, so barriers adapt to the local regime — wider in turbulent stretches, tighter in calm ones. There is no fixed-percentage mode; sizing is always in volatility units.

**The vertical barrier is a bar count.** `max_horizon` sets how many bars forward the scan runs. The method walks forward from the entry bar; the first bar whose return crosses a horizontal barrier sets the label, and if `max_horizon` bars pass with no touch, the label is `0`.

### Setting The Reward-To-Risk Ratio

There is no separate risk-reward parameter — the profit-to-stop ratio *is* `upper_multiple : lower_multiple`. The default `1.0 / 1.0` is symmetric. To label for a 2:1 reward-to-risk setup, widen the profit side or tighten the stop:

```python-fragment
fit_params={'upper_multiple': 2.0, 'lower_multiple': 1.0, 'max_horizon': 24}
```

Widening `lower_multiple` moves the stop further away, so paths that would have been stopped out under a tight stop are instead allowed to reach the profit barrier — shifting labels from `-1` toward `1`. Choose the multiples to match the trade geometry you actually intend to run.

## Attaching The Target

`TripleBarrierTarget` is a class-based target. All of its settings are constructor arguments, so they go in `fit_params`; there are no `transform_params`. Attach it with `with_target_label`:

```python-fragment
from limen.targets import TripleBarrierTarget

.with_target_label(
    'triple_barrier',
    TripleBarrierTarget,
    fit_params={
        'upper_multiple': 2.0,
        'lower_multiple': 1.0,
        'max_horizon': 24,
        'span': 100,
        'min_periods': 100,
    },
)
```

The constructor signature is:

```python-fragment
TripleBarrierTarget(train_data, target_name, upper_multiple=1.0,
                    lower_multiple=1.0, max_horizon=10, span=100,
                    min_periods=100, close_col='close')
```

`train_data` and `target_name` are supplied by the manifest; the rest are the barrier settings above. Each of `upper_multiple`, `lower_multiple`, `max_horizon`, `span`, and `min_periods` must be positive or the constructor raises `ValueError`. See [Targets](Targets.md) for the full parameter table and null-row semantics.

### As A YAML Experiment

In YAML the target is a block with `name`, `class`, and `fit_params`. Barrier settings can be fixed literals or `{param}` references bound from the experiment's `params`.

```yaml
    target:
      name: triple_barrier
      class: limen.targets.TripleBarrierTarget
      fit_params:
        upper_multiple: "{tp_multiple}"
        lower_multiple: "{sl_multiple}"
        max_horizon: "{horizon}"
        span: 100
        min_periods: 100
```

```yaml
  params:
    tp_multiple: [1.5, 2.0]
    sl_multiple: [1.0]
    horizon: [12, 24]
```

Literal values like `span: 100` pass through unchanged; the quoted `"{tp_multiple}"` strings resolve to the current round's parameter values, so barrier geometry can be swept like any other parameter.

## Choosing `span` And `min_periods`

`span` sets the EWMA window for the volatility estimate and `min_periods` sets the warmup before any volatility — and therefore any label — is emitted. Both consume rows: the target produces `null` labels for the first `min_periods` bars of a split, and the manifest drops them. Size them well below the length of your smallest split, or a short validation or test split can be left with too few labeled rows. See the null-handling notes in [Targets](Targets.md).

## Expected Artifacts

The target adds one column, named by the `name` you pass, holding the ternary label (`Int8`, values in `{-1, 0, 1}` or `null`). It appears in the prepared data and flows into the experiment records like any other target. Class balance is worth checking: with symmetric multiples and a short horizon, the `0` (timeout) class can dominate — adjust `max_horizon` and the multiples if the labeled classes are too skewed to learn from.

## Where To Look

- `limen/targets/triple_barrier.py` — the `TripleBarrierTarget` implementation and barrier scan
- `limen.targets.TripleBarrierTarget` — the resolvable path for YAML `class:` and manifest imports
- [Targets](Targets.md) — the full parameter table, null-row semantics, and the adjacent `RiskRewardRatioTarget`

## Read Next

- [Targets](Targets.md) for the complete target reference and other label constructions.
- [Experiment Manifest](Experiment-Manifest.md) for `with_target_label` and parameter references.
- [Perturbation Strategies](Perturbation-Strategies.md) to sweep barrier geometry and features robustly.
- [Backtest](Backtest.md) to evaluate how a labeled strategy would have traded.
