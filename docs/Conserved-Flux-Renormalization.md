# Conserved Flux Renormalization

Conserved Flux treats the total traded value as conserved "stuff", the way in physics mass or charge is treated.

Renormalization repeatedly coarse-grains the data (60 s → 120 s → 240 s …), just like renormalisation-group steps in physics, and then measures how the statistics look after each zoom.

So CFR (Conserved Flux Renormalization) is a short-hand for “the routine that renormalises trade data while respecting the conserved-flux principle.”

## Conservation Law

This statement captures the essential meaning of a conservation law:  

"If nothing is being created or destroyed inside the region you’re watching, then whatever amount of stuff flows into it each moment must flow out again.  The same flow rate shows up no matter how finely or broadly you zoom your boundary.”

Conservation law is a rule saying that “stuff” (mass, charge, energy, trade value, …) cannot appear from nowhere or disappear into nothing. Because it’s conserved, whatever amount enters a region must either remain stored there or leave; in steady conditions, the incoming and outgoing flows balance exactly.

## Renormalization

In physics, **renormalization** means you keep “zooming out,” folding fine-scale details into coarser blocks, and watch how a system’s key numbers behave at each zoom level.  

For market data we do the same trick along the time axis.

1. **Pick a base window**  
   Choose the finest clock slice you trust, e.g.  
   `Δt = 60 s`.

2. **Coarse-grain once**  
   Merge neighbouring windows to make one twice as long  
   `60 s → 120 s`; add up the trade values so nothing is lost.

3. **Iterate the merge**  
   Keep doubling the span:  
   `120 s → 240 s → 480 s → …` until the whole 1-hour bar is covered.

4. **Record two scale-dependent numbers**  
   * **Flux variability** `σ / μ` — how bursty is the dollar flow at that scale?  
   * **Size entropy** `H` — how diverse are trade sizes inside that scale?

5. **Compare across scales**  
   In a well-mixed market both curves stay nearly flat as you zoom.  
   A spike at one rung means “something special happened at that time-scale.”

Each doubling step is an **RG (renormalization-group) step**, and the pair `(σ / μ , H)` are the scale-dependent “couplings.” CFR compresses the entire ladder into four scalars (mean + variance of each curve) so you can spot hours where the market’s flow stops looking scale-invariant.

## Deviation metrics

Once the 6-scale ladder is built, we compare it to its ideal shape  
(flat flux curve, 1-bit-per-octave entropy drop) and store the distance in
**two extra columns**:

| Column | Formula | Interpretation |
|--------|---------|----------------|
| `Δflux_rms` | $$\sqrt{\frac1n \sum_{k=0}^{n-1}\!\bigl((\sigma/\mu)_k-\overline{\sigma/\mu}\bigr)^{2}}$$ | Root-mean-square gap between the real flux-variability ladder and a perfectly **flat** line.  \>0.15 ⇒ one time-scale dominates the dollar flow. |
| `Δentropy_rms` | $$\sqrt{\frac1n \sum_{k=0}^{n-1}\!\bigl(H_k-(H_0-k)\bigr)^{2}}$$ | RMS gap between the real entropy ladder and the ideal **1-bit-per-octave** drop.  \>0.60 ⇒ some scales are dust-filled while others hold blocks. |

*Empirical BTC-spot (1 h) flags*

```text
Δflux_rms    > 0.15   → bursty or thin-book hour
Δentropy_rms > 0.60   → patchy size-mix hour
Trip both               almost certainly anomalous

Δflux_rms and Δentropy_rms are root-mean-square gaps: they take the (e.g., six) scale-by-scale errors, square them, average, and square-root, yielding one always-positive score whose size grows with anomaly strength—so a simple cut (e.g. > 0.15 or > 0.60) cleanly flags bars that break the "ideal" flux or entropy ladder.