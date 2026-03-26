# Vaquum Limen

## Introduction

Vaquum Limen reduces complex and otherwise out-of-reach research, model development, and trading signal workflows into one cohesive Python API, powering Bitcoin quants with unparalleled ergonomics and productivity. Limen does not execute trades, and can be used as a source of alpha with any trading system.

**Limen is fully parametric, closed-loop, and Bitcoin-only.** These three core tenets are explained in the sections below.

## Core Tenets

Vaquum Limen brings every step - data ingestion, feature engineering, machine learning model development, parameter sweep, ensembling, and signal evaluation - into a single closed-loop optimization cycle. This thermodynamic alpha research engine is delivered through one cohesive API and a set of lightweight, single-file templates called SFDs (Single-File Decoder). 

### Parametric

Hard-coded assumptions are the enemy of alpha. Limen exposes the entire decision surface—from neural network hyperparameters to ensemble weights—as a tunable, metric-rich environment. This replaces rigid abstractions with granular control, allowing the system to explore the full topology of potential strategies.

The system relentlessly validates a single core hypothesis:

"Parameter configuration $X$ will yield Profit $Z$ in live market conditions."

By treating strategy design as a high-dimensional search problem, Limen moves beyond intuition. It systematically sweeps this vast n-dimensional space, ensuring that every variable is not just chosen, but empirically optimized for survival.

Read more about the radical benefits of parametrization in [Three Eras of Knowledge Objects](https://medium.com/aecmaster/the-three-eras-of-knowledge-objects-994fa4ed9372).

### Closed-Loop

Traditional quantitative pipelines are fragmented—researchers work in silos and hand off static models to engineers, resulting in friction and signal decay. Limen eliminates this fragility by unifying ingestion, feature engineering, and signal research into a single, continuous continuum.

The system enforces a feedback loop where realized performance is the only truth.

Because the signal generation environment is tightly aligned with the research environment, there is minimal translation loss. Realized performance and downstream market feedback—specifically the divergence between predicted and realized outcomes—can be fed back into the optimization engine. This allows Limen to metabolize volatility and evolve its internal logic dynamically, ensuring that future research is grounded in observed outcomes.

### Bitcoin-Only

Bitcoin is not just another ticker in a dataframe; it is a unique monetary network with distinct microstructure, on-chain dynamics, and volatility profiles. Generic multi-asset platforms pay an "abstraction tax"—they are forced to compromise on feature engineering and model architecture to accommodate thousands of dissimilar assets.

Limen eliminates this compromise. It is purpose-built to exploit the specific idiosyncrasies of the Bitcoin market, unburdened by the compatibility debt of multi-asset support. **Generalization is the dilution of edge.**

By treating Bitcoin as the sole first-class asset, Limen bypasses both the noise of the broader crypto "casino" and the outdated assumptions of traditional finance. The result is a system that optimizes its entire topology around the specific heartbeat of the Bitcoin network.

## Architecture

### Two Core Sub-Systems

Vaquum Limen consists of two distinct sub-systems:

- `Experiment` Sub-System
- `Cohorts` Sub-System

`Experiment` is the sub-system where alpha is systematically discovered, primarily through comprehensive parameter sweep across multiple machine learning architectures.

`Cohorts` is the sub-system that turns discovered alpha from `Experiment` into curated alpha, primarily through ensembling and meta-modelling methods.

Trade decisioning does not live inside Limen. Downstream decision logic belongs in Nexus, which consumes Limen outputs and turns decoder cohorts into validated trading decisions.

### In Practice

`Experiment` starts with [Data](Data-Bars.md), which can be native kline data, Limen's built-in threshold bars, or any external OHLC data as long as it contains the expected price columns.

`Experiment` then continues by converting data into [Indicators](Indicators.md) and [Features](Features.md). In addition to Limen's built-in indicator and feature library, custom polars expressions can also be used to create independent variables for the `Experiment`.

`Experiment` then continues with applying [Scalers](Scalers.md) and [Transforms](Transforms.md), which again could be some of the built-in ones, or any custom polars expressions.

Everything in `Experiment` is captured in [Single-File Decoder](Single-File-Decoder.md), also called SFD, which could be one of three flavors: 

1) One of the built-in SFDs
2) A locally customized version of one of the built-in SFDs
3) A completely custom SFD

Manifest-driven SFDs are the standard path into [Universal Experiment Loop](Universal-Experiment-Loop.md), but custom SFDs can also provide explicit `prep()` and `model()` functions. UEL is Limen's parameter-sweep engine.

Completing an `Experiment` yields several analytical artefacts through [Log](Log.md), namely a parameter sweep log, benchmark-style confusion analytics, and backtest results.

These artefacts can then be used to create [Cohorts](Regime-Diversified-Opinion-Pools.md). 

These cohort outputs can then be passed downstream into Nexus for decisioning and into other Vaquum systems for execution, oversight, and auditability.
