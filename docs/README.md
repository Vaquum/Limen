# Vaquum Limen

## Introduction

Vaquum Limen reduces complex and otherwise out-of-reach research, model development, and trading signal workflows into one cohesive Python API, powering Bitcoin quants with unparalleled ergonomics and productivity. Limen does not execute trades, and can be used as a source of alpha with any trading system.

**Limen is fully parametric, closed-loop, and Bitcoin-only.** These three core tenets will be explained in the below sections. 

## Core Tenets

Vaquum Limen brings every step - data ingestion, feature engineering, machine learning model development, parameter sweep, ensembling, and decision making - into a single closed-loop optimization cycle. This thermodynamic decision signal engine is delivered through one cohesive API and a set of lightweight, single-file templates called SFDs (Single-File Decoder). 

### Parametric

Hard-coded assumptions are the enemy of alpha. Limen exposes the entire decision surface—from neural network hyperparameters to ensemble weights—as a tunable, metric-rich environment. This replaces rigid abstractions with granular control, allowing the system to explore the full topology of potential strategies.

The system relentlessly validates a single core hypothesis:

"Parameter configuration $X$ will yield Profit $Z$ in live market conditions."

By treating strategy design as a high-dimensional search problem, Limen moves beyond intuition. It systematically sweeps this vast n-dimensional space, ensuring that every variable is not just chosen, but empirically optimized for survival.

Read more about the radical benefits of parametrization in [Three Eras of Knowledge Objects](https://medium.com/aecmaster/the-three-eras-of-knowledge-objects-994fa4ed9372).

### Closed-Loop

Traditional quantitative pipelines are fragmented—researchers work in silos and hand off static models to engineers, resulting in friction and signal decay. Limen eliminates this fragility by unifying ingestion, feature engineering, and decision-making into a single, continuous continuum.

The system enforces a feedback loop where realized performance is the only truth.

Because the signal generation environment is identical to the research environment, there is zero translation loss. Live market feedback—specifically the divergence between predicted and realized outcomes—is immediately fed back into the optimization engine. This allows the system to metabolize volatility and evolve its internal logic dynamically, ensuring the strategy deployed is always the strategy tested.

### Bitcoin-Only

Bitcoin is not just another ticker in a dataframe; it is a unique monetary network with distinct microstructure, on-chain dynamics, and volatility profiles. Generic multi-asset platforms pay an "abstraction tax"—they are forced to compromise on feature engineering and model architecture to accommodate thousands of dissimilar assets.

Limen eliminates this compromise. It is purpose-built to exploit the specific idiosyncrasies of the Bitcoin market, unburdened by the compatibility debt of multi-asset support. **Generalization is the dilution of edge.**

By treating Bitcoin as the sole first-class asset, Limen bypasses both the noise of the broader crypto "casino" and the outdated assumptions of traditional finance. The result is a system that optimizes its entire topology around the specific heartbeat of the Bitcoin network.

## Architecture

### Three Sub-Systems

Vaquum Limen consist of three distinct sub-systems: 

- `Experiment` Sub-System
- `Cohorts` Sub-System
- `Manager` Sub-System

`Experiment` is the sub-system where alpha is systematically discovered, primarily through the means of comprehensive parameter sweep involving multiple machine learning architectures.

`Cohorts` is the sub-system that harnesses discovered alpha from `Experiment` into curated alpha, primarily through the means of various ensembling and meta-modelling methods.

`Manager` is the sub-system where alpha is crystallized into tradeable decisions, primarily through combining predictive signals with trading directives.

### In Practice

`Experiment` starts with [Data](Data-Bars.md), which could be standard, imbalance, or run bars. Data could also be any OHLC data from any source, as long as it contains standard OHLC columns.

`Experiment` then continues with converting data into [Indicators](Indicators.md) and [Features](Features.md). In addition to the built-in ones, indicators and features, any custom polars expression could be used to create indepdent variables to be used in the `Experiment`.

`Experiment` then continues with applying [Scalers](Scalers.md) and [Transforms](Transforms.md), which again could be some of the built-in ones, or any custom polars expressions.

Everything in `Experiment` is captured in [Single-File Decoder](Single-File-Decoder.md), also called SFD, which could be one of three flavors: 

1) One of the built-in SFDs
2) A locally customized version of one of the built-in SFDs
3) A completely custom SFD

All of this is wrapped into a manifest for [Universal Experiment Loop](Universal-Experiment-Loop.md), which is basically a comphrensive parameter sweep suite. 

Completing an `Experiment` yields several analytical artefacts called [Log](Log.md), namely a parameter sweep log, an advanced interpretation of a confusion matrix, and comphrensive backtest results. 

These artefacts can be then used to create [Cohorts](Regime-Diversified-Opinion-Pools.md). 

The `Manager` sub-system is currently being built.