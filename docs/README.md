# Vaquum Limen

Vaquum Limen reduces complex and otherwise out-of-reach research, model development, and trading signal workflows into one cohesive Python API, powering Bitcoin quants with unparalleled ergonomics and productivity. Limen does not execute trades, and can be used as a source of alpha with any trading system.

## Introduction

### Core Capabilities

Vaquum Limen brings every step - data ingestion, feature engineering, machine learning model development, parameter sweep, ensembling, and decision making - into a single closed-loop optimization cycle. This thermodynamic decision signal engine is delivered through one cohesive API and a set of lightweight, single-file templates called SFDs (Single-File Decoder). 

**Limen is fully parametric, closed-loop, and Bitcoin-only.** These three core tenets will be explained in the below sections. 

### Parametric

Every aspect of Limen can be controlled through parameters, and every parameter is coupled with rich metrics. This gives humans infinitely deep and infinitely fine control over every aspect of the system, without introducing any cumbersome abstraction that prohibit expression of human creativity. 

The system relentlessly tests the hypothesis **"x paramater value for y parameter will yield z profit and loss in live trading".** across a vast n-dimensional space of possibilities where everything from ensembles to a single hyperparameter are treated as parameters.

Read more about the radical benefits of parametrization in [Three Eras of Knowledge Objects](https://medium.com/aecmaster/the-three-eras-of-knowledge-objects-994fa4ed9372).

### Closed-Loop

Since all parts are inseparably tied together, with zero signal leakage, the system knows all of its previous states, and is able to always adapt to the current prevailing market conditions sufficiently to deliver positive results under any market conditions.

### Bitcoin-Only

Bitcoin commands a dedicated system. A system that ignores all the noise. A system that says "Bitcoin is special". Such a system will always have edge over generic crypto signal systems or other generic trading-related signal systems.

## Architecture

### Three Sub-Systems

Vaquum Limen consist of three distinct sub-systems: 

- `Experiment` Sub-System
- `Cohorts` Sub-System
- `Manager` Sub-System

`Experiment` is the sub-system where alpha is systematically discovered, primarily through the means of comprehensive parameter sweep involving multiple machine learning architectures.

Read more in [Universal Experiment Loop](Universal-Experiment-Loop.md) and [Single-File Decoder](Single-File-Decoder.md).

`Cohort` is the sub-system that harnesses discovered alpha from Experiment into curated alpha. 

Read more in [Regime Diversified Opinion Pools](Regime-Diversified-Opinion-Pools.md).

`Manager` is the sub-system where alpha is crystallized into tradeable decisions.
