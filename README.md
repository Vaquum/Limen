<h1 align="center">
  <br>
  <a href="https://github.com/Vaquum"><img src="https://github.com/Vaquum/Home/raw/main/assets/Logo.png" alt="Vaquum" width="150"></a>
  <br>
</h1>

<h3 align="center">Loop</h3>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#owner">Owner</a> •
  <a href="#integrations">Integrations</a> •
  <a href="#docs">Docs</a>
</p>
<hr>

# Description

Vaquum Loop is a Bitcoin-first research and trading platform for quantitative traders, bringing together a comprehensive set of capabilities into a single closed-loop optimization cycle. Vaquum Loop reduces complex and mostly out-of-reach DataOps, research, model development, and trading workflows into one cohesive API and a set of single-file templates. Vaquum Loop powers quants with unparalleled ergonomics and productivity.

# Owner

- [@mikkokotila](https://github.com/mikkokotila)

# Integrations

- Intgrates upstream with [Binancial](https://github.com/vaquum/binancial).

# Docs

## Vision

**Just-enough trading profits through continuous everything as higher-order functions.**

`Just-enough` means that there is a limit, and beyond that limit, there is no interest for gaining more.

`trading` because all energy is directed towards the sole purpose of yielding probabilities that have positive profit and loss in live trading. 

`profits` because the only relevant metric for trading is the degree to which allocated capital yields positive returns. 

`through` because it creates the connection between the first part, `just-enough trading profits` and the second part `continuous everything as higher-order functions`.

`continuous` because it is never ending. This means it never stops. This is a stark contrast to the typical approach, where one starts something, waits for it to conclude, and keeps working that kind of interrupted way across all manners. 

`everything` because all aspects of activity are subsumed by it. 

`as` to create the connection between `continuous everything` and `higher-order functions` to be such that the latter represents the way in which the former is made practical. 

`higher-order` to indicate that `functions` which follows accept as their input arguments parameters as well as functions. 

`functions` because everything is represented as functions in general, and more specifically, whenever possible, composable, generic functions. 

## Motivation



## An Experiment

The core consept for individual contributors to understand, is that of an `experiment`. The meaning of an `experiment` can be clearly understood from the following. 

- The subject of the `experiment` i.e. the agent performing the `experiment` is the individual contributor
- The object of the `experiment` i.e. that which is being studied is various parameters to various functions throughout phases of an `experiment`
- The action of the `experiment` i.e. the activity is to iterate through various parameters and their value ranges

## The Individual Contributor (the subject)

Everyone in Vaquum is an individual contributor. That is to say, everyone working with Vaquum works individually towards the goal of predictors that yield just-enough trading profits. Just-enough trading profits is a stark contrast to the typical mindset governing financial trading activity, just-in-case trading profits. 

### Just-in-Case Trading Profits

Just-in-case trading profits mindset leads to nothing being enough. No matter the volume of trading profits, one is trapped in always looking for more. This is classic zero-sum mindset.

### Just-Enough Trading Profits

Just-enough trading profits mindset skillfully avoids the sense of nothing ever being enough. First, one clearly establishes what is enough, and never feels the need to exceed that. This is classic positive-sum mindset.

## Phases of an Experiment (the object)

Currently, `Loop` brings together the following capabilities: 

1) Data
2) Aggregates
3) Indicators
4) Models (and their respective data `prep` and `params`)
5) Validation `metrics`
6) Benchmarking

In the future, the following capabities will be added: 

7) Filters
8) Ensembles

These eight capabilities present steps in a step-wise progression. Based on sequentially progressing through these eight phases, the work of an individual contributor yields the following: 

9) Predictors
10) Probabilities

The output of this final step, the actual probabilities, are then fed to the trading optimizer. This is a complete picture of the `experiment` up to the part from where onwards trading logic and other trading-related intricacies come into play.

## The Meaning of Parameter Sweep (the action)

It is typically thought that the focus of the sweep is specifically the model hyperparameters, and only these. This led to the bastardized term "hyperparameter optimization". This perspective is extremely limiting and entirely misses the point of parameter sweeping. 

In short, the point of parameter sweeping is that since such a practice is possible, and since more or less anything and everything can be readily parametrized, there should be no limit to where this approach can be applied. 

**Not only the idea of sweeping through parameters can be extended beyond the model and its hyperparameters, to data fetching, data pre-processing, feature engineering, and all other aspects of classifier development lifecycle, but it can also be extended well beyond input arguments. For example, conditional logic can be handled as parameters, and even individual fragments of code can be fully parametric, and therefore a subject of a parameter sweep.**

In other words, the idea of performing a parameter sweep is equally relevant to all of the above-mentioned ten steps. This is a crucial key point, and our success depends on undertanding it, putting it into practice, and realizing its unrestrained power to yield the most meaningful probabilities for live trading at any given point in time, regardless of the prevailing circumstances.