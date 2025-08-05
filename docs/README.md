# Background

## Vision

**Playing the Song of Bitcoin through a continuous parametric trading system to earn just-enough profit.**

## Mission 

**Uncover hidden alpha and let it crystallize in an autonomous, parametric day-trading system—software that listens, learns, and executes in real time, earning just-enough profit with disciplined precision.**

## Semantics

`Playing` is the act of resembling a song.

`Song` is the the continuous flow of modes in which the song of Bitcoin is played. 

`Parametric` means that everything is considered a parameter.

`Trading` because all energy is directed towards the sole purpose of yielding signals that lead to positive P&L in live trading. 

`System` means Vaquum Loop and the mental-model that underlies it.

`Just-enough` means that there is a limit, and beyond that limit, there is no interest for gaining more.

`Profits` because the only relevant metric for trading is the degree to which allocated capital yields positive returns. 

`Through` because it creates the connection between the second part, `just-enough trading profits` and the first part `parametric day-trading system`.

`Continuous` because it is never ending. This means it never stops. This is a stark contrast to the typical approach, where one starts something, waits for it to conclude, and keeps working in that kind of interrupted way across all manners. 

`Everything` because all aspects of activity are subsumed by it. 

`As` to create the connection between `continuous everything` and ` parametric` to be such that the latter represents the way in which the former is made practical. 

### Just-in-Case Trading Profits

Just-in-case trading profits mindset leads to nothing being enough. No matter the volume of trading profits, one is trapped in always looking for more. This is classic zero-sum mindset.

### Just-Enough Trading Profits

Just-enough trading profits mindset skillfully avoids the incessant sense of insufficiency. First, one clearly establishes what is enough, and then cultivates a sense of not feeling the need to exceed that. This is classic positive-sum mindset.

# The Instrument: Loop

Vaquum Loop is a Bitcoin-first research and trading platform built for quantitative traders. It brings every step—data ingestion, research, model development, and live execution—into a single closed-loop optimization cycle, delivered through one cohesive API and a set of lightweight, single-file templates.

Loop is the instrument we use to play the Song of Bitcoin. Loop gets its energy from Data, Indicators, Features, and Single-File Models. It yields Parameters and Metrics that reveal the Bitcoin’s harnomic resonance. 

**When Loop and Bitcoin hum in unison, every block becomes a beat, every trade a note—and the chorus rings out, ‘Just-enough profit, forever on time.**

## Motivation

The motivation behind Vaquum Loop is to bring together all aspects of research and trading into a single parametric optimization loop.

The reason to bring everything into a single parametric optimization loop, is to unlock the ability to treat everything as a single never-ending experiment loop. 

Within this experiment loop, the hypothesis is always that "x paramater value for y parameter will yield z profit and loss in live trading". 

Since all parts, from data ingestion to actual trading results is inseparably tied together, through parameters and parameter value ranges, the system knows all of its previous states, and is able to always adapt to the current prevailing market conditions.

These parts are here called *Folds*. 

## Folds

 Fold is a self-contained stage in Loop: it takes in data and parameters, performs one focused task, and hands back new data and metrics. Vaquum Loop brings together 12 distinct Folds.

- [`Data`](HistoricalData.md)
- [`Indicator`](Indicators.md)
- [`Feature`](Features.md)
- [`SFM`](Single-File-Model.md)
- [`UEL`](Universal-Experiment-Loop.md) 
- `Log`
- `Benchmark`
- `Backtest`
- `Paper Trade`
- `Live Trade`
- `Trading Result`
- `Feedback`

**NOTE:** Loop should not be thought of as something that runs and completes, but something that is continuous, and where subsequent Folds continuously send feedback to previous Folds. 

Each Fold in itself can be a continuous feedback loop. The “full” Loop consist of up to 11 continuous feedback loops, each feeding back to itself and all the other Folds.

### `Data`

Klines or trades data for spot or futures markets, pulled through one of the supported Loop endpoints.

Read more in: [HistoricalData](Historical-Data.md)

### `Indicator`

Includes common technical indicators, and any other non-compound signal that can be used for training models. Both Indicators and Features can be used in SFMs.

Read more in: [Indicators](Indicators.md)

### `Feature`

More complex than Indicators, and often involve further refining Indicators or combining several Indicators into a single Feature. Feature can be anything, for example data on moon phases, sentiment, electricity prices, etc. Both Indicators and Features can be used in SFMs.

Read more in: [Feature](Features.md)

### `SFM`

Contains all parameters, data preparation code, and model operation codes in a single python file. For example, representing an XGBoost bullish regime binary classifier.

Read more in: [Single-File-Model](Single-File-Model.md)

### `UEL`

(Universal Experiment Loop) is an iterative experiment loop where SFMs are used as basis for ad-hoc or continuous parameter sweeps. 

Read more in: [UEL](Universal-Experiment-Loop.md)

### `Log`

The output of a single run of `uel.run` for n parameter permutations. It contains metrics together with parameter values.

Read more in: [Log](Log.md)

### `Benchmark`

TBA

### `Backtest`

TBA

### `Paper Trade`

TBA

### - `Live Trade`

TBA

### - `Trading Result`

TBA

### - `Feedback`

TBA

## Developer Docs

Read more in: [Development Guidelines](Developer/README.md)