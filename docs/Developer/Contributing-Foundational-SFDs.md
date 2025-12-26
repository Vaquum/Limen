# Contributing Foundational SFDs

## Background

With regard to Single File Decoders (SFD), there are two kinds: `foundational` and `custom`.  This document is focused on Foundational SFDs, and sets forth strict requirements for contributing Foundational SFDs into Vaquum Loop.

## Terminology

The Loop `Foundational SFD` here means two things coming together: The `Foundational SFD` itself coming together with a `Reference Architecture`. 

A canonical example can be found in:

- **Foundational SFD**: https://github.com/Vaquum/Loop/blob/main/loop/sfd/foundational_sfd/logreg_binary.py
- **Reference Architecture**: https://github.com/Vaquum/Loop/blob/main/loop/sfd/reference_architecture/logreg_binary.py

## Motivation

The motivation for a Foundational SFD, is to capture honestly and directly, pertaining to a single trainable reference architecture (e.g. LogReg), the state-of-the-art scientific data and literature without adding any out-of-literature innovation to it. 

**Contributing Foundational SFDs is contributing to the beating heart of Loop, it is where the intelligence resides. Rest of Loop is, for the most part, aggregation and transport.**

**NOTE:** Initially, during January 2026, we want to incorporate 6-8 new Foundational Manifests with their respective underlying reference architectures. https://github.com/Vaquum/Loop/issues/297

## Minimal Requirements

- Relies on a `Reference Architecture`
- Is entirely based on `manifest`
- `Parameters` are richly exposed
- All the requirements in the below sections are satisfied

**The test:** Can anyone just run it with Loop in a large scan, and it will yield something meaningful? 

For a contribution to pass this test, it of course requires sufficient exploration of the parameter space. 

## Foundational SFD

A `Foundational SFD` is an SFD that composes a design of experiment based on a `Reference Architecture` such as `LogReg`.  

The `Foundational SFD` has two parts: `params` and `manifest` which are always named like so.

The `Reference Architecture` has one part: `model` which is named according to the underlying architecture (e.g. logreg) and the type of decoder it is (e.g. binary), for example, `logreg_binary`. The name must have exactly two parts separated by underscore.

The `manifest` in a `Foundational SFD` can be used to incorporate various `Extensions` into the experiment.

`Extensions` are the primary mean by which the contributing modeller transmits intelligence in to the `Foundational SFD`. 

**NOTE:** how this mode of transmitting intelligence is in stark contrast with the approach where the contributing modeller transmits intelligence into the reference architecture through bespoke workflow code. 

**Here, adding workflow code that is not strictly specific to the `Reference Architecture` is strictly prohibited.** 

If and when the authoritative literature implies additional workflow interventions, these must be called from `loop.utils` and must be generally callable by any reference architecture.

## Extensions

Extensions can include  `Data`, `Indicators`, `Features`, `Transforms`, and `Labels`. 

`Data` include any input data, for the time being, various market data, framed in various ways.

`Indicators` include common technical indicators, and any other non-compound signal that can be used for training models. Indicators must be contributed to `loop.indicators`.

`Features` are generally speaking more complex than Indicators, and can, for example, involve further refining Indicators or combining several Indicators into a single Feature. The simplest way to understand a `Feature` is that it's something that is not an `Indicator`, but where it is used as a so-called "independent variable". Features must be contributed to `loop.features`.

`Transforms` include all possible data transformations; everything goes here. Transforms must be contributed to `loop.transforms`.

`Labels` include all so-called "dependent variables" and their various manipulations (e.g. confidence gating). Labels must be contributed to `loop.labels`.

`Parameters` include all the parameters to be included in the experiment. These can include controls for Data, Indicators, Features, Transforms, Labels, and in the future, even Parameters themselves. Parameters are included in the respective Foundational SFD file.

## SFD Manifest Constituents

In simple terms, the following constituents can be included in the `manifest` of an SFD: 

- Manifest
- Reference Architecture
- Parameter
- Data
- Indicators
- Features
- Transforms
- Labels

The contributing modeller may or may not decide to make contributions to `Data`, `Indicators`, `Features`, `Transforms`, and/or `Labels`. They may decide to simply use those that are already available in Loop. 

Full details for working with `SFD Manifest` can be found in [`Experiment-Manifest` documentation](https://github.com/Vaquum/Loop/blob/main/docs/Experiment-Manifest.md)


## Preparation

Once a Reference Architecture is decided upon, certain requirements must be met: 

- A comprehensive literature review to understand what kind of `Data` works best  for the Reference Architecture
- A comprehensive literature review to understand what kind of `Indicators` works best  for the Reference Architecture
- A comprehensive literature review to understand what kind of `Features` works best  for the Reference Architecture
- A comprehensive literature review to understand what kind of `Transforms` works best  for the Reference Architecture
- A comprehensive literature review to understand what kind of `Labels` works best  for the Reference Architecture
- A comprehensive literature review to understand what kind of `Params` works best  for the Reference Architecture

**NOTE:** These are to be performed as separate research projects, not bundled into one.

### Research Constraints

- The scope of research requests is pinned down to: quantitative finance, day trading
- Three frontier model in their deep research mode is used for initial research
- At least three frontier models are used for cross-checking findings and for establishing consensus.

### Deliverables

- A thesis that summarize the findings of the research performed
- A a model card that has the following sections:
  - Reference Architecture description and links to literature
  - Indicators that were selected, with justification
  - Features that were selected, with justification
  - Transforms that were selected, with justification
  - Labels that were selected, with justification
  - Parameters that were selected, with justification
  - Future work which points to possible innovation ideas that had emerged during the research

## Implementation

Once the above has been comprehensively satisfied, after review of the deliverables, the contributing modeller moves to implementation following the guidelines laid out in: https://github.com/Vaquum/Loop/blob/main/docs/Experiment-Manifest.md.
