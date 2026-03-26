# Contributing Foundational SFDs

## Background

Foundational SFDs are Limen's reference-grade experiment templates. Each one pairs:

- a `params()` search space
- a manifest-driven experiment pipeline
- a reference-architecture model function

This document focuses on contributing new Foundational SFDs to Limen.

## Terminology

A Foundational SFD has two layers:

1. the Foundational SFD itself
2. the underlying Reference Architecture

A canonical example:

- Foundational SFD: `limen/sfd/foundational_sfd/logreg_binary.py`
- Reference Architecture: `limen/sfd/reference_architecture/logreg_binary.py`

The Foundational SFD owns experiment design and parameter exposure. The Reference Architecture owns the train/predict/evaluate model logic.

## Design Principle

Foundational SFDs should package the strongest literature-backed version of an approach without baking bespoke workflow logic into the model layer.

That means:

- experiment intelligence belongs primarily in `params()` and `manifest()`
- model-specific training logic belongs in the Reference Architecture
- reusable workflow logic should be implemented as shared Limen building blocks, not hidden inside one Foundational SFD

## Minimal Requirements

- relies on a Reference Architecture
- is manifest-driven
- exposes a meaningful parameter space
- is runnable as part of a large Limen scan without custom manual glue

The practical test is simple: can someone run it inside Limen at scale and get analytically useful output?

## Contribution Surface

Foundational SFDs can compose the following building blocks:

- Data
- Indicators
- Features
- Scalers
- Transforms
- Target construction
- Parameters

### Data

Data contributions belong under `limen.data`.

### Indicators

Indicators belong under `limen.indicators`.

### Features

Features belong under `limen.features`.

### Scalers

Scalers belong under `limen.scalers`.

### Transforms

Transforms belong under `limen.transforms`.

### Target Construction

Target construction lives in the manifest through `.with_target()`, fitted parameter computation, and target transforms. In other words, Limen no longer has a separate labels module contribution surface.

If a new helper is needed for target construction, it should be contributed where it best fits today:

- `limen.features` for target-generating feature helpers such as `quantile_flag`
- `limen.transforms` for reusable transform helpers
- shared utility modules only when the logic is genuinely cross-cutting

### Parameters

Parameters live in the Foundational SFD file itself and may control:

- data selection
- indicators
- features
- target construction
- scalers
- model behavior

## Manifest Constituents

In practice, a Foundational SFD manifest may include:

- data source configuration
- split configuration
- optional bar formation
- indicators
- features
- target construction
- scaler selection
- model selection through `.with_model()`

Full manifest details are documented in [Experiment Manifest](../Experiment-Manifest.md).

## Research Expectations

Before implementation, the contributing modeller should understand:

- which data works best for the Reference Architecture
- which indicators and features are justified
- which scaling and transform choices are appropriate
- how the target should be constructed
- which parameters are worth exposing

These should be treated as explicit research questions, not as one bundled intuition dump.

## Deliverables

Expected deliverables for a serious Foundational SFD proposal:

- a thesis or research summary
- a model card covering:
  - Reference Architecture and literature
  - selected indicators
  - selected features
  - selected scalers
  - selected transforms
  - target construction
  - selected parameters
  - future work

## Implementation Expectations

Once the design is reviewed, implementation should follow the manifest patterns documented in [Experiment Manifest](../Experiment-Manifest.md).

The most important rule is this:

- do not hide general workflow logic inside one Foundational SFD

If a workflow intervention is reusable across architectures, it should be contributed back as a shared Limen building block instead.
