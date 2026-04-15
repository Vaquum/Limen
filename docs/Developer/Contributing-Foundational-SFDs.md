# Contributing Foundational SFDs

This guide covers how to propose and implement a new foundational SFD in Limen.

Foundational SFDs are Limen's reference-grade experiment templates. They are not one-off research scripts. A strong foundational SFD should be reusable, reviewable, and analytically valuable inside a large Limen scan.

## What A Foundational SFD Owns

A foundational SFD usually packages three things:

- a `params()` search space
- a manifest-driven experiment pipeline
- a reference-architecture model function

Canonical example:

- foundational SFD: `limen/sfd/foundational_sfd/logreg_binary.py`
- reference architecture: `limen/sfd/reference_architecture/logreg_binary.py`

The foundational SFD owns experiment design and parameter exposure. The reference architecture owns the train, predict, and evaluate logic for the model family itself.

## Before You Start

Do the research work first. A good foundational SFD proposal should answer:

- what problem or modeling thesis this SFD is meant to capture
- why this reference architecture is the right fit
- which indicators and features are justified
- how the target should be constructed
- which scaler or transform choices belong in the default design
- which parameters are worth exposing as a real search space

If these answers are still hand-wavy, the design is not ready.

## Design Rules

- Keep the SFD manifest-driven.
- Put experiment intelligence in `params()` and `manifest()`, not inside bespoke hidden code paths.
- Put model-specific fitting logic in the reference architecture.
- If a workflow improvement is reusable, contribute it as a shared Limen building block instead of hiding it inside one SFD.
- Expose only meaningful search dimensions. A parameter that does not materially change the experiment should not be in `params()`.

## Contribution Surface

A foundational SFD may compose these existing Limen building blocks:

- data access and selection
- optional bar formation
- indicators
- features
- target construction
- scalers
- transforms
- reference-architecture models

If a needed building block does not exist yet, add it in the right package first:

- `limen.data` for retrieval or bar-prep logic
- `limen.indicators` for low-level signal primitives
- `limen.features` for derived signals and target helpers
- `limen.transforms` for lightweight transform helpers
- `limen.scalers` for train-fitted preprocessing

## Gold-Standard Shape

A strong foundational SFD should look roughly like this:

```python
from limen.experiment import Manifest
from limen.data import HistoricalData
from limen.sfd.reference_architecture import your_model


def params():
    return {
        'lookback': [12, 24, 48],
        'threshold': [0.1, 0.2, 0.3],
        'scaler_type': ['linear', 'robust'],
    }


def manifest():
    return (
        Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
        )
        .set_test_data_source(
            method=HistoricalData.get_any_file,
            params={'file_path_or_url': HistoricalData.DEFAULT_TEST_FILE_URL},
        )
        .set_split_config(8, 1, 2)
        .add_indicator(...)
        .add_feature(...)
        .with_target(...)
        .set_scaler_from_params('scaler_type')
        .with_model(your_model)
    )
```

That is not the only valid shape, but it captures the important properties:

- split-first manifest execution
- explicit parameter exposure
- reusable shared building blocks
- no hidden manual glue

## Review Checklist

Before calling a foundational SFD ready for review, check all of the following:

- the thesis is clear and literature-backed or otherwise strongly justified
- `params()` exposes real search dimensions
- the manifest is readable and uses shared Limen primitives where possible
- target construction is split-safe
- scaler choice is appropriate for the feature surface
- the reference architecture is the right ownership boundary for model logic
- docs and docstrings are updated for any new public helpers added along the way
- the SFD can run inside Limen without custom manual intervention

## Anti-Patterns

Avoid these:

- baking reusable workflow logic into one SFD only
- turning every implementation detail into a parameter
- mixing model-family logic into the foundational SFD when it belongs in the reference architecture
- introducing a new helper in an arbitrary location just because the SFD needs it once
- writing the SFD around a one-off dataset or private workflow assumption

## Deliverables

For a serious foundational SFD contribution, expect to provide:

- a short research thesis or rationale
- the foundational SFD file
- any new shared helpers the design genuinely requires
- updated docs for any new public surfaces
- tests or validation appropriate to the added behavior

## Read Next

- [Experiment Manifest](../Experiment-Manifest.md)
- [Single File Decoder](../Single-File-Decoder.md)
- [Writing Docstrings](Writing-Docstrings.md)
