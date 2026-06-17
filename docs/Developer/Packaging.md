# Packaging

This page is the Limen-local packaging contract.

## Artifact Policy

The sdist is the complete source, docs, tests, fixtures, helper modules, packaging backend, and workflow source used to rebuild and inspect Limen. The wheel is the install artifact: runtime package files, `py.typed`, YAML templates, and public support/docs metadata are included; tests are intentionally excluded from the wheel.

## Dependency Policy

Default installs carry only the CLI, YAML, dataframe, metrics, and manifest runtime needed for a light first import. Heavy or platform-sensitive surfaces are extras: `data` for Arrow dataset IO, `boosting` for LightGBM/XGBoost, `indicators` for TA-Lib comparison tooling, `stats` for SciPy/statsmodels helpers, `tabpfn` for TabPFN, `all` for the full research stack, and `release` for release automation.

Every runtime dependency and extra must have a lower bound and an upper bound. `requirements/constraints.txt` mirrors the supported resolver envelope for local proof and CI reproduction.

## Reproducibility

Build distributions with a fixed `SOURCE_DATE_EPOCH`. `limen_build_backend.py` delegates to setuptools and then normalizes sdist tar and gzip metadata so repeated sdists hash identically when file contents match.

```bash
SOURCE_DATE_EPOCH=1704067200 python -m build
```

## Required Proof

Packaging changes must run `python scripts/package_audit.py`, `python -m build`, `twine check`, `check-wheel-contents`, `check-manifest`, and an install/import smoke test from the built wheel. Release evidence also includes dependency-license and SBOM artifacts, artifact hashes, GitHub build provenance attestations, and uploaded GitHub release assets.

## Footprint

The default package must import without loading LightGBM, XGBoost, TabPFN, PyArrow, SciPy, statsmodels, or TA-Lib. Dataset downloads and caches are runtime behavior, not install-time behavior; docs and release notes must keep that distinction explicit.
