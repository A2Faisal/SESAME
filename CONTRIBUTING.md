# Contributing to SESAME

Thanks for your interest in contributing to SESAME, the Python toolbox behind the SESAME Human-Earth Atlas. This document explains how to report issues, propose changes, and get a development environment running.

## Reporting bugs and requesting features

Please use [GitHub Issues](https://github.com/A2Faisal/SESAME/issues) for both. When reporting a bug, include:

- The SESAME version (`pip show sesame-iesd`) and Python version
- A minimal, reproducible example (a short script or code snippet, ideally with a small sample input)
- The full error traceback, if there is one
- What you expected to happen versus what actually happened

For feature requests, describe the use case, not just the desired API. This project unifies datasets across several formats (CSV, TIFF, netCDF, SHP), so context on which format or workflow motivates the request is especially useful.

## Development setup

SESAME depends on GDAL, GEOS, and PROJ through `geopandas`, `rasterio`, and `cartopy`, which are easiest to install via conda:

```bash
conda create -n sesame_dev python=3.11
conda activate sesame_dev
conda install -c conda-forge gdal rasterio cartopy geopandas h5netcdf matplotlib

git clone https://github.com/A2Faisal/SESAME.git
cd SESAME
pip install -e .
pip install pytest
```

The project is managed with [Poetry](https://python-poetry.org/) (see `pyproject.toml`); `pip install -e .` works for local development, but if you have Poetry installed, `poetry install` will also set up the `dev` dependency group (currently `pytest`).

## Running the tests

Tests live in `test/test_core.py` and run against small fixture datasets in `test/data/`. From the repository root:

```bash
cd test
pytest test_core.py -v
```

This mirrors the GitHub Actions workflow in `.github/workflows/tests.yml`, which runs automatically on every push and pull request to `main`. Please make sure the full suite passes locally before opening a pull request, and add or update tests for any new function or bug fix, tests are how we catch regressions in the gridding and aggregation logic.

## Submitting changes

1. Fork the repository and create a branch from `main` (e.g. `fix/grid-2-table-nan-handling`).
2. Make your changes. Try to match the existing code style in the surrounding file rather than introducing a new one.
3. Add or update tests in `test/test_core.py` covering the change.
4. Update the README or the [software manual](https://a2faisal.github.io/SESAME/) if the change affects public behavior (function signatures, defaults, outputs).
5. Open a pull request against `main` with a clear description of what changed and why. Link any related issue.
6. A maintainer will review the PR; CI (GitHub Actions) must pass before merging.

By contributing, you agree that your contributions will be licensed under the project's [MIT License](https://github.com/A2Faisal/SESAME/blob/main/LICENSE).

## Questions

If you're unsure whether something is a bug, a missing feature, or expected behavior, open an issue and ask, that's what it's for. For anything not suited to a public issue, you can reach the maintainers at [abdullah-al.faisal@mail.mcgill.ca](mailto:abdullah-al.faisal@mail.mcgill.ca) or [maxwell.kaye@mail.mcgill.ca](mailto:maxwell.kaye@mail.mcgill.ca).
