# Contributing to qcr

Thank you for your interest in qcr. This guide explains how to set up a development environment, the
conventions the code follows, and how changes are reviewed.

## Development setup

```bash
git clone https://github.com/arnavk23/qcr.git
cd qcr
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,experiments]"
pre-commit install
```

## Running the checks

All checks run through [tox](https://tox.wiki):

```bash
tox -e py          # unit tests (about one minute)
tox -e lint        # ruff lint + format check
tox -e docs        # Sphinx documentation
```

or directly: `pytest`, `ruff check .`, `ruff format --check .`.

## Pull requests

1. Open an issue first for anything larger than a small fix, so the approach can be agreed on.
2. Work on a branch; keep each pull request focused on one change.
3. Add or update tests. Optimiser changes must keep every equivalence mode verified
   (`test/test_general.py`). A new rewrite rule needs a test showing the output is equivalent to the input.
4. Add a release note for any user-facing change:
   ```bash
   reno new short-description-of-change
   ```
   then edit the generated file in `releasenotes/notes/`. Use the `features`, `issues`, `fixes`,
   `upgrade` or `deprecations` sections as appropriate.
5. Make sure `tox -e py,lint` passes; CI runs the same checks on Linux and macOS for Python 3.10-3.14.

## Code conventions

* Every source file starts with the Apache-2.0 header used throughout the repository.
* Circuits are lists of `qcr.Gate`; qubit 0 is the most significant tensor factor and a list
  `[g1, ..., gL]` implements `gL ... g1` (the conventions of the reference MATLAB code).
* "Exact" means exact: equivalence tests are linear in the operator error (norm distance, not
  `|tr(U†V)|`), use one tolerance (`qcr.general.TOL = 1e-10`) and never a relative tolerance.
* Experiments write raw data to `results/` and never edit the manuscripts by hand: numbers reach the LaTeX sources
  only through the `experiments/make_*` scripts.

## Reporting bugs

Use the [issue tracker](https://github.com/arnavk23/qcr/issues). Include the qcr, Qiskit and Python versions, a
minimal circuit that reproduces the problem, and what you expected to happen.
