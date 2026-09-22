# fdasrsf test suite

Pytest-based tests for the `fdasrsf` package.

## Layout

- `conftest.py` — shared fixtures (sample-data loaders resolved from the repo
  root, session-scoped aligned models) and the `requires_ext` helper that skips
  a test when a compiled C/Cython extension is not built.
- `test_*.py` — one module per area of the public API.

## Running

The compiled extensions must be built first (they are produced by
`pip install -e .`). Tests that reach an extension skip cleanly if it is
missing, but most of the suite depends on `optimum_reparamN2`.

```bash
pip install -e .

pytest              # fast tier only (the default; slow tests deselected)
pytest -m ''        # everything, including the slow/iterative/stochastic tier
pytest -m slow      # only the slow tier
```

## Tiers

- **Fast (default):** deterministic leaf functions plus the two golden
  integration checks (`fdawarp.srsf_align` amplitude variance and
  `fdacurve.karcher_mean` energy) on the shipped sample data.
- **`@pytest.mark.slow`:** iterative / stochastic paths (regression fits,
  bootstrap tolerance, k-means alignment, change-point detection). These use
  small inputs and few iterations so they still run quickly when selected.

The CI Linux job runs the full suite (`pytest --cov -m ''`) since all
extensions are built there.
