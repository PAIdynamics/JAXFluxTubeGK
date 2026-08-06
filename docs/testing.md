# Testing

The default pytest suite is standalone. It uses package-owned synthetic and
compact numerical contracts and does not inspect sibling source trees:

```bash
JAX_ENABLE_X64=1 .venv/bin/python -m pytest
```

Tests marked `external` exercise independently versioned validation or MHD
codes. They are deselected by default and require explicit checkout roots. Each
configured checkout is verified against the full commit in `dependencies.toml`;
a missing path or revision mismatch is an error, not a skip.

```bash
JAX_ENABLE_X64=1 .venv/bin/python -m pytest -m external \
  tests/test_analytic_geometry.py \
  tests/test_benchmark_validation.py \
  --gyaradax-root=../gyaradax \
  --gx-root=../gx

JAX_ENABLE_X64=1 .venv/bin/python -m pytest -m external \
  tests/test_benchmark_references.py \
  --gx-root=../gx \
  --desc-root=../DESC
```

The equivalent environment variables are
`OPTIMAL_FUSION_GYARADAX_ROOT`, `OPTIMAL_FUSION_GX_ROOT`,
`OPTIMAL_FUSION_DESC_ROOT`, `OPTIMAL_FUSION_STELLA_ROOT`, and
`OPTIMAL_FUSION_GKW_ROOT`. Prefer `--name-root=PATH` syntax so pytest does not
mistake a separated path value for an additional collection target.

Prepare pinned dependencies before integration testing when needed:

```bash
.venv/bin/python scripts/bootstrap_dependencies.py --profile validation --local-root ..
```

## Package smoke test

Build both distributions, install the wheel into a temporary isolated
environment, and import it from outside the checkout:

```bash
.venv/bin/python scripts/package_smoke_test.py --python 3.13
```

CI runs lint, the standalone suite, and this packaging check without sibling
repositories or external provider installations.

## Scientific claim ledger

Priority 5 confidence gaps are available as a compact machine-readable ledger:

```bash
.venv/bin/python examples/report_validation_confidence.py
```

The report keeps full GKW state-history parity, the growing multi-time GKW
velocity-slice error, and the low-`ky` Cyclone/GX branch-shape gap explicitly
open. `validation_claim_readiness` and `require_validation_claim_ready` map
each gap to the broad scientific claims it blocks. Independent stella evidence
supersedes the two GKW gaps only for the converged W7-X linear branch; it does
not promote a general GKW/Cyclone history-parity claim.

The kinetic-electron path has a separate algebraic preflight:

```bash
JAX_ENABLE_X64=1 .venv/bin/python examples/run_tem_physics_preflight.py
```

It exercises two kinetic species, charge neutrality, the electron thermal
streaming scale, kinetic quasineutrality, the coupled JIT residual, and the CFL
estimate at TEM-favorable gradients. A passing result deliberately reports
`external_growth_frequency_validated: false`; quantitative TEM validation
still requires a converged revision-pinned independent comparison.
