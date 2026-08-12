# Usage Guide

This guide contains the operational detail intentionally omitted from the
project README. Commands are run from the repository root unless noted.

## Installation

JAXFluxTubeGK requires Python 3.11 or newer and uses `uv`:

```bash
uv sync --extra dev
```

Enable 64-bit JAX for numerical validation and production-style diagnostics:

```bash
export JAX_ENABLE_X64=1
```

The core package has no runtime dependency on an equilibrium or external
gyrokinetic code. To prepare the pinned DESC, GVEC, and VMEC++ providers:

```bash
uv run --no-sync python scripts/bootstrap_dependencies.py --profile mhd
```

Existing sibling checkouts can be reused:

```bash
uv run --no-sync python scripts/bootstrap_dependencies.py \
  --profile mhd --local-root ..
```

Use `.venv/bin/python` or `uv run --no-sync` after preparing optional
providers. A normal exact `uv sync` restores the standalone lock. See
[dependencies.md](dependencies.md) for pinned revisions, profiles, native
build requirements, and validation dependencies.

## Quick checks

```bash
uv run --no-sync pytest tests/test_import.py -q
JAX_ENABLE_X64=1 uv run --no-sync pytest
uv run --no-sync ruff check src tests examples scripts
```

The default test suite excludes explicitly marked external-code integrations.
See [testing.md](testing.md) for the complete test matrix.

## Geometry providers

Every provider returns the same versioned physical geometry contract. The
solver then derives its internal normalized coefficients:

```python
from jax_fluxtube_gk import (
    GeometryRequest,
    SyntheticGeometryProvider,
    internal_geometry_from_result,
    resolve_geometry,
)

request = GeometryRequest(
    configuration="standalone-cosine-B",
    radial_value=0.5,
    alpha=0.0,
    n_z=32,
)
result = resolve_geometry(SyntheticGeometryProvider(), request)
geometry = internal_geometry_from_result(result)
```

The provider can be replaced without changing solver construction:

- `DescGeometryProvider` consumes an in-memory DESC equilibrium or a path.
- `GvecGeometryProvider` consumes an in-memory GVEC state or parameter file.
- `VmecppGeometryProvider` consumes an output, input, `wout` path, or a named
  installed configuration such as `w7x-standard`.
- `EikGeometryProvider` and `StellaGeometryProvider` provide file-based
  interoperability paths.
- `PhysicalArrayGeometryProvider` adapts caller-owned field-line arrays.

VMEC++ and current GVEC evaluation are explicitly non-differentiable. DESC can
retain continuous in-memory array gradients only when the entire upstream path
is JAX-traceable; marking provider metadata as differentiable does not establish
end-to-end equilibrium-solve differentiation.

Geometry caches are optional, explicit, and must be outside the source tree.
Reloaded caches preserve schema and provenance but are non-differentiable. The
full units, normalization, signs, topology, and cache contract are documented
in [geometry_provider.md](geometry_provider.md).

## Geometry optimization notebooks

These notebooks explain and visualize the equilibrium-provider → local
gyrokinetic solver → objective → optimizer workflow:

- [DESC geometry optimization](../examples/desc_geometry_optimization.ipynb)
- [GVEC geometry optimization](../examples/gvec_geometry_optimization.ipynb)
- [VMEC++ geometry optimization](../examples/vmecpp_geometry_optimization.ipynb)

They use their live equilibrium libraries by default and do not consume
exported geometry files: DESC evaluates a named equilibrium, GVEC supplies an
in-memory state, and VMEC++ runs a named input in memory. File-backed or
synthetic modes require explicit selection. Each notebook includes a reduced
GK evaluation, geometry plots, an executable outer-loop surrogate, and a
clearly marked project-specific integration seam.

For a real VMEC++ finite-difference boundary loop:

```bash
JAX_ENABLE_X64=1 uv run --no-sync python \
  examples/vmecpp_w7x_design_loop.py \
  --iterations 1 --n-z 16 --n-vpar 2 --n-mu 2 --n-steps 1 \
  --output /tmp/jax-fluxtube-gk-vmecpp-w7x-design/smoke.json
```

Generated equilibria and optimization records should remain in caller-owned
scratch storage. This workflow demonstrates provider integration and outer
finite differences, not autodifferentiation through VMEC++ or unrestricted
production shape optimization.

## Reduced stellarator scan

The default end-to-end scan runs VMEC++'s named W7-X configuration and consumes
the equilibrium in memory:

```bash
uv run --extra dev --extra vmecpp python examples/run_stellarator_linear_scan.py
```

The output contains geometry audits, growth rates, mode structures,
convergence history, a quasilinear proxy, and the resolved run configuration.
Alternative geometry inputs include:

- `--geometry-source fixture` for the committed DESC D-shape fixture,
- `--geometry-source desc-path --desc-path ...`
- `--geometry-provider vmecpp --configuration w7x-standard`
- `--geometry-source eik --eik-reference ...`
- `--geometry-source stella-geometry --stella-geometry ...`

For a direct named VMEC++ W7-X scan after installing the MHD profile:

```bash
JAX_ENABLE_X64=1 uv run --no-sync python \
  examples/run_stellarator_linear_scan.py \
  --geometry-provider vmecpp \
  --configuration w7x-standard \
  --rho 0.8 \
  --output-dir runs/w7x_vmecpp_linear_scan
```

These are integration workflows. Production claims require the applicable
readiness and convergence gates.

## Direct solver API

Build grids and geometry once, then build reusable residual precomputation:

```python
import jax.numpy as jnp

from jax_fluxtube_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    build_fourier_grid,
    build_linear_residual_precompute,
    build_parallel_grid,
    build_s_alpha_geometry,
    build_velocity_grid,
    linear_residual,
)

velocity = build_velocity_grid(
    VelocityGridSpec(n_vpar=7, n_mu=5, vpar_max=2.0, mu_max=1.5)
)
parallel = build_parallel_grid(
    ParallelGridSpec(n_z=16, z_min=0.0, z_max=1.0, topology="periodic")
)
fourier = build_fourier_grid(
    FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.6, ky_values=(0.0, 0.4))
)
geometry = build_s_alpha_geometry(
    parallel, GeometryScalarParams(q=1.3, shat=0.7, eps=0.18)
)
ion = SpeciesParams(
    charge=1.0,
    mass=2.0,
    density=1.0,
    temperature=1.0,
    density_gradient=1.0,
    temperature_gradient=3.0,
)
electrons = AdiabaticElectronParams(density=1.0, temperature=1.0)

precompute = build_linear_residual_precompute(
    velocity,
    parallel,
    fourier,
    geometry,
    ion,
    electron_params=electrons,
)
shape = (
    velocity.vpar.size,
    velocity.mu.size,
    parallel.z.size,
    fourier.kx.size,
    fourier.ky.size,
)
state = jnp.zeros(shape, dtype=jnp.complex128)
rhs = linear_residual(state, precomputed=precompute)
```

Grid sizes, derivative backends, mode connectivity, selected modes, and file
I/O are static topology. Continuous geometry arrays, species and profile
parameters, residual terms, field solves, time steps, diagnostics, and
objectives form the differentiable path.

The compact public API and opt-in validation namespaces are described in
[api_stability.md](api_stability.md).

## Design objectives

`DesignObjectiveSpec` and `design_objective` provide named outputs for growth,
frequency, phase-invariant mode-shape mismatch, and quasilinear proxies.
Robust scans over fixed radii, field lines, and `ky` values support weighted
mean, hard worst case, and differentiable soft worst case aggregation.

`OptimizationTopologyContract` rejects accidental remeshing or provider
topology changes during one optimization trace. Gradient audits compare JAX
reverse-mode derivatives with central differences and report near-degenerate
mode branches. Schema-versioned checkpoints retain objective policy, topology,
parameters, provider provenance, dependency revisions, and iteration history.

See [optimization_integration.md](optimization_integration.md) for the complete
optimization contract and [performance_and_differentiability.md](performance_and_differentiability.md)
for JAX boundaries and performance guidance.

## Validation workflows

Summarize the reduced validation gates and regenerate paper artifacts:

```bash
uv run --extra dev python examples/run_validation_gates.py
uv run --extra dev python examples/generate_validation_gate_figures.py
```

Run the kinetic-electron TEM algebraic preflight:

```bash
JAX_ENABLE_X64=1 uv run --no-sync python examples/run_tem_physics_preflight.py
```

This is a field/algebra readiness check, not an external growth-rate claim.

Run the focused W7-X RHS comparisons:

```bash
JAX_ENABLE_X64=1 uv run --no-sync pytest \
  tests/test_w7x_ky03_rhs_model_balance.py \
  tests/test_w7x_stella_rhs_trace_comparison.py -q
```

Run the W7-X production-readiness ledger:

```bash
JAX_ENABLE_X64=1 uv run python scripts/run_w7x_production_readiness_gate.py
```

External solver fixtures, native collision traces, long-time W7-X comparisons,
and nonlinear acceptance campaigns are specialist validation workflows. Their
current results and remaining scientific blockers are tracked in
[STATUS.md](../STATUS.md), while exact commands and test selection live in
[testing.md](testing.md).

## Nonlinear heat-flux workflow

Generate a reduced trajectory in caller-owned storage:

```bash
JAX_ENABLE_X64=1 .venv/bin/python examples/run_nonlinear_heat_flux.py \
  --output /tmp/jax-fluxtube-gk-cyclone-nonlinear.json
```

Use `--require-stationary` only for acceptance runs. Long trajectories support
external checkpoints and restart:

```bash
JAX_ENABLE_X64=1 .venv/bin/python examples/run_nonlinear_heat_flux.py \
  --output /tmp/nonlinear-t60.json --final-time 60 \
  --checkpoint-output /scratch/nonlinear-t60.npz

JAX_ENABLE_X64=1 .venv/bin/python examples/run_nonlinear_heat_flux.py \
  --output /tmp/nonlinear-t60-t120.json --final-time 120 \
  --restart-from /scratch/nonlinear-t60.npz \
  --checkpoint-output /scratch/nonlinear-t120.npz
```

The nonlinear path includes adaptive CFL control and compact online
diagnostics. A useful scientific claim additionally requires stationarity,
resolution and domain ladders, independent initial-condition lineages, and an
independent reference comparison. Consult [STATUS.md](../STATUS.md) before
interpreting these runs.

## Documentation map

- [Geometry provider contract](geometry_provider.md)
- [Optimization integration](optimization_integration.md)
- [Dependencies](dependencies.md)
- [Testing](testing.md)
- [API stability](api_stability.md)
- [Performance and differentiability](performance_and_differentiability.md)
- [Velocity backend policy](velocity_backend_policy.md)
- [Fixture policy](fixture_policy.md)
- [Current status](../STATUS.md)
- [Prioritized work](../TODO.md)

Build the physics and numerics paper with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd tex/main.tex
```
