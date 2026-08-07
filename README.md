# stellarator-gk

Differentiable local flux-tube gyrokinetics for stellarator design studies.

This repository is a standalone JAX-first implementation of a linear
electrostatic flux-tube gyrokinetic solver. The physics and sign conventions
are aligned with GKW and Gyaradax. DESC, VMEC++, GVEC, GX, and stella are
separately installed geometry providers or validation tools, not vendored
runtime dependencies.

The current production milestone is a trusted linear W7-X stellarator run.  The
code can already run reduced stellarator scans, reduced optimization examples,
Rosenbluth-Hinton and Cyclone gates, and several geometry/code-to-code parity
checks. A conservative model-collision operator, a dealiased nonlinear ExB
path with adaptive CFL control, and independently parity-checked algebraic
`A_parallel` and coupled `phi`/`B_parallel` field solves, plus their linear RHS
coupling, now exist as development foundations. Full nonlinear turbulence
validation, production DESC shape optimization, converged production
Landau/Fokker--Planck inter-species collisions, and nonlinear turbulence
validation remain explicitly deferred. The pinned heavy-electron kinetic TEM
and production-grid electromagnetic benchmarks pass.

## Example Geometry

![W7-X VMEC stellarator geometry](figures/w7x_vmec_geometry.png)

Regenerate this figure from a user- or provider-supplied VMEC output:

```bash
uv run --no-sync python scripts/visualize_w7x_vmec_geometry.py \
  --vmec /path/to/wout_w7x.nc
```

The image is documentation output; the repository does not provide the W7-X
equilibrium file. The canonical runtime path loads VMEC++'s installed
`w7x-standard` configuration, runs it, and transforms its in-memory `wout`
result without writing an equilibrium file in this repository.

## Repository Map

```text
src/stellarator_gk/      Python package and public solver API
tests/                   Unit, validation, fixture, and regression tests
examples/                User-facing runnable workflows and diagnostics
scripts/                 Fixture generators, external-code prep, audit tools
fixtures/                Small numerical validation contracts
figures/                 Generated CSV/PDF result artifacts for the paper
docs/                    Short developer notes
tex/                     Physics, numerics, and project TeX sources
TODO.md                  Current prioritized backlog
STATUS.md                Current project status and active blocker
```

External MHD and gyrokinetic codes are optional providers and validation
references. Keep their checkouts outside this repository and pass their paths
explicitly to integration workflows; they are not runtime dependencies of the
core solver.

## Install

Use Python 3.11 or newer.  The project is configured for `uv`.

```bash
uv sync --extra dev
```

For stellarator geometry from MHD codes, prepare the pinned provider forks:

```bash
uv run --no-sync python scripts/bootstrap_dependencies.py --profile mhd
```

This fetches, builds, and installs VMEC++, DESC, and GVEC outside the tracked
source tree. Existing sibling clones can be reused without modifying them:

```bash
uv run --no-sync python scripts/bootstrap_dependencies.py \
  --profile mhd --local-root ..
```

GX, stella, GKW, and Gyaradax are validation dependencies and are prepared
separately with `--profile validation`. See
[`docs/dependencies.md`](docs/dependencies.md) for profiles, pinned revisions,
native build requirements, and dry-run/fetch-only options.

For commands that consume these providers, use `.venv/bin/python` or
`uv run --no-sync`; a normal exact `uv` sync intentionally restores the core
lock and removes the external profile.

Use `uv run --no-sync` after the initial sync when a command must not replace a
prepared provider environment. Run `uv sync --extra dev` explicitly when you
want to restore the standalone lock.

For numerical parity tests and production-style diagnostics, enable JAX x64:

```bash
export JAX_ENABLE_X64=1
```

## Quick Checks

Run the import smoke test:

```bash
uv run --no-sync pytest tests/test_import.py -q
```

Run the focused W7-X RHS comparison tests:

```bash
JAX_ENABLE_X64=1 uv run --no-sync pytest \
  tests/test_w7x_ky03_rhs_model_balance.py \
  tests/test_w7x_stella_rhs_trace_comparison.py -q
```

Run the full test suite when changing shared solver behavior:

```bash
JAX_ENABLE_X64=1 uv run --no-sync pytest
```

The default suite excludes explicitly marked external-code integrations and
does not require sibling repositories. To run those checks, pass revision-
checked roots explicitly; see [`docs/testing.md`](docs/testing.md).

Run linting:

```bash
uv run --no-sync ruff check src tests examples scripts
```

Run the kinetic-electron TEM algebraic preflight:

```bash
JAX_ENABLE_X64=1 uv run --no-sync python \
  examples/run_tem_physics_preflight.py
```

This is a multispecies/kinetic-field readiness check, not a quantitative TEM
growth-rate validation. The generated report explicitly leaves external
growth, frequency, and mode-structure parity open.

These are the same standalone boundaries exercised by GitHub Actions. The CI
job also builds an sdist and wheel, installs the wheel into a fresh environment,
and verifies that `stellarator_gk` imports without repository-relative files.

Build the paper:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd tex/main.tex
```

## Run a Reduced Stellarator Scan

The easiest end-to-end run is the reduced stellarator scan:

```bash
uv run --extra dev python examples/run_stellarator_linear_scan.py \
  --output-dir runs/dshape_linear_scan
```

By default this uses the committed DESC DSHAPE fixture
`fixtures/desc_geometry_dshape_rho05_alpha0.npz`.  The command writes:

```text
geometry_audit.json
geometry_audit.csv
ky_growth.csv
mode_structures.csv
convergence_history.csv
convergence_metadata.json
quasilinear_proxy.json
run_config.json
```

The scan can also read:

- `--geometry-source desc-path --desc-path ...` for a DESC equilibrium path,
- `--geometry-provider vmecpp --configuration w7x-standard` for the installed
  live W7-X VMEC++ design, with `--vmec-wout ...` as an interoperability path,
- `--geometry-source eik --eik-reference ...` for GX/GIST/GS2 eik tables,
- `--geometry-source stella-geometry --stella-geometry ...` for stella
  `.geometry` files.

These runs are useful integration checks.  They are not production
stellarator-optimization claims unless the readiness gates pass.

## Public Geometry Provider API

All geometry backends return the same versioned physical contract before the
solver derives its internal coefficients:

```python
from stellarator_gk import (
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

`DescGeometryProvider`, `VmecppGeometryProvider`, `GvecGeometryProvider`,
`GxEikGeometryProvider`, and `StellaGeometryProvider` can replace the synthetic
provider without changing solver construction. VMEC++ runs a packaged named
W7-X input and consumes `VmecOutput.wout` directly; GVEC consumes an in-memory
state or an explicit parameter file and normalizes its SI output to the same
minor-radius/edge-flux contract. File I/O and validation happen before JAX
tracing; continuous in-memory DESC arrays can retain gradients. Native VMEC++
and current GVEC evaluations are declared non-differentiable. Opt-in tests
cross-check direct W7-X geometry against stella, matched GX/GIST, and a common
GVEC/VMEC equilibrium.

Optional geometry caches are explicit and must be outside the source tree.
They preserve the schema and provenance but are non-differentiable after
reload. See [`docs/geometry_provider.md`](docs/geometry_provider.md) for the
complete units, signs, topology, validation, and caching contract.

## Differentiable Design Interface

`DesignObjectiveSpec` and `design_objective` provide a stable, named contract
for growth rate, real-frequency targets, phase-invariant complex mode-shape
penalties, and quasilinear proxies. Gradient audits compare JAX reverse-mode
derivatives with central differences and report near-degenerate `ky` branches.
`OptimizationTopologyContract` rejects reuse after a grid, mode-connectivity,
field-line-linking, or provider-topology change.

Fixed scans over multiple surfaces, field lines, and `ky` values can be reduced
with weighted-mean, hard-worst-case, or differentiable soft-worst-case
objectives. Schema-versioned checkpoints retain objective policies, sample
coordinates, topology fingerprints, parameters, provider and dependency
revisions, command, random seed, and iteration history. See
[`docs/optimization_integration.md`](docs/optimization_integration.md).

To exercise the real MHD boundary, prepare the MHD profile and run:

```bash
JAX_ENABLE_X64=1 uv run --no-sync python \
  examples/vmecpp_w7x_design_loop.py \
  --iterations 1 --n-z 16 --n-vpar 2 --n-mu 2 --n-steps 1 \
  --output /tmp/optimal-fusion-vmecpp-w7x-design/smoke.json
```

This obtains W7-X from VMEC++, performs fresh finite-difference equilibrium
solves, and keeps its record outside the repository. It demonstrates reduced
provider integration, not autodifferentiation through VMEC++ or production
full-boundary shape optimization.

## Common Workflows

### Validation Gates

Summarize the main reduced validation gates:

```bash
uv run --extra dev python examples/run_validation_gates.py
```

Regenerate validation CSV/PDF artifacts used by `tex/main.tex`:

```bash
uv run --extra dev python examples/generate_validation_gate_figures.py
```

### W7-X Reduced Benchmark

Run directly from the named VMEC++ design after preparing the MHD profile:

```bash
JAX_ENABLE_X64=1 uv run --no-sync python \
  examples/run_stellarator_linear_scan.py \
  --geometry-provider vmecpp \
  --configuration w7x-standard \
  --rho 0.8 \
  --output-dir runs/w7x_vmecpp_linear_scan
```

Generate a reduced W7-X benchmark from explicit GX/GIST geometry and input
paths:

```bash
JAX_ENABLE_X64=1 uv run --no-sync python \
  scripts/generate_w7x_itg_reduced_benchmark.py \
  --eik-reference /path/to/w7x.eik.out \
  --gx-input /path/to/itg_w7x_adiabatic_electrons.in \
  --output-dir /path/to/output
```

Run the W7-X production-readiness ledger:

```bash
JAX_ENABLE_X64=1 uv run python scripts/run_w7x_production_readiness_gate.py
```

The committed historical ledger remains conservative. For a new external run,
pass the generated solver/reference fixtures and timing artifact explicitly;
generated W7-X state stays outside the repository.

### Matched stella W7-X Path

The preferred external W7-X reference is stella.  The prepared run directory is:

```text
fixtures/stella_w7x_mode_structure_run/
```

The matched stella fixture has already been exported to:

```text
fixtures/w7x_itg_external_mode_structure_fixture.csv
```

Compare a solver mode-structure fixture against the stella reference:

```bash
JAX_ENABLE_X64=1 uv run python examples/run_w7x_mode_structure_gate.py \
  --observed-fixture fixtures/w7x_itg_stella_matched_time_ladder/runs/time_200/mode_structures.csv \
  --reference-fixture fixtures/w7x_itg_external_mode_structure_fixture.csv \
  --ky-values 0.1,0.2,0.3 \
  --resample-reference-to-observed-z
```

Current status: the converged `t=500`, `ky=0.3` production branch passes. The
source-matched path uses a 32x8 midpoint/Gauss-Laguerre velocity grid, `dt=0.1`,
SSP RK3 explicit stages, one full stella-cubic mirror characteristic, one full
implicit response, and `--initial-condition stella_maxwellian`. Its
growth/frequency/phase-aligned-profile errors are approximately
`0.00032/0.00218/0.01084`, all below 0.02. The initializer is analytic and no
W7-X distribution is stored. Lower-`ky` frequencies remain diagnostic because
stella's own omega window is unconverged even at `t=1000`.
Reference export rejects half-window omega changes above 0.02 by default;
`--allow-unconverged-omega` is diagnostic-only.

Run guarded timing of the validated algorithm only after supplying a passing
external gate ledger:

```bash
JAX_ENABLE_X64=1 uv run python scripts/run_w7x_production_cpu_timing.py \
  --preset stella-production \
  --stella-geometry /external/run/w7x.geometry \
  --readiness-gate /external/run/readiness_gate.json \
  --output /external/run/production_cpu_timing.json \
  --n-windows 100 --require-pass
```

The scan output used by the timer is created in disposable system scratch and
removed automatically.

### W7-X `ky=0.3` RHS Trace Work

Regenerate the solver-side W7-X `ky=0.3` RHS/model balance:

```bash
JAX_ENABLE_X64=1 uv run python scripts/audit_w7x_ky03_rhs_model_balance.py \
  --output-dir fixtures/w7x_ky03_rhs_model_balance
```

For direct array comparison, emit the single selected case to external scratch
storage (large numerical traces are deliberately refused inside the repository):

```bash
JAX_ENABLE_X64=1 uv run python scripts/audit_w7x_ky03_rhs_model_balance.py \
  --array-output /tmp/stellarator_gk_w7x_ky03_solver_arrays.npz
```

The archive uses `(z, vpar, mu)` phase-space order and contains coordinates,
quadrature weights, the distribution, potential, individual RHS terms, total
RHS, quasineutrality numerator/denominator, and accumulated log normalization.
For unlike velocity grids, the comparison adapter performs separable linear
complex interpolation onto a common target grid, rejects extrapolation, and
uses that target grid's `w_z*w_vpar*w_mu` quadrature for error norms.
For native solver coupling, `build_velocity_grid_from_nodes(...)` accepts
provider-supplied monotone `vpar`/`mu` nodes and weights. A nonseparable or
geometry-dependent integration measure can be passed as
`phase_space_measure[z,vpar,mu]` to `build_linear_residual_precompute(...)`;
built-in tensor-product grids retain their existing `B*w_vpar*w_mu` measure.
To match the current stella trace time (`code_time=199.9`) and run the partial
weighted comparison:

```bash
JAX_ENABLE_X64=1 uv run python scripts/audit_w7x_ky03_rhs_model_balance.py \
  --n-windows 1999 \
  --output-dir /tmp/stellarator_gk_w7x_ky03_solver_balance_t1999 \
  --array-output /tmp/stellarator_gk_w7x_ky03_solver_arrays_t1999.npz
uv run python scripts/compare_w7x_stella_rhs_trace_to_solver_balance.py \
  --require-raw-trace \
  --solver-balance-dir /tmp/stellarator_gk_w7x_ky03_solver_balance_t1999 \
  --solver-array /tmp/stellarator_gk_w7x_ky03_solver_arrays_t1999.npz \
  --output-dir /tmp/stellarator_gk_w7x_ky03_comparison_t1999 \
  --array-comparison-output /tmp/stellarator_gk_w7x_ky03_array_comparison_t1999.csv
```

The v3 trace labels all three RHS calls and includes stella-side quasineutrality
numerator/denominator and native normalization records. The complete weighted
array contract is ready, but numerical parity currently fails.

The boundary discriminator is available as `--case stella_open_16x8`. It uses
the solver's existing open-chain GKW-style upwind operator to test stella's
resolved unconnected boundary contract; it is a discriminator, not a claim
that the two discretizations are identical.

Replay the external stella states directly through the solver operators:

```bash
JAX_ENABLE_X64=1 uv run python scripts/replay_w7x_stella_state_in_solver.py \
  --trace /tmp/stellarator_gk_stella_w7x_rhs_trace_v3/run/stellarator_gk_w7x_ky03_rhs_trace.dat
```

This writes only compact results. It compares periodic, open-chain, and
explicitly labeled source-derived discriminators on grids contained in the
trace domain, including a provider-native 32×8 case with arbitrary velocity
nodes and a z-dependent phase-space measure. That acceptance case passes the
0.1 same-state RHS tolerance with a maximum relative L2 error of about 0.04393.
Production geometry schema v2 preserves the confirmed `flux_fac` drive scale,
separate grad-B and curvature drift components, and the verified mirror-force
orientation. Traced stella coefficients and source-specific stencils remain
diagnostic replay inputs, not production defaults.

Prepare and run the patched stella RHS trace in a scratch tree:

```bash
uv run --no-sync python scripts/prepare_stella_w7x_rhs_trace_run.py \
  --stella-source /path/to/revision-pinned/stella \
  --vmec-file /path/to/wout_w7x.nc \
  --overwrite \
  --output-root /tmp/stellarator_gk_stella_w7x_rhs_trace
bash /tmp/stellarator_gk_stella_w7x_rhs_trace/build_stella_rhs_trace.sh
bash /tmp/stellarator_gk_stella_w7x_rhs_trace/run_stella_rhs_trace.sh
```

Summarize and compare the trace:

```bash
uv run python scripts/summarize_stella_w7x_rhs_trace.py \
  /tmp/stellarator_gk_stella_w7x_rhs_trace/run/stellarator_gk_w7x_ky03_rhs_trace.dat \
  --stella-source /path/to/revision-pinned/stella \
  --stella-executable \
    /tmp/stellarator_gk_stella_w7x_rhs_trace/stella/COMPILATION/build_cmake/COMPILATION/stella \
  --output fixtures/w7x_ky03_stella_rhs_trace_summary/rhs_trace_summary.json
uv run python scripts/compare_w7x_stella_rhs_trace_to_solver_balance.py \
  --require-raw-trace
```

The committed comparator remains a compact scalar diagnostic. Direct weighted
array parity uses the external raw stella trace and the opt-in solver archive;
neither large array artifact belongs in Git.

### External GX W7-X Cross-Check

GX is a secondary W7-X reference path because it needs a suitable GPU/CUDA
environment.  Package the prepared handoff bundle with:

```bash
uv run python scripts/package_w7x_external_reference_bundle.py \
  --output fixtures/gx_w7x_mode_structure_run/w7x_external_reference_bundle.tar.gz
```

On a GX-capable machine:

```bash
GX_EXECUTABLE=/path/to/gx \
  bash fixtures/gx_w7x_mode_structure_run/run_external_reference.sh
```

After returned GX outputs are copied back:

```bash
bash fixtures/gx_w7x_mode_structure_run/ingest_returned_outputs.sh \
  --copy-outputs --resample-reference-to-observed-z
```

### External Gyaradax TEM Reference

The kinetic-electron TEM target is generated on demand from an explicit,
revision-pinned Gyaradax checkout and written to caller-selected scratch
storage:

```bash
PYTHONPATH=/path/to/gyaradax uv run --extra reference python \
  scripts/run_gyaradax_tem_reference.py \
  --gyaradax-root /path/to/gyaradax \
  --expected-revision 8d9dc2d205e8993ae9e43e6e1e82ec1ea2875234 \
  --output /tmp/optimal-fusion-gyaradax-tem-ky07.json
```

The exact notebook case produces `gamma=0.66370834` and `omega=-1.02976757`
at GKW `kthrho=0.7`. For this s-alpha case, GKW/Gyaradax divides by
`kthnorm=q/(2*pi*eps)`, giving the solver coordinate `krho=0.56548668`; the
local matched profile uses that internal value directly. The output records
both conventions plus provenance, case parameters, timestep range, and window
histories; it is not stored in the repository.

The same producer can drive an electromagnetic resolution ladder without
committing generated references:

```bash
JAX_ENABLE_X64=1 uv run --extra reference python \
  scripts/run_gyaradax_em_resolution_ladder.py \
  --gyaradax-root /path/to/gyaradax \
  --expected-revision 8d9dc2d205e8993ae9e43e6e1e82ec1ea2875234 \
  --output-dir /tmp/optimal-fusion-em-ladder
```

The default `8x8x4`, `12x12x6`, and `16x16x8` rungs enforce independent
parity at every rung and 5% growth/frequency convergence between the finest
two. Add `--profile production` for the declared `16x16x8`, `24x24x12`, and
`32x32x16` confirmation ladder. Custom rungs can be supplied by repeating
`--resolution NZxNVPARxNMU`; generated references and summaries remain in the
caller-selected output directory.

## Using the Python API

The examples are the best starting point.  For direct API use, build grids and
geometry first, then build a reusable residual precompute object.

```python
import jax.numpy as jnp

from stellarator_gk import (
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

state_shape = (
    velocity.vpar.size,
    velocity.mu.size,
    parallel.z.size,
    fourier.kx.size,
    fourier.ky.size,
)
state = jnp.zeros(state_shape, dtype=jnp.complex128)
rhs = linear_residual(state, precomputed=precompute)
```

Important API rule: grid sizes, derivative backends, mode connectivity, and file
I/O are static topology.  Continuous geometry arrays, species/profile
parameters, RHS terms, field solves, time steps, diagnostics, and objectives are
the differentiable path.

For finite-difference Fokker--Planck collisions,
`collision_conservation_model="xu_species_local"` reproduces the pinned
GKW/Gyaradax local correction. The experimental `"pairwise_exchange"` option
retains every ordered target/background stencil and conserves density plus
combined momentum and energy independently for each unordered species pair.
The distinct `"reciprocal_exchange"` option makes each target's low-rank
momentum/energy response depend explicitly on its collision partner. It is
JIT-compatible and differentiable, conserves every pair to roundoff, and uses
a dense-operator-checked conservative CFL bound. This establishes reciprocal
software dataflow, not stella/Landau coefficient parity; production collision
claims remain blocked.

Run the paired native-stella field-particle discriminator in caller-owned
scratch storage after preparing the pinned validation dependency:

```bash
python3 scripts/bootstrap_dependencies.py --dependency stella \
  --local-root .. --skip-project
.venv/bin/python scripts/run_stella_collision_field_particle_discriminator.py \
  --output-dir /tmp/optimal-fusion-stella-collisions \
  --stella-executable .dependencies/bin/stella \
  --stella-source ../stella
```

The report verifies identical initial diagnostics and measures the one-step
field-particle effect. It records the revision from the stella source checkout;
the executable's embedded NetCDF version is informational because some
out-of-tree builds resolve the enclosing repository instead. This discriminator
does not replace the remaining signed coefficient/action parity test.

For the signed native action, prepare and build an instrumented copy in scratch
storage; the dependency checkout is never edited:

```bash
.venv/bin/python scripts/prepare_stella_collision_field_particle_trace_run.py \
  --stella-source ../stella \
  --output-root /tmp/optimal-fusion-stella-collision-trace --overwrite
bash /tmp/optimal-fusion-stella-collision-trace/build_stella_collision_trace.sh
bash /tmp/optimal-fusion-stella-collision-trace/run_stella_collision_trace.sh
.venv/bin/python scripts/summarize_stella_collision_field_particle_trace.py \
  /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_field_particle_trace.dat \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --output /tmp/optimal-fusion-stella-collision-trace/trace-summary.json
.venv/bin/python scripts/summarize_stella_collision_field_particle_components.py \
  --components \
    /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_field_particle_components.dat \
  --aggregate \
    /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_field_particle_trace.dat \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --output /tmp/optimal-fusion-stella-collision-trace/component-summary.json
JAX_ENABLE_X64=1 .venv/bin/python \
  scripts/summarize_stella_collision_field_particle_factors.py \
  --factors \
    /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_field_particle_factors.dat \
  --aggregate \
    /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_field_particle_trace.dat \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --output /tmp/optimal-fusion-stella-collision-trace/factor-summary.json
JAX_ENABLE_X64=1 .venv/bin/python \
  scripts/summarize_stella_collision_field_particle_primitives.py \
  --primitives \
    /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_field_particle_primitives.dat \
  --quadrature \
    /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_velocity_quadrature.dat \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --output /tmp/optimal-fusion-stella-collision-trace/primitive-summary.json
JAX_ENABLE_X64=1 .venv/bin/python \
  scripts/summarize_stella_collision_field_particle_drivers.py \
  --drivers \
    /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_field_particle_drivers.dat \
  --primitives \
    /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_field_particle_primitives.dat \
  --quadrature \
    /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_velocity_quadrature.dat \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --output /tmp/optimal-fusion-stella-collision-trace/driver-summary.json
```

This traces both the aggregate signed field-particle RHS and all eight
`(l,m,j)` contributions before stella's final implicit differential inversion.
The component validator fails unless their signed sum reconstructs the
aggregate action. It also separates every contribution into its complex scalar
response and real target-space basis. Those factors replay through the local
JAX low-rank kernel at roundoff. The primitive validator separately checks
collision frequency, spherical-harmonic normalization, associated Legendre
factor, gyroaverage, mass scaling, `Delta_j`, and sign. The local response
builder matches all 44,928 native rows at `1.84e-14` relative L2, while its
analytic `Delta_0` matches at `4.25e-13`. The recursive `Delta_j` builder
consumes the native velocity-measure
contract and matches every traced `j=0,1` value at `7.84e-13` scaled L2. The
normalized driver is independently assembled from the
reversed-pair recurrence, Maxwellian, background gyroaverage, velocity measure,
and pairwise normalization; it matches 44,928 native coefficients at
`3.07e-11` scaled L2. The public assembly builder combines these local driver
and response tensors into the differentiable low-rank precompute. A second
state can be generated without rebuilding the instrumented executable:

```bash
.venv/bin/python scripts/run_stella_collision_trace_state.py \
  --executable \
    /tmp/optimal-fusion-stella-collision-trace/stella/COMPILATION/stella \
  --output-dir /tmp/optimal-fusion-stella-collision-second-state \
  --initial-amplitude 0.017 --initial-width 0.7
```

That state changes the native aggregate collision RHS to L2 `2.46052e-3` and
replays through the local solved-psi factor kernel at `1.39e-13` relative L2;
its driver and primitive coefficients are byte-identical to the first state.
`implicit_laguerre_legendre_collision(...)` combines a supplied
`I-dt*C_test` matrix with the local low-rank coefficients using the same
inhomogeneous inversion and response-system ordering as stella. Its Woodbury
form matches an independently materialized dense backward-Euler solve at
roundoff and remains differentiable. The remaining collision acceptance
requirement is construction and native parity of stella's exact differential
test-particle matrix.

`build_fokker_planck_test_particle_matrix(...)` materializes the package's
independently Gyaradax-matched nine-point stencil as `I-dt*C_test` with the
batch layout expected by the implicit solve. Matrix application and the
original stencil action agree at roundoff. This supplies a complete standalone
implicit collision path; a stricter same-executable stella matrix comparison
and production-residual selection remain before claiming full stella parity.

The same scratch patch now exports stella's unfactorized band matrix before
LAPACK factorization and its final collision state. Validate the matrix
structure and replay the complete native implicit update with:

```bash
.venv/bin/python scripts/summarize_stella_collision_test_particle_matrix.py \
  /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_test_particle_matrix.dat \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --output /tmp/optimal-fusion-stella-collision-trace/test-particle-matrix.json

.venv/bin/python scripts/replay_stella_implicit_collision_state.py \
  --matrix /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_test_particle_matrix.dat \
  --field-particle /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_field_particle_trace.dat \
  --final-state /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_final_state.dat \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --output /tmp/optimal-fusion-stella-collision-trace/implicit-collision-replay.json
```

For the pinned compact case, all 234 matrices are real `12x12` operators with
bandwidth 3. The nonzero-`k_perp` correction is purely diagonal and linear in
`k_perp^2` to `6.44e-14`. The reconstructed final state agrees with stella at
`1.64e-16` relative L2. This validates storage and implicit ordering;
independent local construction of stella's differential coefficients remains
required.

The pair-resolved runner also retains each channel's test-particle matrix. It
fails unless identity plus the four isolated departures reconstructs the full
matrix. The pinned case closes at `3.77e-17` relative L2; the electron-electron
and electron-ion blocks dominate the remaining coefficient target.

The analytic coefficients beneath that matrix can be validated separately:

```bash
JAX_ENABLE_X64=1 .venv/bin/python \
  scripts/summarize_stella_collision_test_particle_primitives.py \
  /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_test_particle_primitives.dat \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --output /tmp/optimal-fusion-stella-collision-trace/test-particle-primitives.json
```

`build_stella_test_particle_primitives(...)` independently constructs speed,
Maxwellian, `nupa`, `nuD`, and `nux` arrays from velocity nodes, magnetic field,
species masses, and pair frequencies. All 624 native rows agree at relative L2
errors below `2e-16`. The remaining differential-matrix gap is confined to
finite-difference interior/boundary assembly and the diagonal gyro term.

`build_stella_test_particle_gyro_diagonal(...)` closes the latter term. The
local result matches all 208 nonzero-`k_perp` native matrix differences with
`6.65e-14` maximum absolute error. The remaining matrix gap is only stella's
zero-`k_perp` finite-difference interior and boundary coefficient generation.

The patch also exports stella's pair-resolved lower, diagonal, and upper
velocity blocks. Validate their independent JAX packing with:

```bash
JAX_ENABLE_X64=1 .venv/bin/python \
  scripts/summarize_stella_collision_test_particle_blocks.py \
  --blocks /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_test_particle_blocks.dat \
  --matrix /tmp/optimal-fusion-stella-collision-trace/run/stellarator_gk_collision_test_particle_matrix.dat \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --output /tmp/optimal-fusion-stella-collision-trace/test-particle-blocks.json
```

`assemble_stella_test_particle_blocks(...)` sums background channels, packs
the `aa`/`bb`/`cc` blocks in stella's `vpar*nmu+mu` ordering, and adds the
identity. All 1,248 traced blocks reconstruct the 26 zero-`k_perp` matrices
exactly (`0.0` maximum and relative-L2 error). Matrix layout and packing are
closed; the remaining task is to construct the interior and boundary block
coefficients locally from the validated primitives and velocity grid.

The native coefficient target can be split without modifying the pinned source
by running identical cases with stella's `mu_operator` enabled and disabled:

```bash
.venv/bin/python scripts/run_stella_collision_block_decomposition.py \
  --executable /tmp/optimal-fusion-stella-collision-trace/stella/COMPILATION/build_cmake/COMPILATION/stella \
  --output-dir /tmp/optimal-fusion-stella-block-decomposition \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --overwrite
```

The two operator paths reconstruct the full 1,248-row block trace exactly. The
`mu_operator` path has Frobenius norm `32.2503` (`88.20%` of the full-block
norm) and is almost entirely diagonal-block work (`32.2473`). The remaining
`vpar` path has norm `14.4030`; its lower and upper blocks each have norm
`10.0501`. Because this compact gate has only two mu nodes, every mu row uses a
boundary formula. The first local coefficient target is therefore the mu
boundary closure, starting with the electron-ion and electron-electron pairs.
The runner also sets `nuxfac=0` with both `mu_operator` states, producing an
exact four-case factorial split. Pure mu diffusion has norm `32.2473`
(`88.19%` of the full norm) and appears exclusively in the diagonal `bb`
blocks. The mixed term nested in the mu path is only `0.444517`. Pure vpar
diffusion has norm `0.0846490`, while the vpar-path mixed derivative has norm
`14.4028`. The implementation order is therefore pure two-node mu-boundary
diffusion, vpar-path mixed derivatives, then the two small residual branches.

`build_stella_two_mu_diffusion_blocks(...)` now implements that first slice as
a JIT-compatible, differentiable JAX kernel. It evaluates the ordered half-mu
collision frequencies and stella's default two ghost-cell closures, returning
the pair-resolved `bb` blocks. Validate it with:

```bash
JAX_ENABLE_X64=1 .venv/bin/python \
  scripts/summarize_stella_collision_two_mu_diffusion.py \
  --primitives /tmp/optimal-fusion-stella-block-decomposition/stellarator_gk_collision_test_particle_primitives.dat \
  --no-mixed-full /tmp/optimal-fusion-stella-block-decomposition/collision_blocks_no_mixed_full.dat \
  --no-mixed-vpar /tmp/optimal-fusion-stella-block-decomposition/collision_blocks_no_mixed_vpar.dat \
  --vpar /tmp/optimal-fusion-stella-block-decomposition/collision_blocks_vpar.dat \
  --full /tmp/optimal-fusion-stella-block-decomposition/collision_blocks_full.dat \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --output /tmp/optimal-fusion-stella-block-decomposition/two-mu.json
```

All 1,248 native block rows agree at `1.10e-15` relative L2 and `7.99e-15`
maximum absolute error. The remaining compact-case coefficient target is led
by the vpar-path mixed derivative; general grids additionally require the
interior-mu formulas.

Passing the full four-case factorial traces also validates
`build_stella_two_mu_vpar_mixed_blocks(...)`. The constructor covers both vpar
ghost boundaries and every interior lower/upper block. It matches the native
mixed-vpar factorial component at `9.11e-16` relative L2 and `2.66e-15`
maximum absolute error.

`build_stella_vpar_diffusion_blocks(...)` closes the pure-vpar half-grid flux
for arbitrary mu-node counts. On the pinned trace it agrees at `1.35e-15`
relative L2 and `1.47e-17` maximum absolute error.

`build_stella_two_mu_mixed_blocks(...)` closes the final compact component.
Its mu-path mixed blocks agree at `2.80e-15` relative L2 and `4.32e-16`
maximum absolute error. Summing all four independently constructed components
reproduces the complete native block trace at `1.08e-15` relative L2 and
`7.99e-15` maximum absolute error. General `nmu>2` interior-mu formulas remain
the next portability requirement.

`build_stella_mu_diffusion_blocks(...)` extends the dominant pure-mu branch to
nonuniform general mu grids, including the interior three-point formulas. A
fresh pinned `nmu=4` factorial trace exercises two interior rows and matches at
`1.42e-15` relative L2 and `1.53e-16` maximum absolute error. The scratch
producer accepts `--nmu` and `--nvgrid` so larger-grid parity is reproducible
without storing native artifacts.

The companion `build_stella_vpar_mixed_blocks(...)` and
`build_stella_mu_mixed_blocks(...)` constructors extend both mixed derivatives
to the same nonuniform grid, including interior and ghost-boundary formulas.
On the 4,992-row `nmu=4` trace their isolated errors are `1.30e-15` and
`6.48e-16` relative L2. All four general-grid components together reproduce
the complete native blocks at `9.62e-16` relative L2 and `1.80e-16` maximum
absolute error.

`build_stella_test_particle_matrix(...)` now packages those pair-resolved
blocks, the implicit identity, and the validated mode-dependent gyro diagonal
into the global species/velocity matrix used by the Woodbury field-particle
solve. Its zero-`kperp` production layout matches the pinned `nmu=4` native
matrix at `6.39e-17` relative L2 and `3.33e-16` maximum absolute error.
`build_stella_implicit_collision_precompute(...)` constructs the complete
test/field-particle backend, and `stella_implicit_collision_step(...)` selects
it as a backward-Euler collision substep through the fixed-step split driver’s
`collision_step_fn` hook.

The same patched executable can generate pair-resolved native targets using
stella's four collision-frequency knobs:

```bash
.venv/bin/python scripts/run_stella_collision_channel_traces.py \
  --output-dir /tmp/optimal-fusion-stella-collision-channels \
  --patched-stella-executable \
    /tmp/optimal-fusion-stella-collision-trace/stella/COMPILATION/build_cmake/COMPILATION/stella \
  --expected-revision 564ca09b89904c231421c17c00068a9362061278 \
  --overwrite
```

The report requires identical input states and retains separate ion-ion,
ion-electron, electron-electron, and electron-ion signed traces, including all
eight Laguerre--Legendre contributions for every channel.

`gyrokinetic_heat_response(...)` supplies the collocation-space
`J0 T_s (E_s-3/2) f_s` velocity moment used with `radial_flux_spectrum(...)`.
This makes nonlinear heat-flux histories measurable from evolved states; it
does not by itself establish stationarity or external turbulence parity.

GX nonlinear heat-flux output can be converted into the package's compact
external-reference contract without copying the NetCDF file into this
repository:

```bash
.venv/bin/python scripts/prepare_gx_nonlinear_heat_flux_run.py \
  --gx-root /path/to/gx \
  --output-dir /scratch/gx-cyclone-nonlinear \
  --gx-executable /path/to/gx/bin/gx

# Run the printed GX command on a CUDA-capable machine, then run its printed
# summarization command.

.venv/bin/python scripts/summarize_gx_nonlinear_heat_flux.py \
  --gx-root /path/to/gx \
  --expected-revision bc2fe5523c23e3d0198181a3e3b7c8a482e25ba5 \
  --run-manifest /scratch/gx-cyclone-nonlinear/gx_nonlinear_run.json \
  --netcdf /scratch/gx-cyclone-nonlinear/optimal_fusion_cyclone_nonlinear.nc \
  --output /tmp/gx-cyclone-nonlinear-heat-flux.json
```

The preparation helper starts from the pinned sibling GX input, writes the
case-matched s-alpha, electrostatic, linked-boundary input outside this
repository, and records its SHA-256, revision, physics/box contract, run
command, and summarization command. The summarizer rejects a changed input,
revision, or NetCDF path. The final campaign also compares the GX case contract
to the finest local rung, so an unrelated stationary GX run cannot satisfy the
independent parity gate. The helper explicitly zeros GX species collisions and
sets a large step ceiling so the declared `t_max=500` controls termination.

Generate the matching local reduced trajectory in caller-owned storage with:

```bash
JAX_ENABLE_X64=1 .venv/bin/python examples/run_nonlinear_heat_flux.py \
  --output /tmp/optimal-fusion-cyclone-nonlinear.json \
  --flux-moment gx_total_energy
```

The local producer makes the GX profile conversion explicit: its default
`fprim=0.8` and `tprim=2.49` are multiplied by `Rmaj/Lref=2.77778` before they
enter the local residual as `R/Ln` and `R/LT`. All three inputs and both derived
gradients are recorded in the JSON report.

For GX parity, `--flux-moment gx_total_energy` evaluates the same gyroaveraged
total-energy moment used by pinned GX revision
`bc2fe5523c23e3d0198181a3e3b7c8a482e25ba5`. In s-alpha geometry GX has
`grho=1`, so its flux weight is the same normalized Jacobian average used
locally; the local nonzero-`ky` Parseval factor also matches GX's explicit
factor of two. Reports from this mode therefore use `gx_Q_over_Q_GB` directly.
The default `nonadvective_heat` mode retains the historical
`T_s(E_s-3/2)f_s` diagnostic and `optimal_fusion_native` label; the two differ
by `3/2 T_s` times the particle flux and must not be silently interchanged.

The default random perturbation seeds only nonzonal modes. This avoids a large
artificial `ky=0` potential from the weak zonal polarization denominator and
lets zonal flows arise through nonlinear mode transfer. Use
`--initial-zonal-fraction` only for a controlled zonal-seed experiment.

`--collision-frequency` enables the package's density/momentum/energy-
conserving BGK model and includes its stiffness in adaptive CFL control. It is
off by default because it is a numerical discriminator, not a replacement for
matching GX's hypercollision controls or for the open Landau collision gate.

The local nonlinear producer now defaults to a cell-centered finite-difference
parallel grid, shear-consistent `kx` spacing, and twist-and-shift boundary
connectivity. The shift scales with the `ky/ky_min` mode index. The former
independent periodic-chain configuration remains available explicitly as
`--parallel-boundary-model periodic_chains` for historical discrimination;
reports from that profile must not be used to claim GX nonlinear parity.

Add `--require-stationary` only for acceptance runs; short smoke trajectories
are expected to report `stationary=false`. The stationarity amplitude guard is
applied to the heat-carrying nonzonal potential rather than the total field;
the report also includes total, nonzonal, and per-`ky` initial/final RMS values
so zonal growth cannot conceal decay of the turbulent spectrum.

Long adaptive runs can write and resume caller-owned state without committing
large artifacts:

```bash
JAX_ENABLE_X64=1 .venv/bin/python examples/run_nonlinear_heat_flux.py \
  --output /tmp/nonlinear-t60.json --final-time 60 \
  --checkpoint-output /scratch/nonlinear-t60.npz

JAX_ENABLE_X64=1 .venv/bin/python examples/run_nonlinear_heat_flux.py \
  --output /tmp/nonlinear-t60-t120.json --final-time 120 \
  --restart-from /scratch/nonlinear-t60.npz \
  --checkpoint-output /scratch/nonlinear-t120.npz
```

The checkpoint records complex state, absolute time, a schema-versioned
grid/physics contract, and trajectory lineage. Restarts reject changed
topology, gradients, damping, collision controls, or checkpoints lacking the
lineage schema. Reports preserve the originating seed, initial amplitude,
zonal fraction, and complete segment-end schedule; this matters because a
truncated adaptive step at a checkpoint boundary can eventually select a
different chaotic realization. The resumed JSON statistics cover only the new
segment, which lets an initial transient be excluded deliberately.

Flux means are weighted by physical time, not by the number of accepted
adaptive steps. Uncertainty is estimated from equal-duration block means
(default block duration 5), and stationarity requires at least six blocks in
addition to 100 samples and duration 10. These controls are configurable but
should remain unchanged for acceptance ladders. The gate
also fits the logarithmic nonzonal-potential growth rate over exactly the same
window and requires its magnitude below `0.02` by default, preventing a slowly
growing or decaying field from passing on an accidentally flat flux interval.
Reports retain compact heat-flux and nonzonal-potential RMS time traces, so
adjacent checkpoint segments can be merged into windows independent of the
chosen checkpoint boundaries without storing phase-space state histories.

Merge new-format contiguous segments and recompute one candidate window with:

```bash
.venv/bin/python scripts/merge_nonlinear_heat_flux_segments.py \
  /tmp/nonlinear-t220-t300.json /tmp/nonlinear-t300-t400.json \
  --output /tmp/nonlinear-t220-t400-merged.json
```

The merger fails on gaps, reordered segments, normalization differences,
trajectory-initialization differences, or changes to any grid/physics contract
field. It recomputes the physical-time blocks over the merged window. Its
output is schema-v1 and can be passed directly to the convergence and parity
comparison functions.

Both local and GX schema-v1 reports carry a required top-level `stationary`
decision. Downstream convergence and parity preserve this producer decision in
addition to applying their drift and uncertainty limits, so a rejected window
cannot be promoted later by a weaker subset of the gates.

Nonlinear adaptive evolution compiles the state-dependent CFL evaluation and
one complete RK4 step while retaining host-controlled accept/termination
decisions. This reduces repeated dispatch overhead without moving nonsmooth
adaptive decisions onto the differentiable trajectory path.
Acceptance runs retain only their initial/final phase-space states and compute
heat flux plus total/nonzonal/per-`ky` potential amplitudes online. This avoids
an `O(n_steps * phase_space_size)` state-history allocation; the compact
diagnostics still default to every accepted step. `--diagnostic-stride` can
reduce diagnostic cadence for exceptionally long runs, while the final sample
is always retained.
Acceptance trajectories fail immediately unless JAX x64 is enabled. Checkpoint
contracts also record the complex state dtype, preventing a float32 state from
being resumed into an x64 campaign.

Load these schema-v1 reports with `load_nonlinear_heat_flux_record(...)` and
evaluate them using `compare_nonlinear_heat_flux(...)`. Local native heat flux
and GX `Q/Q_GB` labels deliberately fail comparison unless a documented
`local_to_reference_factor` is supplied. Resolution and box-size ladders use
`compare_nonlinear_heat_flux_convergence(...)` and require every rung to be
stationary before testing the finest-pair mean.

Once caller-owned stationary reports exist, evaluate the complete nonlinear
gate with separate resolution/domain ladders and at least three independently
initialized coarse lineages:

```bash
.venv/bin/python scripts/validate_nonlinear_heat_flux_campaign.py \
  --resolution-report /scratch/resolution-coarse.json \
  --resolution-report /scratch/resolution-fine.json \
  --domain-report /scratch/domain-narrow.json \
  --domain-report /scratch/domain-wide.json \
  --lineage-report /scratch/coarse-seed-1.json \
  --lineage-report /scratch/coarse-seed-2.json \
  --lineage-report /scratch/coarse-seed-3.json \
  --reference-report /scratch/gx-cyclone-nonlinear.json \
  --output /scratch/nonlinear-campaign.json
```

No conversion factor is needed when the local reports were generated with
`--flux-moment gx_total_energy`. For a genuinely different documented
normalization, pass `--local-to-reference-factor FACTOR`; the factor must not
be fitted to make the means agree. The command exits nonzero unless all lineage
roots are unique, every lineage is stationary, their means agree within 15%,
and resolution convergence, domain convergence, and independent parity pass.

The campaign also validates what each ladder changes. A resolution ladder must
strictly refine at least one of `n_z`, `n_vpar`, or `n_mu` on every rung while
holding the complete Fourier grid and physics contract fixed. A domain ladder
must hold phase-space resolution and physics fixed, reduce `delta_kx` and/or
`delta_ky`, and retain at least the previous maximum `|kx|` and `ky`. Repeated
reports, lost spectral bandwidth, and a damping/profile change therefore fail
before their flux means are considered. For the current coarse Fourier grid,
the corresponding two-rung domain study is `9x5` at `ky_min=0.1` followed by
`17x9` at `ky_min=0.05`; both retain `ky_max=0.4` and the same radial bandwidth.

The current coarse three-root campaign passes its ensemble sub-gate. Seeds
18, 19, and 20 give stationary means `-24.5555`, `-24.3764`, and `-23.7248`;
the ensemble mean is `-24.2189` and the maximum relative deviation is `2.04%`.
These caller-owned trajectories justify proceeding to the resolution and
domain ladders, but do not replace those ladders or the pinned GX comparison.

## Current Validation Status

Passing guardrails:

- Rosenbluth-Hinton late-plateau gate.
- Cyclone selected-ky scalar growth gate.
- Cyclone term-level and GKW RHS/action parity gates.
- DESC/GX/GX-GIST eik geometry contracts.
- Reduced stellarator scan and fixed-topology optimization plumbing.

Open blockers:

- end-to-end gradients through production MHD solvers and unrestricted shape
  optimization beyond the completed reduced Priority 4 W7-X loop,
- convergence of the independent low-`ky` stella branches before making a
  broader spectral claim,
- full GKW state-history/long-time velocity-slice parity and the low-`ky`
  Cyclone/GX complex branch shape, retained by the machine-readable Priority 5
  confidence ledger,
- inter-species Landau/Fokker--Planck field-particle parity (the species-local
  Xu correction matches pinned Gyaradax/GKW factors and action, but does not
  provide reciprocal exchange); electromagnetic one-state field/full-RHS,
  five-step RK4, final-time-10 dispersion, and production resolution gates pass,
- statistically stationary, resolution-converged nonlinear heat flux with an
  independent parity gate, and full DESC optimization.

Read `STATUS.md` for the latest state and `TODO.md` for the next concrete
tasks.

## Development Notes

- Keep new physics and numerical schemes documented in `tex/main.tex`.
- Add focused tests for every new public function in `src/`.
- Prefer small explicit fixture contracts over hidden convention fixes.
- Keep integer topology outside JAX gradient-traced paths.
- Do not label reduced examples as production physics results.

Useful companion docs:

- `docs/performance_and_differentiability.md`
- `docs/optimization_integration.md`
- `docs/velocity_backend_policy.md`
- `TODO.md`
- `STATUS.md`
