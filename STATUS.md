# STATUS

Last updated: 2026-08-05

## Executive Summary

`optimal-fusion` is now a standalone JAX package at its core boundary. An exact
development sync contains no MHD or reference-code provider, and the complete
default suite, package build, fresh wheel install, and import all pass without
sibling repositories. External validators are opt-in, revision checked, and
receive explicit source, executable, equilibrium, or generated-data paths.

The public MHD geometry boundary is now schema-versioned and provider neutral.
The immediate work is a direct in-memory VMEC++ W7-X provider followed by the
GVEC transformation and real-MHD gradient checks. The externally validated
W7-X scientific gate remains open.

The target architecture is:

- `optimal-fusion` owns the gyrokinetic solver and a code-neutral physical
  flux-tube geometry contract;
- installed MHD providers construct named equilibria such as W7-X and return
  geometry in memory;
- VMEC++, DESC, and GVEC are optional dependencies, not vendored source trees;
- GX, stella, GKW, and Gyaradax are independently versioned validation tools;
- equilibrium files and generated W7-X geometry/run artifacts are not stored
  in this repository.

A pinned dependency preparation layer now exists. `dependencies.toml` records
the exact PAIdynamics fork revisions, and `scripts/bootstrap_dependencies.py`
supports managed fetch/build/install as well as read-only reuse of matching
sibling clones.

Priority 0 standalone work also added clean-clone CI, external pytest markers,
installed-package smoke testing, path/revision provenance reporting, and a
declared NetCDF4 development reader. Raw GKW state/RHS/matrix dumps were
removed, reducing `fixtures/` from about 175 MiB to 5.2 MiB.

Priority 1 added the common request/provider/result API, extended physical
geometry metadata, DESC/GX/stella adapters, validation, external caching, and a
provider-independent solver path. Its implementation and conventions are
documented in `docs/geometry_provider.md`.

## Verified Implementation

### Core numerics

Implemented in `src/stellarator_gk/`:

- frozen JAX PyTree parameter/grid/geometry data models;
- Chebyshev velocity and open-parallel grids, Fourier periodic-parallel grids,
  and GKW-style finite-difference velocity/operator paths;
- perpendicular Fourier modes, mode-chain connectivity, and linked
  twist-and-shift operations;
- analytic circular and s-alpha geometry;
- a physical stellarator array model and one map to internal `B`, `F`, `G`,
  drift, and metric coefficients;
- FLR/Bessel factors, Maxwellians, equilibrium drive, parallel streaming,
  mirror force, magnetic drift, damping, and recurrence terms;
- matrix-free multi-species linear electrostatic RHS assembly with matrix,
  `gkw_upwind`, and fused `gkw_igh` parallel backends;
- adiabatic- and kinetic-species quasineutrality solvers;
- fixed-step RK4, modal filtering, CFL estimates, growth/frequency/mode-shape
  diagnostics, and state normalization;
- dense reduced-size operator/eigensystem helpers;
- differentiable objectives and fixed-topology reduced optimization wrappers;
- Hermite-Laguerre transforms, moments, hypercollision helpers, and a reduced
  GX-style moment RHS.  This moment path is not integrated as the production
  kinetic solver backend.

### Geometry interoperability

Present today:

- schema-v1 `GeometryRequest`, `GeometryProvider`, `GeometryResult`, and static
  metadata covering coordinates/units, signs, topology/linking, endpoint
  policy, normalization, differentiability, and provenance;
- `PhysicalFluxTubeGeometry` carrying `alpha`, `iota`, shear, `nfp`, field
  periods, topology, normalization, and provider identity as a JAX PyTree;
- synthetic and generic physical-array providers, DESC object/path provider,
  GX/GIST eik provider, and stella `.geometry` provider;
- strict provider-boundary validation and explicit external cache round trips;
- one physical-to-internal map for `F`, `G`, `E_y`, `D_x`, `D_y`, and metrics;
- a reduced scan and solver-construction test that are provider independent.

Not present today:

- a VMEC/VMEC++ adapter, a GVEC adapter, or direct named W7-X provider;
- a demonstrated gradient from real MHD equilibrium/design parameters through
  geometry into the GK objective.

The sibling repositories contain viable W7-X inputs.  In particular, VMEC++
has `examples/data/input.w7x` and exposes `VmecInput.from_file(...)` plus an
in-memory `vmecpp.run(...).wout` result.  That example input is excluded from
the VMEC++ source distribution, so the VMEC++ fork needs a supported packaged
W7-X configuration API/resource before it can serve as a stable dependency.

### Optimization maturity

The optimization API is differentiable for analytic geometry and supplied
fixed-grid arrays.  Its equilibrium coefficients, beta, and pressure-gradient
modulations are explicitly toy algebraic controls.  The DESC fixture loop is a
reduced plumbing demonstration, not end-to-end MHD shape optimization.

## Validation State

Previously established numerical guardrails remain represented in code and
committed contracts:

- Rosenbluth-Hinton late-plateau validation;
- selected-ky Cyclone scalar growth and extensive GKW term/action/state
  comparisons;
- DESC/GX eik conversion and geometry checks;
- reduced W7-X scans, convergence ledgers, and optimization readiness guards.

The scientific W7-X gate remains open:

- matched stella geometry, field-line length, `ky`, `kx=0`, and late-time
  controls brought growth close at `t=200`;
- the `ky=0.3` real-frequency and phase-aligned mode-profile comparison remains
  outside tolerance;
- the solver-side RHS balance reconstructs and is streaming dominated;
- direct stella-vs-solver term-array comparison remains blocked by incompatible
  velocity/z contracts and missing weights in the retained v1 trace summary;
- production convergence, CPU timing, and MHD design optimization remain
  blocked behind external W7-X parity.

These results currently depend on committed derived fixtures.  They must be
reproduced through live, explicitly versioned providers/validators before they
can support the standalone production claim.

## Repository Audit

### Standalone acceptance on 2026-08-05

```text
ruff check src tests examples scripts: PASS
uv sync --extra dev: PASS (exact provider-free environment)
uv pip check --python .venv/bin/python: 25 packages compatible
JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q:
  329 passed, 14 external tests deselected in 478.45 s
scripts/package_smoke_test.py --python 3.13:
  sdist build, wheel build, fresh wheel install, and import: PASS
```

The prior 13 failures were replaced with synthetic contracts or opt-in external
tests. The default suite has no source-tree or executable availability skips;
external tests are deselected by the registered `external` marker. NetCDF4 is
declared in the development extra, while provider packages remain installed by
the pinned manifest bootstrapper.

### Repository boundary

- Active Python, scripts, tests, README files, and fixture metadata contain no
  `relevant-codes/...` checkout-relative paths.
- External workflows require explicit inputs and announce their absolute path,
  enclosing Git revision, or unversioned status.
- Clean-clone CI runs the provider-free standalone and packaging gates.
- Generated egg-info is ignored and no longer tracked.
- `benchmarks.py` is 16,564 lines and `__init__.py` eagerly re-exports much of
  the validation surface through the default package namespace.
- `fixtures/` is 5.2 MiB with no individual file over 500 KiB. Full GKW
  generated state/RHS/matrix dumps and their repository-coupled tests are gone.
- Compact historical W7-X validation records remain temporarily; Priority 2
  removes superseded geometry artifacts after the live provider reproduces
  their contracts.
- There are no installed console entry points for the solver workflows; the
  user interface is currently a collection of examples and scripts.

### Dependency preparation

Implemented profiles:

- `mhd`: VMEC++, DESC, and GVEC, installed as Python packages (including their
  native build steps where defined by those packages);
- `validation-python`: Gyaradax;
- `validation-native`: GX, stella, and GKW, built with their native systems;
- `validation` and `all`: combined profiles.

Managed clones/builds/executable links/state live below ignored
`.dependencies/`. Every dependency is pinned to a full Git commit. With
`--local-root`, a matching sibling clone is revision-checked and used without
automatic checkout or source modification. Native executables are exposed via
`.dependencies/bin`, and `.dependencies/state.json` records the local resolved
state.

The manifest/bootstrap unit tests and Ruff pass, and dry runs succeed for both
managed-fetch and current sibling-checkout paths. The complete local `mhd`
profile was also built and import-verified on macOS/Python 3.13. VMEC++ uses the
documented Homebrew OpenMP root, and a build-only compatibility shim supports
the old Versioneer copy in the pinned DESC fork without modifying its source.
Bootstrap project syncs are inexact so repeat runs retain installed providers.
The final environment is checked for package incompatibilities; the MHD profile
resolves DESC's JAX `<0.10` requirement rather than leaving the core lock's JAX
0.10.1 installed alongside it.
Native validation builds remain platform dependent: GX requires CUDA and
`GK_SYSTEM`, while stella and GKW may require site compiler/MPI/NetCDF setup.

Provider forks remain manifest-installed instead of PEP 508 extras: their Git
pins, native prerequisites, build commands, and executable discovery require
the bootstrap ledger. Pure-Python file readers belong in project extras;
NetCDF4 is now in `dev`. An exact later `uv sync --extra dev` intentionally
removes providers and restores the standalone lock, while bootstrap syncs are
inexact so prepared providers remain installed.

## Active Priorities

1. Add VMEC++ as the first live named W7-X provider and expose the standard
   W7-X input through a supported API/resource in the VMEC++ fork.
2. Convert the in-memory VMEC output to physical flux-tube arrays without an
   intermediate committed `wout`/GX/GIST/stella file.
3. Add the corresponding live GVEC transformation and a real-MHD gradient
   check through the provider result.
4. Cross-check the live providers against independent GX and stella references,
   then remove superseded tracked W7-X equilibrium/derived artifacts.
5. Resume the stella `ky=0.3` weighted term-array parity gate, followed by
   convergence, CPU timing, and real MHD optimization gradients.

## Current Risks

- External integrations can drift from their pinned forks or fail on local
  native toolchains even though the provider-free suite remains green.
- Real VMEC++ and GVEC outputs still need transformations into the now-stable
  physical provider contract.
- The current W7-X claim is fixture-driven rather than provider-driven.
- The fixed-step initial-value objective may change branches or become
  ill-conditioned during design optimization; branch/gradient checks are not
  yet implemented for real equilibria.
- Production CPU scaling on approximately 100 CPUs has not been demonstrated.
- Collisions, electromagnetic physics, nonlinear turbulence, kinetic-electron
  TEM production validation, and full shape optimization are deferred.

## Round Log

### 2026-08-05: Priority 1 Public Geometry Interface Complete

- Added schema-v1 request/provider/result types with explicit units,
  normalization, signs, coordinates, endpoint policy, topology/linking, and
  provenance.
- Extended physical geometry with `alpha`, `iota`, shear, `nfp`, field periods,
  normalization, topology, endpoint policy, and provider identity while
  retaining continuous arrays as JAX leaves.
- Added actionable validation, differentiability tests, and external cache
  serialization that preserves provenance and never claims gradients after I/O.
- Added DESC object/path, GX/GIST eik, stella geometry, synthetic, and generic
  physical-array providers; moved eik/stella parsing into focused modules.
- Routed the reduced scan and a common linear-residual smoke run through the
  provider-neutral result and canonical physical-to-internal mapper.
- Verified Ruff and the complete suite: 329 passed, 14 external tests
  deselected. The only initial acceptance failure was a stale assertion for the
  corrected `alpha = theta - iota * phi` field-line label and passed after the
  contract assertion was updated.

### 2026-08-05: Priority 0 Standalone Boundary Complete

- Replaced the 13 checkout-dependent failures with synthetic contracts or
  opt-in revision-pinned integration tests.
- Removed active `relevant-codes/...` defaults and required explicit external
  workflow paths with path/revision provenance output.
- Added the `external` pytest marker, provider-root fixtures, clean-clone CI,
  package consistency checks, and an sdist/wheel installation smoke test.
- Stopped tracking generated egg-info and declared NetCDF4 in the development
  environment; documented why provider forks remain manifest-installed.
- Removed approximately 170 MiB of raw GKW state, RHS, input, and matrix dumps;
  retained synthetic loader tests and explicit external comparison workflows.
- Verified Ruff, an exact provider-free sync, package compatibility, 305
  standalone tests with 14 external tests deselected, and fresh wheel import.

### 2026-08-04: Standalone Architecture and Code Audit

- Removed the untracked nested external-code copies and generated W7-X/stella
  outputs; committed cleanup as `1495de4`.
- Updated ignore rules so external trees and generated reference-run outputs
  are not recommitted.
- Read the solver, geometry, physics, time-advance, optimization, benchmark,
  script, test, packaging, and fixture structure.
- Ran the full x64 test suite and Ruff; recorded the exact standalone failures.
- Confirmed programmatic W7-X starting configurations in the sibling VMEC++,
  DESC, and GVEC repositories.
- Reprioritized the roadmap around a live VMEC++ W7-X provider and a green
  standalone boundary before further scientific promotion.

### 2026-08-04: Pinned Dependency Bootstrap

- Added full-commit pins and preparation profiles for VMEC++, DESC, GVEC,
  Gyaradax, GX, stella, and GKW.
- Added managed clone/build/install handling, safe sibling-clone reuse, native
  executable links, dry-run/fetch-only modes, and a local state ledger.
- Added focused manifest/profile/dry-run/environment-path tests and dependency
  setup documentation.
- Built and import-verified the full VMEC++/DESC/GVEC profile from revision-
  checked sibling clones on macOS/Python 3.13.

### 2026-06-29 and Earlier

- Implemented the core grids, geometry arrays, linear physics, quasineutrality,
  RHS, time advancement, objectives, performance helpers, reduced optimization,
  and extensive GKW/GX/stella validation infrastructure.
- Narrowed the W7-X stella mismatch to velocity/RHS term parity, with the
  largest scale-free scalar discrepancy in the parallel-streaming bundle.
