# STATUS

Last updated: 2026-08-04

## Executive Summary

`optimal-fusion` contains a substantial JAX-first linear electrostatic
flux-tube gyrokinetic implementation, but it is not yet a clean standalone
stellarator-design package.  The numerical core is broadly exercised and the
current audit found no new core numerical regression.  The immediate blocker
is now the software boundary: tests and workflows still depend on paths from
the removed in-repository copies of Gyaradax, GX, stella, and DESC.

The target architecture is:

- `optimal-fusion` owns the gyrokinetic solver and a code-neutral physical
  flux-tube geometry contract;
- installed MHD providers construct named equilibria such as W7-X and return
  geometry in memory;
- VMEC++, DESC, and GVEC are optional dependencies, not vendored source trees;
- GX, stella, GKW, and Gyaradax are independently versioned validation tools;
- equilibrium files and generated W7-X geometry/run artifacts are not stored
  in this repository.

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

- `PhysicalFluxTubeGeometry` and `FluxTubeGeometry` array containers;
- direct construction from physical/precomputed arrays;
- lazy DESC object/path extraction through `eq.compute`;
- GX/GIST eik parsing and parity logic inside the large benchmark module;
- stella `.geometry` parsing inside `examples/run_stellarator_linear_scan.py`.

Not present today:

- a `GeometryProvider` protocol or provider registry;
- a request/result metadata model covering normalization, `alpha`, `iota`,
  field periods, topology/linking, provenance, and differentiability;
- a VMEC/VMEC++ adapter, a GVEC adapter, or direct named W7-X provider;
- a single package-level path shared by DESC, VMEC++, GVEC, GX/GIST, and stella;
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

### Verification on 2026-08-04

```text
ruff check src tests examples scripts: PASS
JAX_ENABLE_X64=1 python -m pytest -q:
  306 passed, 13 failed, 8 skipped in 544.46 s
```

All 13 failures are caused by removed hard-coded external paths:

- 3 tests read Gyaradax geometry source directly;
- 8 tests read GX Cyclone inputs or GX/GIST eik tables directly;
- 2 tests read and patch stella source directly.

The eight skips cover unavailable optional NetCDF4/DESC integrations and local
GX/GIST W7-X inputs.  No failure in this run identified a core JAX solver,
physics, time-advance, objective, or differentiability regression.

### Standalone/package gaps

- 37 tracked files still contain `relevant-codes/...` references.
- There is no CI configuration or pytest external-integration marker scheme.
- `pyproject.toml` declares only core, `dev`, and a small `reference` extra; it
  does not declare DESC, VMEC++, GVEC, or NetCDF4 integration extras.
- `src/stellarator_gk.egg-info/` is tracked and is modified by local builds.
- `benchmarks.py` is 16,564 lines and `__init__.py` eagerly re-exports much of
  the validation surface through the default package namespace.
- `fixtures/` occupies about 175 MiB, mostly full GKW state/RHS/matrix dumps.
- 173 tracked W7-X-related files occupy about 5 MiB.  Equilibrium and derived
  geometry/run artifacts in this set should be removed after live providers
  reproduce the required contracts.
- There are no installed console entry points for the solver workflows; the
  user interface is currently a collection of examples and scripts.

The local `.venv` was moved with the repository, leaving stale absolute
shebangs in console scripts.  The audit therefore invoked pytest and Ruff with
`python -m ...`.  This is a local environment relocation issue; clean-clone
wheel/CI tests are still needed to validate the documented installation path.

## Active Priorities

1. Restore a green standalone test suite by removing the 13 direct source/input
   dependencies and separating external integration tests with markers.
2. Define the versioned `GeometryRequest` / `GeometryProvider` /
   `GeometryResult` contract, including the parallel grid and full metadata.
3. Add VMEC++ as the first live named W7-X provider and expose the standard
   W7-X input through a supported API/resource in the VMEC++ fork.
4. Convert the in-memory VMEC output to physical flux-tube arrays without an
   intermediate committed `wout`/GX/GIST/stella file.
5. Cross-check the live provider against independent GX and stella references,
   then remove superseded tracked W7-X equilibrium/derived artifacts.
6. Resume the stella `ky=0.3` weighted term-array parity gate, followed by
   convergence, CPU timing, and real MHD optimization gradients.

## Current Risks

- A large passing test count masks that the default suite is not standalone.
- Geometry conventions are distributed across the package, benchmark module,
  examples, scripts, and fixture metadata.
- `PhysicalFluxTubeGeometry` is not yet rich enough to be the stable MHD/GK
  interface because field-line and normalization metadata are missing.
- The current W7-X claim is fixture-driven rather than provider-driven.
- The fixed-step initial-value objective may change branches or become
  ill-conditioned during design optimization; branch/gradient checks are not
  yet implemented for real equilibria.
- Production CPU scaling on approximately 100 CPUs has not been demonstrated.
- Collisions, electromagnetic physics, nonlinear turbulence, kinetic-electron
  TEM production validation, and full shape optimization are deferred.

## Round Log

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

### 2026-06-29 and Earlier

- Implemented the core grids, geometry arrays, linear physics, quasineutrality,
  RHS, time advancement, objectives, performance helpers, reduced optimization,
  and extensive GKW/GX/stella validation infrastructure.
- Narrowed the W7-X stella mismatch to velocity/RHS term parity, with the
  largest scale-free scalar discrepancy in the parallel-streaming bundle.
