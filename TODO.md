# TODO: Differentiable Flux-Tube Stellarator GK Solver

Last reviewed: 2026-08-05

## Goal

Build `optimal-fusion` as a standalone JAX package for differentiable, local
flux-tube gyrokinetics in stellarator design.  The package must own its solver,
geometry data contract, fixtures, and core tests.  MHD and reference
gyrokinetic codes must remain separately installable providers or validation
tools, not copied source trees or implicit runtime dependencies.

The current scientific milestone remains a trusted, externally validated
linear electrostatic W7-X run.  Nonlinear turbulence, collisions,
electromagnetic effects, and production equilibrium-shape optimization remain
later milestones.

Keep the workflow simple:

1. Specify physics and numerics in `tex/main.tex`.
2. Implement reusable package functionality in `src/stellarator_gk/`.
3. Add focused, self-contained tests for each public contract.
4. Keep external-code comparisons reproducible through explicit paths,
   versions, and generated fixtures.
5. Keep `STATUS.md` current with commands, results, and blockers.

## Current State

Implemented and tested:

- core PyTree types, grids, mode connectivity, linear electrostatic RHS, field
  solve, time advance, diagnostics, objectives, and fixed-topology gradients;
- a versioned MHD-neutral `GeometryProvider` contract, extended physical
  geometry model, strict validation, and one map from physical arrays to the
  solver's internal coefficients;
- synthetic, physical-array, DESC object/path, GX/GIST eik, and stella
  `.geometry` providers, plus explicit external caching and provenance;
- reduced RH, Cyclone, GKW/Gyaradax, DESC/eik, W7-X, stella, and GX validation
  infrastructure;
- reduced stellarator scans and optimization examples.

Repository cleanup already completed:

- [x] Remove tracked copies of external codes and papers from this repository.
- [x] Keep DESC, GX, stella, GKW, Gyaradax, VMEC++, GVEC, and papers as sibling
  repositories/reference material outside `optimal-fusion`.
- [x] Move the project TeX sources to `tex/`.
- [x] Add a pinned external dependency manifest and bootstrapper with separate
  `mhd`, Python-validation, and native-validation profiles; managed content is
  kept under ignored `.dependencies/`, while sibling clones can be verified
  and reused without modification.

Standalone Priority 0 and geometry-interface Priority 1 are complete. The next
architectural gaps are:

- there is no first-class VMEC/VMEC++ `wout` or GVEC adapter, so W7-X geometry
  is currently obtained indirectly from GX/GIST or stella artifacts;
- `benchmarks.py` is 16,564 lines and the top-level `__init__.py` eagerly
  re-exports most of it, coupling the solver API to validation infrastructure;
- compact historical W7-X validation records still need to migrate to live
  providers as the direct W7-X path is implemented in Priority 2.

Standalone acceptance on 2026-08-05:

- `ruff check src tests examples scripts`: pass;
- provider-free x64 pytest run after an exact `uv sync --extra dev`: 329 passed,
  14 external tests deselected;
- sdist and wheel build, fresh-environment wheel install, and import: pass;
- fixtures reduced from about 175 MiB to 5.2 MiB; no fixture exceeds 500 KiB.

## Priority 0: Make the Repository Genuinely Standalone

- [x] Remove all hard-coded `relevant-codes/...` defaults from tracked Python,
  tests, README files, and fixture metadata.  Never resolve a dependency by its
  location relative to the `optimal-fusion` checkout.
- [x] First restore a green standalone suite by fixing the 13 known failures:
  replace three Gyaradax source-reading tests, eight GX input/eik tests, and two
  stella source-patching tests with synthetic/compact contracts or explicitly
  configured integration tests.
- [x] Make external source trees explicit optional inputs such as
  `--desc-root`, `--gx-root`, `--stella-root`, `--gkw-root`, and executable or
  equilibrium paths.  Environment-variable fallbacks may be offered, but every
  workflow must print the resolved path and external revision.
- [x] Provide one preparation command that fetches exact fork commits,
  builds/installs Python MHD providers, compiles native validation codes on
  request, links discovered executables under `.dependencies/bin`, and writes
  a local revision/command ledger.
- [x] Keep only small, code-independent numerical validation contracts needed
  by the default test suite in `fixtures/`; record their origin, generating
  dependency/version, configuration identifier, and command. Do not commit
  external implementation source or full generated run output. Historical
  compact W7-X comparison records are temporary contracts; Priority 2 removes
  superseded geometry artifacts after a live provider reproduces them.
- [x] Separate tests into self-contained core/fixture tests and opt-in external
  integration tests with pytest markers.  Core tests must skip neither because
  a sibling checkout is absent nor because an external executable is absent.
- [x] Replace tests that read Gyaradax/GX/DESC source files with committed
  numerical contracts or synthetic provider doubles.
- [x] Add a clean-clone CI job that installs only declared dependencies and
  runs import, unit, reduced W7-X fixture, and packaging tests with no sibling
  repositories available.
- [x] Add an sdist/wheel smoke test and verify installed-package workflows do
  not depend on repository-relative files.
- [x] Stop tracking `src/stellarator_gk.egg-info/`; build metadata must be
  generated and ignored.
- [x] Decide whether provider packages should remain manifest-installed or
  also be exposed as PEP 508 extras. Add NetCDF4/reader extras as needed. The
  bootstrapper now uses an inexact project sync to retain installed providers;
  document that a later exact `uv sync` intentionally restores the core lock.
- [x] Ignore local external checkout directories and generated external-run
  outputs so they cannot be accidentally recommitted.

Acceptance gate: **passed**. From the exact core lock, `uv sync --extra dev`
followed by the core test command passes without DESC, GX, stella, GKW,
Gyaradax, VMEC++, or GVEC source trees. Provider/validator tests are opt-in and
consume explicit revision-checked roots rather than checkout-relative paths.

## Priority 1: Define One Public MHD Geometry Interface

- [x] Define and document a small public `GeometryProvider` protocol that
  accepts a `GeometryRequest` and returns a `GeometryResult` containing the
  `ParallelGrid`, `PhysicalFluxTubeGeometry`, and metadata.  Solvers must
  consume this contract rather than knowing which MHD code produced it.
- [x] Version the serialized geometry schema and specify coordinates, units,
  normalization, sign conventions, radial coordinate, `alpha`, field periods,
  periodic endpoint policy, twist-and-shift/linking data, quadrature weights,
  and provenance.
- [x] Extend the physical geometry model to retain field-line `alpha`, `iota`,
  `nfp`, topology/linking, normalization, and provider metadata before treating
  it as the public inter-code contract.
- [x] Separate static topology/file I/O from differentiable arrays.  State
  clearly which provider outputs can carry gradients back to equilibrium
  parameters and test those claims with `jax.grad`.
- [x] Move GX/GIST eik and stella `.geometry` parsing out of examples/benchmark
  code into focused optional adapter modules that produce the same physical
  contract.
- [x] Add schema validation with actionable errors for missing fields,
  incompatible normalization, non-finite arrays, duplicate endpoints, and
  inconsistent grid sizes.
- [x] Allow optional user-controlled serialization/caching of provider output
  for expensive runs, but keep it outside the source tree and make the live
  provider API the canonical path.  The solver must not require a separately
  stored geometry file.
- [x] Keep solver-internal coefficients (`F`, `G`, `E_y`, `D_x`, `D_y`, metric
  terms) derived in one tested location; providers should expose physical
  quantities, not duplicate solver conventions.

Acceptance gate: **passed at the interface level**. One solver-construction
path switches among synthetic, DESC/VMEC++/GVEC physical-array handoffs,
GX/GIST, and stella without provider-specific solver branches. Priority 2
replaces the VMEC++/GVEC array handoffs with live MHD transformations.

## Priority 2: Add Clean MHD Providers and a Direct W7-X Path

### DESC

- [x] Turn the existing DESC object/path adapter into an optional provider with
  a declared installation extra and no `PYTHONPATH` mutation or source-tree
  assumption.
- [x] Pin/test the supported DESC API and add a small mock-based unit test plus
  an opt-in integration test against the sibling DESC checkout or installed
  release.
- [x] Verify gradients from continuous DESC equilibrium parameters through
  sampled geometry and the reduced GK objective while holding grid topology
  fixed.

### VMEC / VMEC++ and W7-X

- [x] Define a stable programmatic equilibrium request such as an MHD provider
  plus a named configuration (`W7-X`, configuration variant, surface, and
  field line).  Record those inputs and dependency versions for reproducibility
  instead of committing an equilibrium artifact.
- [x] Use VMEC++ as the first live W7-X provider: construct `VmecInput`, call
  `vmecpp.run(...)`, and consume the returned in-memory `VmecOutput.wout`
  without writing `wout_w7x.nc` in this repository.
- [x] Update the VMEC++ fork to expose its W7-X standard input as a supported
  package resource or named Python API.  The current input exists at
  `vmecpp/examples/data/input.w7x`, but examples are excluded from its sdist and
  therefore are not a stable dependency interface.
- [x] Implement or upstream the VMEC-output-to-field-line transformation that
  evaluates all physical flux-tube arrays required by
  `PhysicalFluxTubeGeometry` in memory; do not route the canonical path through
  GX/GIST or a temporary NetCDF file.
- [x] Keep `wout*.nc` import as an optional user interoperability path, not the
  canonical W7-X source and not a committed project fixture.
- [x] Define equivalent named-configuration hooks for DESC and GVEC where their
  APIs can construct or load the W7-X design programmatically.
- [x] Remove committed W7-X equilibrium and derived-geometry artifacts after
  the live provider reproduces them.  Integration tests should generate data
  in a temporary/cache directory and retain only compact numerical assertions
  where a regression contract is essential.
- [x] Cross-check the direct W7-X provider against the committed GX/GIST and
  stella geometry contracts term by term, including `B`, parallel derivative
  scaling, metric elements, drift coefficients, field-period count, and
  endpoint/linking conventions.
  The same-source stella comparison covers every listed term (`B` is within 1%
  relative L2 and legacy equal-arc metrics/drifts within a documented 30%
  envelope). The matched GX VMEC/GIST pair adds a same-surface comparison for
  all nine GIST fields, shear, field periods, and endpoint policy.
- [x] Make the reduced W7-X example accept a provider and named design directly
  (for example `--geometry-provider vmecpp --configuration w7x-standard`), with
  optional user-supplied equilibrium files as a secondary path.

### GVEC

- [x] Specify the minimum GVEC output/API needed to build the physical
  flux-tube contract and implement it as an optional adapter or exporter.
- [x] Add an opt-in GVEC integration test and compare a common equilibrium with
  VMEC/DESC after matching surface and field line.
  The live sibling-GVEC integration test passes. A second opt-in test loads
  GVEC's VMEC-initialized W7-X case and compares every normalized physical term
  against the direct VMEC++ provider on the same surface and field line.

Acceptance gate: **passed**. W7-X runs from a named configuration through the
installed VMEC++ dependency without a repository-stored design/geometry file.
A user-supplied equilibrium file remains optional, while GX/stella are
independent validation programs rather than canonical geometry sources.
Priority 2 is complete; further tolerance tightening belongs to the scientific
validation work in Priority 3.

## Priority 3: Preserve and Close the W7-X Scientific Validation Gate

- [x] Rerun the patched sibling stella RHS trace in format v3 with `wgts_vpa`
  and z-dependent `wgts_mu`; require an explicit stella source/executable path
  and record its commit.
- [x] Drop the duplicate periodic stella z endpoint before array comparison.
- [x] Emit a solver-side selected-mode full-array trace on a stella-compatible
  `z/vpa/mu` grid, or add a documented interpolation and weighting adapter.
- [x] Compare velocity-weighted complex arrays for distribution, parallel
  streaming, mirror force, magnetic drift, equilibrium drive, field-drive
  terms, total RHS, quasineutrality numerator/denominator, and normalization.
  The complete contract is now executable and currently fails numerical parity;
  keep downstream mode-structure and production gates blocked.
- [x] Inspect the parallel-streaming derivative/linking convention first if
  weighted arrays confirm the current scalar mismatch.
- [x] Replay each labeled stella state through the solver RHS on a contained
  common velocity grid. The source-derived stella coefficient discriminator
  reduces total-RHS error from 1.32 to 0.237 and brings equilibrium drive below
  0.1. An exact-stella-node 32×4 discriminator improves drift to 0.164 and
  streaming to 0.115. Reproducing stella's sign-dependent third-order upwind
  mirror stencil and tracing its native in-memory coefficient reconstructs the
  native-grid term within 0.004 relative L2. After interpolation to the solver's
  uniform four-node mu grid the mirror error is 0.243, identifying interpolation
  of coefficient-state products as the remaining mirror comparison floor. The
  native gyroaverage plus z-dependent stella velocity weights also reconstruct
  its quasineutrality numerator to machine precision. The remaining numerator
  gap is therefore a cross-grid velocity-measure contract, not an unknown state
  scale; the solver needs an explicit native-quadrature adapter before parity.
- [x] Separate equilibrium-gradient drive scale from magnetic-drift geometry in
  provider schema v2 and map stella's native `flux_fac` into that field.
- [ ] Promote the remaining verified Maxwellian/quadrature normalization,
  drift factor, and mirror orientation into provider-neutral geometry/state
  contracts; validate each change against the non-stella gates.
- [ ] After term parity passes, rerun the W7-X mode-structure gate against the
  matched stella fixture.
- [ ] Replace the reduced convergence ladder with production controls in
  parallel and velocity resolution, backend, modes, field-line length,
  timestep, and growth window.
- [ ] Run guarded CPU timing only after parity and convergence pass; keep DESC
  production optimization blocked until the readiness ledger passes.
- [ ] Optionally run the sibling GX W7-X workflow on a CUDA machine as a
  secondary moment-method cross-check.

## Priority 4: Differentiable Design Integration

- [ ] Define a stable objective API from equilibrium/provider parameters to
  growth rate, real frequency, mode-structure penalties, and quasilinear
  proxies.
- [ ] Add finite-difference checks of gradients through geometry and time
  advance, including near-degenerate eigenmode/branch cases.
- [ ] Define remeshing and topology-change behavior: gradients are valid only
  within a fixed grid/field-line connectivity contract, and optimization must
  detect when rebuilding is required.
- [ ] Demonstrate a reduced W7-X design loop through a real MHD provider before
  claiming full shape optimization.
- [ ] Add multiple surfaces and field lines, robust aggregation objectives,
  checkpointing, and reproducible optimization metadata.

## Priority 5: Confidence Gaps and Deferred Physics

- [ ] Keep GKW selected-mode state-history and multi-time velocity-slice gaps
  visible until closed or superseded by an independent stellarator reference.
- [ ] Decide whether Chebyshev/GKW finite-difference velocity collocation is
  the production CPU backend or the Hermite-Laguerre backend must become fully
  production capable first.
- [ ] Keep the multi-ky Cyclone/GX low-ky branch-shape gap in the validation
  ledger.
- [ ] Validate kinetic-electron TEM physics.
- [ ] Add collisions and electromagnetic perturbations.
- [ ] Add the nonlinear ExB pseudo-spectral bracket, dealiasing, nonlinear
  timestep control, saturated heat-flux diagnostics, and nonlinear parity.
- [ ] Extend to full equilibrium-shape optimization only after the standalone
  geometry, W7-X parity, convergence, timing, and gradient gates pass.

## Priority 6: Maintainability After the Standalone Boundary Is Green

- [ ] Split the 16k-line `benchmarks.py` into focused fixture I/O, Cyclone/GKW,
  geometry parity, and W7-X validation modules.
- [ ] Keep benchmark-only symbols out of the default top-level import path;
  expose a compact solver API and a separate validation namespace.
- [ ] Replace historical `Phase N` docstrings with subsystem descriptions and
  document which APIs are stable at version `0.1.x`.
- [ ] Decide which large GKW traces are essential compact regression contracts,
  regenerate smaller selected slices where possible, and move archival raw
  traces out of the source distribution.

## Project Rules

- `optimal-fusion` must never require a particular sibling-directory layout.
- External codes are optional producers and validators; their source is not
  part of this package.
- `tex/main.tex` is the concise physics/numerics source of truth.
- `STATUS.md` is a current snapshot and short round log, not an exhaustive
  transcript.
- Every external fixture records code/version, input, command, normalization,
  and checksum/provenance.
- Prefer small explicit contracts and adapter tests over hidden convention
  fixes.
- Do not claim production W7-X optimization or nonlinear turbulence until the
  corresponding parity, convergence, timing, gradient, and physics gates pass.
