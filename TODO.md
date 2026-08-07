# TODO: Differentiable Flux-Tube Stellarator GK Solver

Last reviewed: 2026-08-06

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

Standalone Priority 0, geometry-interface Priority 1, live-MHD Priority 2, and
W7-X scientific-validation Priority 3 are complete. The next architectural
gaps are:

- validate end-to-end gradients through a real installed MHD solve and define
  fixed-topology/remeshing behavior for Priority 4 design optimization;
- `benchmarks.py` remains oversized and the top-level `__init__.py` eagerly
  re-exports much of it, coupling the solver API to validation infrastructure;
- retain historical W7-X fixtures only as compact independent regression
  evidence while canonical geometry comes from installed MHD providers.

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
  The provider-native 32×8 same-state acceptance case now passes its 0.1
  relative-L2 tolerance with a maximum RHS error of 0.04393.
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
  its quasineutrality numerator to machine precision. An explicit arbitrary-node
  velocity-grid builder and z-dependent phase-space measure now preserve that
  provider-native quadrature contract without storing W7-X state in the solver.
- [x] Separate equilibrium-gradient drive scale from magnetic-drift geometry in
  provider schema v2 and map stella's native `flux_fac` into that field.
- [x] Promote the verified provider-neutral conventions: arbitrary velocity
  nodes and quadrature, a z-dependent phase-space measure, separate grad-B and
  curvature drift components, and the mapped mirror-force orientation. The
  stella Maxwellian conversion and traced source algebra remain explicit replay
  adapter conventions rather than hidden production defaults. Focused geometry,
  optimization, RHS, quasineutrality, and replay gates pass.
- [x] After term parity passes, rerun the W7-X mode-structure gate against the
  matched stella fixture. The corrected `t=200` production run keeps growth
  within tolerance but leaves frequency/profile errors of 0.131/0.149. A clean
  pinned stella rerun reproduces the fixture and proves its averaging window is
  unconverged; a scratch `t=500` run converges `ky=0.3` to approximately
  `(gamma, omega)=(0.01754, 0.04638)`, while the matched solver converges to
  `(-0.00013, -0.05497)` from the old generic initial state. This established
  the reproducible long-time discriminator later closed by source-matched stage
  ordering and initialization below.
- [x] Replace the reduced convergence ladder with production controls. Source
  tracing established the exact stella step as SSP RK3 explicit stages, one
  full direction-aware cubic mirror characteristic, and one full implicit
  streaming/field response. JAX replay of a native stella step agrees to
  `7.19e-6` after RK3, `1.34e-4` after mirror advance, `2.17e-5` in the final
  distribution, and `1.34e-4` in potential. The remaining long-time profile
  mismatch was an initial-state mismatch, not a timestep defect. An analytic
  `stella_maxwellian` initializer now reproduces stella's default perturbation
  without storing equilibrium or distribution artifacts. At `t=500`, 32×8,
  `dt=0.1`, the converged unstable `ky=0.3` branch has growth/frequency/profile
  errors `3.20e-4/2.18e-3/1.08e-2`, passing the 0.02 gate. The `ky=0.1,0.2`
  omega windows remain diagnostic: the independent stella reference fails its
  own convergence check even at `t=1000`, so those branches are not used to
  weaken or broaden the production claim.
- [x] Run guarded CPU timing after parity. The `stella-production` timing preset
  exercises the validated 256×32×8, `ky=0.3` advance end to end in disposable
  external scratch space. A 100-step CPU run passed in 13.34 s including geometry
  loading, JAX compilation, and diagnostics, with a 5.13 MiB memory estimate.
  DESC production optimization remains a Priority 4 gradient/readiness task.
- [x] Keep the sibling GX CUDA workflow optional as a secondary moment-method
  cross-check; it is not a Priority 3 acceptance requirement and no CUDA claim
  is made from this CPU campaign.

Acceptance gate: **passed for the converged unstable W7-X branch**. Priority 3
is complete. External stella source, geometry, and generated run data remain
explicit scratch inputs and are not stored as canonical W7-X design artifacts.

## Priority 4: Differentiable Design Integration

- [x] Define a stable objective API from equilibrium/provider parameters to
  growth rate, real frequency, mode-structure penalties, and quasilinear
  proxies.
- [x] Add finite-difference checks of gradients through geometry and time
  advance, including near-degenerate eigenmode/branch cases.
- [x] Define remeshing and topology-change behavior: gradients are valid only
  within a fixed grid/field-line connectivity contract, and optimization must
  detect when rebuilding is required.
- [x] Demonstrate a reduced W7-X design loop through a real MHD provider before
  claiming full shape optimization.
- [x] Add multiple surfaces and field lines, robust aggregation objectives,
  checkpointing, and reproducible optimization metadata.

Acceptance gate: **passed for reduced fixed-topology design integration**. The
stable objective, gradient audits, remeshing detection, robust multi-sample
aggregation, and checkpoint schema are covered by standalone tests. A local
smoke run completed center/plus/minus solves from VMEC++'s installed
`w7x-standard` configuration and retained only an external JSON record. This
does not claim differentiation through VMEC++, unrestricted boundary-shape
optimization, or nonlinear turbulent transport; those remain deferred.

## Priority 5: Confidence Gaps and Deferred Physics

- [x] Keep GKW selected-mode state-history and multi-time velocity-slice gaps
  visible until closed or superseded by an independent stellarator reference.
- [x] Decide whether Chebyshev/GKW finite-difference velocity collocation is
  the production CPU backend or the Hermite-Laguerre backend must become fully
  production capable first.
  Decision: collocation remains the production CPU representation. Chebyshev
  is the general reduced/differentiable default, GKW finite differences are a
  convention-parity specialist, and the midpoint/Gauss-Laguerre recipe is the
  validated W7-X path. Hermite-Laguerre remains an experimental GX
  discriminator until it passes the full residual, field, convergence,
  gradient, external-parity, and timing gates.
- [x] Keep the multi-ky Cyclone/GX low-ky branch-shape gap in the validation
  ledger.
- [x] Validate kinetic-electron TEM physics.
  Algebraic preflight now passes for a charge-neutral heavy-electron,
  TEM-favorable case: kinetic quasineutrality closes, the multispecies RHS is
  finite, and the electron/ion streaming ratio matches the mass-ratio scaling.
  Keep this item open until a converged growth/frequency/mode-structure scan
  agrees with a revision-pinned independent code.
  A normalized multi-window discriminator now exposes growth, frequency, late
  drift, and CFL metadata. Its reduced ladders are not resolution-converged;
  the investigation also fixed CFL omissions for explicit recurrence terms
  and the kinetic-quasineutrality field response. The corrected default
  reduced-case bound is `3.54e-3`; quantitative external parity remains the
  acceptance gate.
  The exact pinned Gyaradax producer is now executable without stored output
  and reproduces `gamma=0.66370834`, `omega=-1.02976757` at GKW
  `kthrho=0.7` (`krho=0.56548668` after the s-alpha `kthnorm` conversion).
  A reference-matched local profile uses the same `32x32x16` grid,
  cell-centered finite differences, Gyaradax's separable upwind
  streaming/trapping backend, and `cosine2` initialization. At final time 40,
  relative growth, frequency, and phase-aligned complex mode-structure errors
  are `2.23e-9`, `1.02e-9`, and `8.84e-10`, with late-window growth drift
  `2.35e-14`. The fused `gkw_igh` path remains available for GKW convention
  parity but is not the discrete operator used by this Gyaradax producer.
- [x] Add a differentiable collision foundation to the production collocation
  residual. The current species-local linearized BGK model preserves discrete
  density, parallel momentum, and energy exactly and contributes to the CFL
  bound.
- [ ] Validate a production Landau/Fokker--Planck collision operator including
  inter-species exchange, and add electromagnetic `A_parallel`/`B_parallel`
  perturbations with independent parity gates. The conserving BGK model is not
  a substitute for this acceptance claim. A differentiable mixed-variable
  `A_parallel` algebraic solve and coupled kinetic `phi`/`B_parallel` solve now
  agree with pinned Gyaradax precomputes. The mixed `g` to physical `f`
  transform and generalized-potential `A_parallel` coefficient also match the
  pinned reference. Linear RHS coupling now includes the generalized-potential
  drive and both `B_parallel` compression terms. Fields, physical-state
  recovery, the isolated electromagnetic increment, and the complete
  one-state RHS agree with pinned Gyaradax when the matched separable-upwind
  backend is selected. One RK4 step and an unnormalized five-step trajectory
  also agree, exercising repeated mixed-state field solves. The mixed-field
  CFL estimate includes the `g`-to-`f` gain and all three algebraic field
  feedback paths and bounds a small exact dense-operator row sum. A
  revision-pinned reduced `8x8x4`, beta-0.01 run reaches final time 10 and
  passes the unchanged growth/frequency/mode/drift gate with errors
  `2.30e-3`, `2.23e-9`, `6.82e-10`, and drift `7.23e-5`. The producer writes
  only to caller-selected scratch storage. The on-demand default resolution
  ladder also passes: its finest `12x12x6 -> 16x16x8` pair changes local
  growth and frequency by `4.64%` and `3.93%`, below the declared 5% gate,
  while both rungs independently retain Gyaradax parity. The production
  `16x16x8 -> 24x24x12 -> 32x32x16` ladder now passes as well. Its finest-pair
  local growth/frequency changes are `2.20%`/`1.21%`, the independent changes
  are `2.20%`/`1.21%`, and finest-rung growth/frequency/mode errors are
  `1.72e-5`/`1.08e-9`/`7.58e-10`. Landau/Fokker--Planck inter-species
  collisions remain open before closing this combined item. A standalone
  nine-point test-particle Fokker--Planck stencil now covers pitch-angle
  scattering, energy diffusion, friction, pair mass/thermal-speed scaling,
  and finite-difference boundary closure. Its single-species stencil and
  action match pinned Gyaradax to `2e-12`, and the frequency remains
  differentiable. The explicit `collision_model="fokker_planck"` path is now
  integrated into the linear residual, and its stencil row-sum enters the CFL
  bound. An opt-in discrete field-particle completion now preserves each species
  density and combined momentum/energy to roundoff and contributes its induced
  norm to the CFL bound. It establishes the algebraic exchange-conservation
  contract. A separate `conservation_model="xu_species_local"` path now matches
  the pinned Gyaradax/GKW Xu momentum and energy factors, quadrature weights,
  and corrected action to `2e-12`; it removes each species' defects locally and
  is not presented as reciprocal inter-species exchange. An experimental
  `conservation_model="pairwise_exchange"` path now retains every ordered
  target/background stencil, couples the two distributions in each unordered
  pair, and preserves each pair's species densities and combined physical
  momentum/energy to roundoff. Its pairwise induced-norm bound enters the CFL
  estimate. This closes the reciprocal data-flow and conservation architecture,
  but not the physics gate. A caller-owned paired run against pinned stella
  revision `564ca09b89904c231421c17c00068a9362061278` now supplies an executable
  native discriminator. Its field-particle-on/off inputs have identical initial
  diagnostics, while one implicit step changes `h2_vs_vpamus` by `31.14%` in
  relative L2 (`g2_vs_vpamus` changes by `0.1996%`). This proves that the compact
  case is sensitive to stella's Laguerre--Legendre field-particle path. A
  non-destructive scratch-build patch now also exports all 2,808 rows of the
  aggregate signed field-particle RHS before the final implicit differential
  inversion. The trace is finite and nonzero (`L2=1.71214e-3`, `9.9778%` of the
  traced input norm); its ion/electron norms are `4.08757e-5`/`1.71165e-3`.
  This closes native signed-action observability, not parity. The remaining
  collision implementation is a common-grid Laguerre--Legendre coefficient and
  action comparison; the existing GKW finite-difference projection must not be
  treated as that model.
- [x] Add the nonlinear ExB pseudo-spectral bracket and 3/2 dealiasing for the
  centered-`kx`, nonnegative-`ky` Hermitian storage convention, integrate it
  with the electrostatic residual, and provide an amplitude-aware nonlinear
  CFL estimate.
- [x] Add candidate-saturation radial/heat-flux window statistics with mean,
  variance, standard error, and relative drift diagnostics.
- [x] Add host-controlled adaptive nonlinear RK4 time advance using the minimum
  of coupled linear and instantaneous ExB CFL bounds. Fixed-step integration
  remains the differentiable trajectory path because adaptive accept decisions
  are nonsmooth.
- [ ] Demonstrate statistically stationary heat flux with resolution/domain
  convergence and pass an independent nonlinear parity gate. The operator,
  adaptive driver, and diagnostics alone do not establish a turbulence claim.
  The collocation diagnostic now evaluates the gyroaveraged non-advective heat
  response `J0 T_s (E_s-3/2) f_s` directly from each evolved species and has a
  JIT/direct-quadrature contract. A reproducible driven nonlinear trajectory,
  stationarity window, resolution/box ladder, and independent GX or stella
  heat-flux comparison are still required. A revision-pinned GX producer
  contract now reads `Diagnostics/HeatFlux_st` from a caller-owned NetCDF file,
  validates its time trace, and emits schema-v1 mean, uncertainty, drift, and
  provenance JSON without storing GX output here. The local trajectory and
  cross-code normalization/parity comparison remain the next implementation.
  A deterministic local reduced producer now connects the driven s-alpha ITG
  setup, fourth-order Fourier hyperdiffusion, adaptive nonlinear integration,
  self-consistent field history, heat response, flux trace, and stationarity
  acceptance into one artifact-free JSON workflow. A `4x4x3`, `3x2` Fourier,
  final-time-0.2 smoke completes with finite output in two accepted steps and
  correctly reports `stationary=false`; it is plumbing evidence, not a
  turbulence result. Schema-v1 local/GX loaders and a shared acceptance layer
  now require stationary low-uncertainty windows, normalized mean-flux parity,
  and stationary finest-pair convergence. Historical `optimal_fusion_native`
  reports remain intentionally incompatible with GX; new acceptance runs must
  select the source-matched `gx_total_energy` moment. Long resolution and box
  ladders are the active numerical blocker. The first driven `8x8x4`, `5x3`
  Fourier runs remain
  nonstationary: at final time 20 the mean native flux is `-3.28e-8` with
  drift `0.335`, and at final time 100 it is `-3.21e-8` with drift `-0.523`
  despite `1.56%` relative standard error. Extending the window does not fix
  the plateau, so initial growth, damping, domain, and saturation behavior must
  be discriminated before launching a resolution ladder.
  Including the validated `ky=0.3` branch (`n_ky=4`) initially exposed a false
  stationarity positive: drift `0.185` and `1.42%` relative error passed while
  potential RMS fell to `52.5%` of its initial value. The local acceptance gate
  now also requires a configurable final/initial potential-RMS ratio (default
  `0.8`), so the same 284-step run correctly fails. The active problem is a
  decaying driven setup, not insufficient statistical precision. A follow-up
  `12x12x6`, `5x4`, hyperdiffusion-0.005 discriminator initially also decayed,
  exposing a setup normalization bug: GX's `fprim=0.8` and `tprim=2.49` had
  been passed directly where the local residual requires `R/Ln` and `R/LT`.
  The producer now applies the documented `Rmaj/Lref=2.77778`, yielding
  `R/Ln=2.222224` and `R/LT=6.9166722`, and records both input and derived
  controls. With that fix, the same final-time-20 run grows: nonzonal
  potential RMS increases `12.59x`, with `ky=0.2`/`0.3` ratios
  `13.40x`/`18.59x`. Flux drift `-3.23` correctly rejects this unsaturated
  growth. The amplitude gate acts on nonzonal potential RMS and reports
  per-`ky` ratios, preventing zonal energy from masking heat-carrying modes.
  Long-time saturation is now the active discriminator before a resolution
  and box ladder. At final time 60, both hyperdiffusion `0.005` and `0.05`
  remain strongly growing in the `5x4` box; expanding to the intended `9x5`
  box delays but does not plateau by time 40 (nonzonal RMS `365x`, flux drift
  `-4.48`). That run also exposed an artificially large randomly seeded zonal
  potential. The producer now defaults the zonal seed to zero, with an explicit
  opt-in fraction, so zonal flow must arise from nonlinear transfer. The
  zero-zonal `9x5` long-time run is the next saturation discriminator.
  That run remains nonstationary at time 60 (nonzonal RMS `5822x`, flux drift
  `-3.74`) despite strong self-generated zonal flow. The local producer now
  exposes an opt-in conserving-BGK frequency whose stiffness participates in
  adaptive CFL control, allowing a velocity-dissipation discriminator. It is
  deliberately off by default and cannot satisfy external parity in place of
  GX's documented hypercollision model. A `nu=0.1` BGK run changes the time-60
  result only slightly (nonzonal RMS `5664x`, drift `-3.68`), ruling out weak
  velocity damping as the dominant reduced-run blocker. The next structural
  implementation was a shear-consistent `kx` grid with twist-and-shift parallel
  connectivity. That path is now the producer default: a public cell-centered
  finite-difference parallel grid feeds the existing GKW upwind boundary
  stencil, radial spacing follows `q*shat*ky_min/(eps*ikxspace)`, and the
  boundary shift scales with `ky/ky_min`. The former periodic-chain path is an
  explicit historical diagnostic. Long-time stationarity and GX parity on the
  corrected topology remain open. The first corrected `12x12x6`, `9x5`,
  hyperdiffusion-0.05 run reaches time 60 in 1,714 adaptive steps. Twist-and-
  shift reduces peak native flux from the periodic-chain result `43.2` to
  `0.314` and final nonzonal RMS from `1.33` to `0.102`, but drift `-4.71`
  still rejects the growing window. A longer-time and wider-box ladder, with
  explicit runtime budgeting, is the next numerical gate. The producer now
  supports caller-owned schema-v1 checkpoints with absolute time and an exact
  topology/physics contract, so extensions do not recompute the transient and
  incompatible restarts fail closed. Resumed reports summarize only the new
  segment. The stationarity gate now also requires at least 100 samples and
  duration 10 by default, closing a false positive found in a two-step restart
  smoke. Extending the corrected time-60 state is the next numerical run.
  The nonlinear driver now compiles the CFL evaluation and complete RK4 step
  while keeping adaptive acceptance host-controlled. A production-shape
  `12x12x6`, `9x5` time-20 segment completes in `49.36 s` and 572 steps with a
  valid checkpoint, making staged long-time runs practical.
  Staged extension through time 220 shows intermittent nonlinear bursts rather
  than a plateau. The time-140-to-220 segment's final 40-unit window has mean
  flux `-30.80`, relative error `0.895%`, and drift `0.633`; its two halves
  average `-35.46` and `-26.16`. The gate now also fits nonzonal-potential
  logarithmic growth over the candidate window and requires magnitude below
  `0.02`, preventing a slowly evolving field from passing a coincidentally
  flat flux interval.
  The next checkpoint segment, time 220-to-300, is the first passing local
  stationary candidate: its final 40-unit window has mean native flux
  `-25.7841`, relative standard error `0.462%`, drift `0.0572`, and nonzonal
  potential growth `-1.93e-3`. Candidate half means are `-27.235` and
  `-24.336`; the RMS ratio is `0.879`. This establishes one coarse local rung,
  not convergence or GX parity. The next acceptance work is an independently
  initialized wider/resolved rung and the native-to-GX normalization contract.
  A second time-300-to-400 candidate has a consistent mean `-26.256` (within
  `1.8%` of the first) but fails drift at `-0.450`; its half means are
  `-23.145`/`-29.363`. Across time 220-to-400 the mean is `-27.100`, while the
  final 90-unit half has mean `-25.251`, relative error `0.658%`, and drift
  `-0.068`. Producer reports now retain compact nonzonal-RMS histories beside
  flux histories, enabling a checkpoint-boundary-independent merged-window
  gate without storing phase-space trajectories.
  A schema-v1 segment merger now implements that contract: it requires
  contiguous absolute time, removes only duplicate boundary samples, rejects
  normalization or grid/physics changes, and recomputes uncertainty, drift,
  amplitude ratio, and fitted field growth over one merged window. New-format
  segments can therefore feed the existing convergence/parity layer directly.
  Both local and GX producers now record an explicit top-level stationarity
  decision. The shared loader requires and preserves that decision, preventing
  a producer-rejected duration, amplitude, or growth window from being accepted
  later using drift and uncertainty alone. A standalone campaign evaluator now
  requires separate two-or-more-rung resolution and domain ladders plus an
  independently stationary reference comparison, and exits nonzero unless all
  three gates pass. Source tracing against pinned GX established its
  gyroaveraged total-energy moment, Hermitian factor, and s-alpha flux weight;
  the local producer now exposes that exact `gx_total_energy` diagnostic with
  the `gx_Q_over_Q_GB` label while retaining the historical non-advective heat
  moment separately. The checkbox remains open because the stationary
  resolution/domain rungs and pinned GX run are scientific evidence that
  cannot be replaced by acceptance plumbing.
  A source-matched regeneration invalidated the earlier single-window success
  as acceptance evidence. Changing the checkpoint schedule from `0->20->60`
  to `0->60` perturbs one adaptive step and eventually produces a distinct
  chaotic realization. Its time-220-to-300 GX-energy window has mean
  `-26.6568`, block relative error `6.47%`, and drift `-0.2204`; merging through
  time 400 gives mean `-24.2323`, 17 physical-time blocks, block relative error
  `7.12%`, drift `-0.6913`, and nonzonal growth `0.00424`. It correctly fails.
  The local/GX-energy and non-advective-heat traces agree to roundoff, ruling
  out residual particle flux as the discrepancy. Local and GX summarizers now
  use physical-time-weighted block means rather than timestep-count-weighted
  means and naive independent-sample errors. New checkpoints fail closed
  without schema-v1 trajectory lineage and carry the originating seed,
  amplitude, zonal fraction, and complete segment-end schedule into reports.
  The next numerical gate is robust stationarity across lineage-identified
  coarse realizations before spending resources on resolved/domain ladders.
- [ ] Extend to full equilibrium-shape optimization only after the standalone
  geometry, W7-X parity, convergence, timing, and gradient gates pass.

Acceptance status: **implementation foundations advanced; scientific gate
remains open**. The standalone suite covers the model collision operator,
nonlinear discretization, adaptive control, and diagnostics. Priority 5 cannot
be marked complete until the inter-species collision, stationary nonlinear-flux,
and unrestricted shape-optimization claims named above pass.

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
