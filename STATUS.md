# STATUS

Last updated: 2026-08-06

## Executive Summary

`optimal-fusion` is now a standalone JAX package at its core boundary. An exact
development sync contains no MHD or reference-code provider, and the complete
default suite, package build, fresh wheel install, and import all pass without
sibling repositories. External validators are opt-in, revision checked, and
receive explicit source, executable, equilibrium, or generated-data paths.

The public MHD geometry boundary is schema-versioned and provider neutral.
Priority 2 now supplies optional live DESC, VMEC++, and GVEC adapters. VMEC++
loads its installed named W7-X design, runs it, and constructs the complete
flux-tube geometry contract from in-memory Fourier output without a repository
`wout` file. The externally validated W7-X scientific gate remains open.

### 2026-08-06: Priority 3 Native Velocity and Mirror Discriminator

- Added a provider-neutral, zero-free midpoint parallel-velocity grid with
  symmetric Simpson weights and rescaled Gauss-Laguerre mu quadrature. The W7-X
  scan selects it without storing provider state or equilibrium artifacts.
- Replaced its ill-conditioned global derivative with bounded local polynomial
  stencils. The production CFL estimate improved from `7.8e-9` to `5.5e-2`,
  making `dt=0.02` valid.
- Added an opt-in physical `2*pi*B*dv*dmu` measure consistent with the solver's
  normalized Maxwellian and the verified stella replay normalization.
- Added a Strang-split, semi-Lagrangian mirror advance with a zero-incoming
  velocity boundary. This removed the spurious explicit-mirror branch: a
  `t=20` precheck moved growth from approximately 1.0 to 0.00894.
- Ran the full `ky=0.3`, 32x8, `t=200` discriminator in `/private/tmp`. It
  produced `(gamma, omega)=(0.01794, 0.01246)` with growth/frequency errors
  `0.00832/0.01531`, both below 0.02. Profile error remains 0.1634 and the
  late-window growth delta is 0.0256, so Priority 3 remains open for converged
  `t=500` per-ky references, implicit parallel streaming, and cubic mirror
  interpolation.
- Verified the complete standalone x64 suite after these changes: 376 passed
  and 20 external tests were deselected.
- Added source-default cubic mirror interpolation and a differentiable
  implicit-midpoint parallel streaming propagator. The latter is precomputed
  per parallel-velocity node and Strang-split around the explicit field/RHS
  terms.
- Repeated the native `ky=0.3` gate at `t=500` against the converged pinned
  stella output. With `dt=0.02`, growth/frequency/profile errors are
  `0.00395/0.00064/0.11498`; matching stella's `dt=0.1` gives
  `0.00396/0.00062/0.11489`. Scalar parity is closed and timestep dependence is
  negligible, but complex-profile parity remains open.
- Profile audits ruled out z reversal, complex conjugation, periodic shifts,
  and a simple linear phase gauge. The residual blocker is the source-specific
  extended-domain near-centered streaming response coupled to the
  quasineutrality field response; the current implicit spectral propagator does
  not claim equivalence to that algorithm.
- Added a differentiable Schur-complement response that advances streaming,
  parallel electric-field drive, and quasineutrality in one implicit step. A
  dense small-system regression verifies exact equivalence to the coupled
  midpoint equations, and a precomputed field inverse avoids a dense solve at
  every split stage.
- Added an opt-in `stella_implicit` diagnostic with sign-dependent `0.51/0.49`
  spatial centering and `0.51/0.49` time weights. The converged `t=500` W7-X
  run in `/private/tmp` gives `ky=0.3` `(gamma, omega)=(0.01516, 0.05580)`,
  within scalar tolerances of the stella reference, but the phase-aligned
  profile error is `0.13913`; the all-ky maximum profile/frequency errors are
  `0.17676/0.03670`. The streaming-only spectral path therefore remains the
  validated default. Source inspection narrows the remaining mismatch to
  stella's centering of z-dependent Maxwellian/geometry factors and its exact
  inhomogeneous/homogeneous delta-phi response construction.
- Verified the complete standalone x64 suite after the coupled-response work:
  378 tests pass and 20 explicitly external tests are deselected.

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

Priority 2 added declared optional MHD extras, named installed DESC and VMEC++
paths, direct VMEC full/half-grid Fourier geometry, live GVEC PEST evaluation,
and a named W7-X reduced-scan path. The VMEC++ fork now packages
`w7x-standard` behind a public Python API. Direct VMEC++ W7-X geometry is
checked term by term against independent same-source stella and matched
same-surface GX/GIST results. GVEC is also checked against VMEC++ on GVEC's
tracked VMEC-initialized W7-X equilibrium. Priority 2 is complete.

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

- schema-v2 `GeometryRequest`, `GeometryProvider`, `GeometryResult`, and static
  metadata covering coordinates/units, signs, topology/linking, endpoint
  policy, normalization, differentiability, and provenance;
- `PhysicalFluxTubeGeometry` carrying `alpha`, `iota`, shear, `nfp`, field
  periods, topology, normalization, and provider identity as a JAX PyTree;
- synthetic and generic physical-array providers; DESC object/path/named
  provider; direct VMEC++ output/input/path/named provider; GVEC state/file
  provider; GX/GIST eik and stella `.geometry` validation providers;
- strict provider-boundary validation and explicit external cache round trips;
- one physical-to-internal map for `F`, `G`, `E_y`, `D_x`, `D_y`, and metrics;
- a distinct physical `equilibrium_drive_scale`; the stella adapter supplies
  geometry-header `flux_fac` instead of conflating diamagnetic drive with
  local grad-B drift geometry;
- a reduced scan and solver-construction test that are provider independent;
- packaged VMEC++ `w7x-standard` lookup followed by `vmecpp.run(...)` and an
  in-memory field-line transformation with no canonical NetCDF/GX/stella hop;
- fixed-topology JAX gradient coverage from a DESC-compatible continuous
  equilibrium parameter through sampled geometry and the reduced objective.

Not present today:

- a stable named W7-X constructor in GVEC (the provider accepts live state and
  parameter-file inputs, and the direct named W7-X path is VMEC++);
- an end-to-end gradient through an installed production MHD solve (native
  VMEC++ and GVEC boundaries are currently non-differentiable);

The pinned VMEC++ fork now installs the W7-X standard input as package data and
exports `named_configuration("w7x-standard")` and `named_configurations()`.
The optimal-fusion manifest pins that fork commit. DESC resolves named examples
through its installed API; GVEC presently requires a state or parameter file.

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
- the retained stella trace summary is now format v3 with explicit RHS-call
  labels, velocity quadrature, quasineutrality, and normalization; the complete
  weighted array contract runs but numerical parity fails;
- production convergence, CPU timing, and MHD design optimization remain
  blocked behind external W7-X parity.

The canonical W7-X geometry path no longer depends on a committed equilibrium
or derived geometry file. Compact historical stella/GX contracts remain only
as independent regression evidence for the still-open scientific parity gate.

## Repository Audit

### Standalone acceptance on 2026-08-05

```text
ruff check src tests examples scripts: PASS
uv sync --extra dev: PASS (exact provider-free environment)
uv pip check --python .venv/bin/python: 25 packages compatible
JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q:
  342 passed, 18 external tests deselected in 452.94 s
scripts/package_smoke_test.py --python 3.13:
  sdist build, wheel build, fresh wheel install, and import: PASS
```

The prior checkout-dependent failures were replaced with synthetic contracts
or opt-in external tests. The default suite has no source-tree or executable
availability skips; external tests are deselected by the registered `external`
marker. NetCDF4 is
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

1. Resume the stella `ky=0.3` weighted term-array parity gate, followed by
   convergence, CPU timing, and real MHD optimization gradients.
2. Tighten the legacy stella and finite-element GVEC metric/drift comparison
   envelopes as the independent reference pipelines mature.
3. Add a named GVEC W7-X hook if its installed API gains a stable constructor.

## Current Risks

- External integrations can drift from their pinned forks or fail on local
  native toolchains even though the provider-free suite remains green.
- On the current macOS environment, loading DESC, GVEC, and VMEC++ sequentially
  in one pytest process causes VMEC++ to segfault inside its native run. Each
  provider's isolated integration test passes, so external native-provider
  suites must remain process-isolated until the shared-library conflict is
  resolved.
- Native VMEC++ and GVEC calls do not currently preserve JAX gradients into
  their design variables; production design optimization therefore needs a
  differentiable provider path (currently DESC-compatible) or custom rules.
- Direct stella term comparison has excellent normalized `B` agreement but a
  deliberately recorded 30% envelope for equal-arc metric/drift terms; tighter
  same-grid cross-code parity remains work.
- The fixed-step initial-value objective may change branches or become
  ill-conditioned during design optimization; branch/gradient checks are not
  yet implemented for real equilibria.
- Production CPU scaling on approximately 100 CPUs has not been demonstrated.
- Collisions, electromagnetic physics, nonlinear turbulence, kinetic-electron
  TEM production validation, and full shape optimization are deferred.

## Round Log

### 2026-08-05: Priority 3 Weighted stella Trace

- Built a non-destructively patched stella copy from revision
  `564ca09b89904c231421c17c00068a9362061278` and ran the matched W7-X
  `ky=0.3`, `t=200` explicit-term trace using the scratch executable recorded
  in the compact result provenance.
- Replaced the retained v1 summary with a v2 record covering 1,382,403 rows,
  all required RHS terms, `wgts_vpa`, and z-dependent `wgts_mu` on stella's
  257×32×8 raw grid. The 329 MiB raw trace remains outside the repository.
- Made the summarization command require explicit stella source and executable
  paths and record the exact source commit.
- Applied the shared endpoint-excluded convention to trace arrays by dropping
  stella's upper periodic z endpoint; the comparison grid is now 256 points,
  matching the solver geometry grid.
- Added an opt-in solver selected-mode array archive with canonical
  `(z, vpar, mu)` ordering, explicit quadrature, distribution, potential, all
  split RHS terms, total RHS, quasineutrality numerator/denominator, and
  accumulated normalization. Large archives are required to remain outside
  the repository.
- Added the cross-grid adapter contract: separable complex interpolation in
  `z/vpar/mu`, strict rejection of extrapolation, target-grid quadrature, and
  weighted errors both before and after one global complex amplitude/phase
  alignment.
- Emitted an external solver archive at the trace-matched `t=199.9` and added
  streaming loading of the 329 MiB raw stella trace. The partial comparison
  covers distribution, streaming/field-drive, mirror, drift/field-drive,
  equilibrium drive, and total RHS for all three inferred stella RHS calls.
- Restricted comparison to the common target domain (256 z, 16 vpar, 7 mu
  nodes). Distribution-derived complex alignment leaves relative L2 errors
  near one for every available quantity and every inferred call, confirming a
  structural/convention mismatch rather than a scalar normalization offset.
  Full parity remains blocked because v2 does not label its three calls and
  lacks stella quasineutrality numerator/denominator and normalization arrays.
- Audited the parallel boundary contract: stella resolves its default to
  zero/unconnected boundaries for the nonzonal `ky=0.3` mode, while the
  original solver balance uses a periodic Fourier derivative. A matched-time
  open-chain GKW-upwind discriminator did not close the gap: its growth and
  frequency were approximately `-3.07e-4` and `-9.94e-2`, and its best
  distribution-aligned array errors stayed near one. The periodic assumption
  is therefore a real contract mismatch, but simply switching to the existing
  GKW open stencil is not a stella-parity fix.
- Rebuilt and reran the pinned stella trace in format v3. The 332 MiB raw trace
  remains external; its compact summary covers 1,383,948 rows, three explicit
  RHS calls, velocity weights, quasineutrality numerator/denominator, and native
  state scale. The complete 30-row weighted comparison is now contract-ready
  but fails its 0.1 relative-L2 tolerance (maximum about 1.004). After applying
  the documented opposite denominator sign, quasineutrality-denominator
  arrays agree within `5e-4` relative L2; the distribution, phi, numerator, and RHS
  structures remain mismatched.
- Verified the complete standalone suite with x64 enabled: 352 tests passed and
  20 opt-in external tests were deselected. Focused Ruff checks also pass.
- Added a same-state replay gate that interpolates each explicitly labeled
  stella distribution and potential onto a contained 256×16×4 solver grid and
  evaluates periodic and open-chain solver operators without a fitted state
  scale. This separates operator errors from independently evolved eigenmodes.
- Audited the pinned stella source and added a non-production coefficient
  discriminator: open streaming, the traced mirror orientation, the stella
  magnetic-drift factor 1/2, geometry-header `flux_fac`, and stella's
  Maxwellian normalization. It reduces the total-RHS relative L2 error from
  about 1.32 to 0.255. Equilibrium drive reaches 0.068, while streaming is
  0.172, magnetic drift 0.173, and mirror force 0.398. The denominator remains
  consistent to `4.6e-4` after its documented sign conversion, but the
  distribution moment normalization remains unresolved. At this stage the
  discriminator remains separate from production except for the independently
  confirmed equilibrium-drive scale described next.
- Promoted the first confirmed convention into geometry schema v2: physical
  providers may now supply `equilibrium_drive_scale`, and the stella adapter
  maps its header `flux_fac` directly. Other providers retain their prior
  drift-derived fallback pending native-coordinate validation, and external
  schema-v1 caches must be regenerated.
- Verified the schema-v2 migration with the complete standalone x64 suite:
  358 tests passed and 20 opt-in external tests were deselected.
- Added a 32×4 same-state replay whose parallel-velocity nodes exactly match
  stella's 32 nodes from -3 to 3. It improves the source-derived discriminator's
  streaming error to 0.115, magnetic drift to 0.164, and total RHS to 0.237,
  while mirror remains about 0.411. The mirror gap is therefore not explained
  by the earlier 16-node interpolation and remains the leading term-level
  blocker.
- Reproduced stella's explicit mirror derivative exactly: sign-dependent
  third-order upwinding, first/third-order zero-value boundary closures, and
  the same uniform velocity spacing. The mirror raw error only falls from
  about 0.411 to 0.396, although its best-fit structural error improves to
  0.204 with a scale near 0.74; total RHS improves to 0.231. The remaining gap
  is therefore in the coefficient/geometry normalization rather than the
  velocity stencil. The committed stella geometry table carries only about
  four significant digits, so the next trace must expose native in-memory
  mirror/drift coefficients rather than reconstructing derivatives from that
  lossy table.
- Extended the external trace to v4 with one native in-memory snapshot of the
  mirror, drift, equilibrium-drive, and streaming coefficients. The raw 413 MiB
  trace remains outside the repository; its compact summary records all five
  coefficient arrays and their pinned stella provenance.
- Made the streaming trace loader accept interleaved coefficient records from a
  single explicitly labeled RHS call. Using the native mirror coefficient with
  the source-matched upwind stencil reconstructs stella's native 256×32×8
  mirror term within 0.004 relative L2. The 32×4 solver-grid comparison is 0.243,
  so its residual is now attributed to interpolating a coefficient-state
  product onto a different mu grid rather than to the mirror convention itself.
- Extended the external trace contract to v5 with stella's native gyroaverage
  `J0`. Combining that array with the traced distribution, `wgts_vpa`, and
  z-dependent `wgts_mu` reconstructs all three quasineutrality numerators within
  `1.2e-15` relative L2. This rules out an unknown amplitude normalization: the
  solver-grid numerator mismatch comes from mapping stella's native velocity
  measure and FLR product to a different uniform-mu quadrature.
- Added a provider-neutral arbitrary-node velocity-grid interface and optional
  z-dependent phase-space measure for quasineutrality. The native 256×32×8
  replay can therefore retain stella's quadrature without committing W7-X state
  or adding a solver-specific production dependency.
- Reconstructed stella's traced drift algebra to machine precision and its
  257-point source parallel stencil in the diagnostic adapter. Together with
  the native mirror stencil, gyroaverage, and measure, the same-state acceptance
  case now passes: its maximum RHS relative L2 error is `0.0439222`, below the
  `0.1` gate. The largest remaining term is streaming at about `0.0392`; mirror
  is `0.00370`, equilibrium drive `0.000847`, and drift is at roundoff.
- Identified and fixed a production drift error: combining grad-B and curvature
  geometry before multiplying by `(v_parallel^2 + mu B)` created unphysical
  cross terms. Physical providers now preserve the two components and evaluate
  `v_parallel^2 D_curvature + mu B D_gradB`; legacy analytic geometry retains
  its combined-coefficient fallback. The mapped mirror RHS orientation was
  corrected at the same provider boundary. Focused geometry, optimization,
  primitive, RHS, quasineutrality, and replay tests pass.
- Reran the corrected production mode-structure case at `t=200`. Growth remains
  inside the 0.02 tolerance, while maximum frequency/profile errors improve to
  `0.1312` and `0.1491`; the gate remains open. The existing GKW open-upwind
  discriminator is not suitable for stella: at `ky=0.3` it gives growth/frequency
  near `0.086/-1.445`.
- Built and ran the clean pinned stella dependency against the sibling GX W7-X
  equilibrium without storing its output in the repository. It exactly
  reproduces the committed `t=200` fixture, but the reference is transient:
  over `t=100–200`, the `ky=0.3` omega/growth standard deviations are about
  `0.072/0.074`. At `t=500`, the branch is stable near
  `(gamma, omega)=(0.01754, 0.04638)`, whereas a matched converged solver run is
  near `(-0.00013, -0.05497)` with profile error `0.1254`.
- Hardened stella fixture export with a default 0.02 two-half-window omega
  convergence check. Unconverged references now fail export unless explicitly
  requested for diagnostics. The next production discriminator must address
  native 32×8 velocity quadrature/FLR and stella's implicit parallel treatment.

### 2026-08-05: Priority 2 Live MHD Providers

- Declared selective `desc`, `vmecpp`, `gvec`, and combined `mhd` extras while
  retaining exact fork pins in the dependency bootstrap manifest.
- Made DESC lazy and source-tree independent, added installed named-example
  lookup, opt-in real W7-X evaluation, and a fixed-topology JAX gradient test.
- Updated the VMEC++ fork to package and expose `w7x-standard`; pinned that fork
  and verified a complete live named W7-X run without a stored equilibrium.
- Added direct in-memory VMEC Fourier reconstruction using VMEC's PEST mapping,
  staggered radial grids, orientation, normalization, metrics, grad-B drifts,
  and curvature drifts. User `wout` import remains an optional secondary path.
- Added a live GVEC PEST adapter for state/parameter-file inputs and verified it
  against a revision-checked sibling example.
- Routed the reduced scan through
  `--geometry-provider vmecpp --configuration w7x-standard` and recorded full
  provider/version/revision provenance.
- Cross-checked every required direct-VMEC term against an independent
  same-source stella W7-X geometry: normalized `B` is below 1% relative L2;
  equal-arc metrics and drifts have correct scale and remain within the explicit
  30% legacy-grid envelope.
- Added a matched same-surface GX/GIST check for all nine geometry terms,
  shear, field periods, and endpoint policy; all term errors remain below 20%.
- Normalized live GVEC fields to the common minor-radius, edge-flux, and field
  references, including explicit flux-coordinate orientation handling.
- Compared all live GVEC physical terms with direct VMEC++ on GVEC's tracked
  VMEC-initialized W7-X equilibrium; invariant terms agree within 8%, while the
  full finite-element projection contract remains within its documented 75%
  pre-minimization envelope.

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
