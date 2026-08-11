# TODO: JAXFluxTubeGK

Last reviewed: 2026-08-10

## Goal

Build JAXFluxTubeGK (`jax-fluxtube-gk`) as a standalone JAX package for
differentiable, local flux-tube gyrokinetics in magnetic-confinement design.
The package must own its solver, geometry data contract, fixtures, and core
tests. MHD and reference gyrokinetic codes must remain separately installable
providers or validation tools, not copied source trees or implicit runtime
dependencies.

The trusted, externally validated linear electrostatic W7-X milestone and the
reduced fixed-topology design loop are complete. Production collisions and
linear electromagnetic fields are implemented and independently exercised.
The active scientific milestone is electrostatic nonlinear-turbulence
acceptance: stationary resolution/domain ladders and a revision-pinned GX
comparison. Nonlinear electromagnetic evolution and unrestricted production
equilibrium-shape optimization remain later milestones.

Keep the workflow simple:

1. Specify physics and numerics in `tex/main.tex`.
2. Implement reusable package functionality in `src/jax_fluxtube_gk/`.
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
  repositories/reference material outside `jax-fluxtube-gk`.
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
  location relative to the `jax-fluxtube-gk` checkout.
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
- [x] Stop tracking `src/jax_fluxtube_gk.egg-info/`; build metadata must be
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

### Objective and current boundary

Priority 5 is no longer blocked on a missing local operator. The production
CPU velocity representation, kinetic-electron physics, linear electromagnetic
response, Landau/Fokker--Planck collision path, nonlinear ExB bracket,
adaptive driver, restart contract, diagnostics, and fail-closed acceptance
tools are implemented and tested. The remaining work is long-running
scientific acceptance evidence plus the design demonstration that depends on
that evidence.

The authoritative remaining-work checklist is:

- [ ] Obtain a stationary `129x65`, `ky_min=0.00625` local nonlinear report
  and finish the bandwidth-preserving domain-convergence ladder.
- [ ] Run the matched revision-pinned GX case on CUDA/native capacity and pass
  the independent nonlinear heat-flux parity gate.
- [ ] Run the complete campaign evaluator with stationary resolution, domain,
  three-lineage, and GX reports; retain its passing schema-v1 report outside
  the repository.
- [ ] Attempt unrestricted equilibrium-shape optimization only after every
  prerequisite gate passes, then add a checkpointed end-to-end design run.

Do not mark Priority 5 complete from a flat-looking trace, a single seed, a
local-only comparison, or a reduced tolerance. Generated states, NetCDF files,
and JSON reports remain caller-owned scratch artifacts and must not be
committed.

### Recommended execution sequence

#### 1. Finish the active `129x65` CPU trajectory

The latest validated state is
`/private/tmp/p5-domain-next-seed19-t140.npz`: a finite complex128
`(12,6,12,129,65)` checkpoint at time 140 with exact seed-19 lineage through
`[20,40,60,80,100,120,140]`. The time-120-to-140 late candidate passes every
physical statistic (mean `-7.03914`, `0.259%` block error, drift `0.0401`, RMS
ratio `0.994`, and field growth `-9.63e-5`) but remains fail-closed with only 37
samples, one block, and duration `9.914`. The time-80-to-140 merge has 109
samples and acceptable error/amplitude/growth, but narrowly fails drift
(`-0.2389`) and contains only five complete blocks. Continue from the time-140
state without changing the contract:

```console
JAX_ENABLE_X64=1 uv run python examples/run_nonlinear_heat_flux.py \
  --output /private/tmp/p5-domain-next-seed19-t140-t160.json \
  --restart-from /private/tmp/p5-domain-next-seed19-t140.npz \
  --checkpoint-output /private/tmp/p5-domain-next-seed19-t160.npz \
  --final-time 160 --n-z 12 --n-vpar 12 --n-mu 6 \
  --n-kx 129 --n-ky 65 --ky-min 0.00625 \
  --flux-moment gx_total_energy --seed 19 --diagnostic-stride 8
```

Continue in bounded 20-time-unit segments. After each segment:

1. Verify that the state is finite complex128, the checkpoint time is exact,
   and `trajectory_lineage.segment_end_times` contains the full schedule.
2. Inspect the segment, but do not accept its mean when `stationary=false`.
3. Merge only contiguous post-turnover segments with
   `scripts/merge_nonlinear_heat_flux_segments.py`; do not include startup
   growth merely to increase the sample count.
4. Stop extending only when a merged late window reports `stationary=true`.

At the current diagnostic cadence, two 20-unit segments provide only about 73
candidate samples and four blocks after the merger applies its late-half
window. Three segments provide 109 samples but only five complete blocks
because their late window is just under 30 time units. Reach time 160 and merge
four post-turnover segments so the candidate window can contain eight blocks:

```console
uv run python scripts/merge_nonlinear_heat_flux_segments.py \
  /private/tmp/p5-domain-next-seed19-t80-t100.json \
  /private/tmp/p5-domain-next-seed19-t100-t120.json \
  /private/tmp/p5-domain-next-seed19-t120-t140.json \
  /private/tmp/p5-domain-next-seed19-t140-t160.json \
  --output /private/tmp/p5-domain-next-seed19-t80-t160-merged.json
```

If that window fails a physical statistic, extend from time 160 and shift the
merge start forward as needed. Never concatenate noncontiguous reports or
reports with different contracts.

The unchanged stationarity requirements are at least 100 samples, duration
10, six physical-time blocks, relative block error at most `0.10`, absolute
window drift at most `0.20`, candidate nonzonal-potential RMS ratio at least
`0.8`, and absolute fitted field growth at most `0.02`.

#### 2. Decide the domain-convergence gate

Compare the accepted `129x65` mean against the stationary `65x33` result
`-4.41179645` from
`/private/tmp/p5-domain-finest-seed19-t100-t200-merged.json`. Normalize the
absolute difference by the finer `129x65` mean. The gate passes at or below
`15%`.

If it fails, do not adjust the tolerance or dissipation. Add the next
bandwidth-preserving rung by halving `ky_min` and increasing the Fourier grid
to `257x129`, while holding `n_z=12`, `n_vpar=12`, `n_mu=6`, the physical
profiles, boundary model, hyperdiffusion, flux moment, and seed fixed. Bootstrap
that rung from its own seed-19 initial state and repeat the checkpointed
stationarity procedure. Do not interpolate a turbulent checkpoint between
Fourier grids and call it the same lineage.

#### 3. Obtain independent GX parity

This gate needs a CUDA-capable machine and the pinned GX revision
`bc2fe5523c23e3d0198181a3e3b7c8a482e25ba5`. Prepare the external run with
`scripts/prepare_gx_nonlinear_heat_flux_run.py`, execute the command it prints,
and summarize the caller-owned NetCDF output with
`scripts/summarize_gx_nonlinear_heat_flux.py`. Keep the generated GX input,
manifest, NetCDF file, and summary outside this repository.

```console
uv run python scripts/prepare_gx_nonlinear_heat_flux_run.py \
  --gx-root /path/to/gx --gx-executable /path/to/gx/bin/gx \
  --output-dir /scratch/gx-cyclone-nonlinear

# Run the command printed by the preparation helper on the CUDA host, then run
# its printed summarization command against the resulting NetCDF file.
```

The GX report must be stationary and must match the finest accepted local case
in geometry, profiles, Fourier box, linked boundary, electrostatic physics,
hyperdiffusion, numerical resolution, and `gx_Q_over_Q_GB` normalization. The
local/GX mean difference must be at most `20%`. CUDA comparison is currently
deferred, not passed.

#### 4. Run the final nonlinear campaign gate

Use `scripts/validate_nonlinear_heat_flux_campaign.py` with:

- at least two stationary phase-space resolution reports;
- at least two stationary bandwidth-preserving domain reports;
- at least three stationary reports with distinct initialization roots; and
- the stationary pinned GX report.

The existing `16x16x8 -> 20x20x10` resolution pair passes at `3.88%`, and the
existing three-root coarse ensemble passes with `2.04%` maximum deviation.
They remain usable only if their case contracts match what the evaluator
requires. The evaluator is authoritative: it must exit successfully and emit
`passed=true`. It also verifies that resolution rungs change only phase-space
resolution, domain rungs change only the physical box while preserving
bandwidth, every producer accepted its own stationarity window, lineage roots
are distinct, and the GX provenance is pinned and case-matched.

```console
uv run python scripts/validate_nonlinear_heat_flux_campaign.py \
  --resolution-report /scratch/resolution-coarse.json \
  --resolution-report /scratch/resolution-fine.json \
  --domain-report /scratch/domain-coarse.json \
  --domain-report /scratch/domain-fine.json \
  --lineage-report /scratch/lineage-seed-18.json \
  --lineage-report /scratch/lineage-seed-19.json \
  --lineage-report /scratch/lineage-seed-20.json \
  --reference-report /scratch/gx-cyclone-nonlinear.json \
  --output /scratch/nonlinear-campaign.json
```

#### 5. Demonstrate unrestricted design optimization

Only after steps 1--4 pass, connect the accepted nonlinear objective to the
existing MHD geometry/design interface. Preserve the fixed topology and
remeshing contract, run a checkpointed end-to-end optimization, and verify the
objective gradient against finite differences at representative design
points. Record solver/provider revisions and commands, but keep large geometry,
state, and solver-output artifacts outside the repository.

### Definition of done

Priority 5 is complete only when the repository can point to a passing final
campaign report and a reproducible unrestricted design run. Collision and
linear-electromagnetic implementation are already complete for the accepted
scope. Nonlinear electromagnetic evolution, multispecies implicit-response
shortcuts beyond their declared scope, and CUDA performance comparisons are
future physics/performance extensions; they are not substitutes for, and do
not silently block, the electrostatic acceptance sequence above.

<details>
<summary>Historical implementation and numerical campaign log</summary>

The material below preserves the detailed evidence trail. It is not the
authoritative remaining-work checklist; follow the sequence above.

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
- [x] Validate a production Landau/Fokker--Planck collision operator including
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
  estimate. A distinct experimental `conservation_model="reciprocal_exchange"`
  path now makes each target's momentum/energy response depend on its partner's
  defects. It preserves pair density, momentum, and energy to roundoff under
  JIT and differentiation, and its conservative CFL bound exceeds the exact
  dense operator row sum. This closes the reciprocal data-flow and conservation
  architecture, but not the physics gate. A caller-owned paired run against
  pinned stella revision `564ca09b89904c231421c17c00068a9362061278` now
  supplies an executable native discriminator. Its field-particle-on/off inputs
  have identical initial diagnostics, while one implicit step changes
  `h2_vs_vpamus` by `31.14%` in
  relative L2 (`g2_vs_vpamus` changes by `0.1996%`). This proves that the compact
  case is sensitive to stella's Laguerre--Legendre field-particle path. A
  non-destructive scratch-build patch now also exports all 2,808 rows of the
  aggregate signed field-particle RHS before the final implicit differential
  inversion. The trace is finite and nonzero (`L2=1.71214e-3`, `9.9778%` of the
  traced input norm); its ion/electron norms are `4.08757e-5`/`1.71165e-3`.
  This closes native signed-action observability, not parity. The remaining
  collision implementation is a common-grid Laguerre--Legendre coefficient and
  action comparison; the existing GKW finite-difference projection must not be
  treated as that model. The signed producer now isolates all four native
  collision channels with identical input states. Their RHS L2 norms are
  `4.00831e-5` (ion-ion), `9.39860e-7` (ion-electron), `1.71045e-3`
  (electron-electron), and `1.30298e-6` (electron-ion). The isolated sum agrees
  with the all-channel implicit result to `8.59e-5` relative L2, providing
  pair-resolved targets for the common-grid implementation. The scratch patch
  now also resolves all eight `(l,m,j)` contributions in every channel. Each
  signed component sum reconstructs its aggregate action exactly. In the full
  case, `(l=0,m=0,j=1)` dominates at `1.712095e-3`; the only other nonzero
  terms are `(l=1,m=+/-1,j=0)` at `4.070812e-7` each. This closes native
  coefficient-action decomposition. A provider-neutral JAX low-rank contract
  now applies pair/component driver and response tensors under JIT and
  differentiation with a dense-checked CFL bound. Native `psi` and response
  factors replay through that local kernel with relative L2 errors from
  `7.36e-14` to `1.24e-10` across the full and four isolated channels. The
  response tensor is now constructed locally from the velocity grid, magnetic
  field, pair frequencies, species masses, gyroaverages, and `Delta_j` values.
  A 44,928-row primitive trace verifies the raw native factor product exactly
  and the independent JAX response at `1.84e-14` relative L2. The analytic
  incomplete-gamma construction of `Delta_0` also matches stella at
  `4.25e-13`. The higher-`j` orthogonalization recurrence now uses the traced
  `integrate_vmu` product weights and stella's mass-ratio self-adjointness
  branch; all `j=0,1` values agree across the same trace at `7.84e-13` scaled
  L2. The normalized moment driver is now built locally from the recursive
  reversed-pair `Delta_j`, Maxwellian, gyroaverage, complete velocity measure,
  and pairwise `psijnorm`; it matches 44,928 traced coefficients at `3.07e-11`
  scaled L2. A public builder assembles the independently constructed driver
  and response into the JIT/differentiable low-rank precompute. The remaining
  second trace state (`phiinit=0.017`, `width0=0.7`) changes the initial field
  norm and aggregate collision RHS (`L2=2.46052e-3`), retains byte-identical
  state-independent coefficient traces, and replays the native solved-`psi`
  action in JAX at `1.39e-13` relative L2. The implicit response-system algebra
  is now implemented with a differentiable
  Woodbury solve and matches an independently formed dense backward-Euler
  system at roundoff. The remaining collision blocker is construction and
  native parity of stella's exact differential test-particle matrix; once that
  matrix is supplied, the complete implicit `psi` and phase-space update are
  locally determined. The independently Gyaradax-matched differential stencil
  can now be materialized as `I-dt*C_tp` and passed directly to that solve,
  providing a complete standalone implicit operator. A scratch-only native
  export now reconstructs all 234 unfactorized stella matrices. They are real
  `12x12` operators with bandwidth 3. Across 208 nonzero-`k_perp` modes,
  subtracting the zero-`k_perp` matrix changes only the diagonal and is linear
  in `k_perp^2` to `6.44e-14`; this excludes band-storage indexing and
  off-diagonal gyro diffusion as the discrepancy. Replaying the complete native
  update as `solve(I-dt*C_tp, g_in + dt*C_fp)` matches stella's 2,808-value
  final state at `1.64e-16` relative L2 (`9.20e-19` maximum absolute error).
  The stricter remaining gate is independent local construction and coefficient
  parity of stella's zero-`k_perp` differential matrix and gyro-diffusion
  diagonal, followed by production-residual selection. The four native ordered
  species-channel matrices reconstruct the all-channel matrix at `3.77e-17`
  relative L2. Their departure-from-identity Frobenius norms are `1.864`
  (ion-ion), `0.1213` (ion-electron), `79.86` (electron-electron), and `75.18`
  (electron-ion), so the next coefficient comparison should start with the two
  electron target blocks. The analytic coefficient layer is now local and
  differentiable: a 624-row native trace of speed, Maxwellian, `nupa`, `nuD`,
  and `nux` agrees with the JAX constructor at relative L2 errors `6.36e-17`,
  `1.99e-16`, `1.08e-16`, `7.15e-17`, and `8.27e-17`. The remaining matrix
  work is therefore finite-difference interior and boundary assembly plus the
  already isolated diagonal gyro term, not collision-frequency normalization.
  The local gyro-diagonal constructor now matches all 208 nonzero-`k_perp`
  native matrix differences with `6.65e-14` maximum absolute error. The only
  remaining differential-matrix gap is the zero-`k_perp` finite-difference
  interior and boundary coefficient generation. A public JIT-compatible,
  differentiable block assembler now sums ordered background channels, packs
  native `aa`/`bb`/`cc` velocity blocks, and adds the identity. A 1,248-row
  scratch trace reconstructs all 26 zero-`k_perp` native matrices exactly
  (`0.0` maximum and relative-L2 error). Block layout and dense packing are
  therefore closed; independent local construction of the interior and
  boundary `aa`/`bb`/`cc` coefficients is the sole matrix blocker. Paired
  native runs with `mu_operator` enabled/disabled split this target exactly.
  The mu path has Frobenius norm `32.2503` (`88.20%` of the full-block norm)
  and diagonal-block norm `32.2473`; the vpar path has norm `14.4030`. With
  `nmu=2`, all mu rows exercise boundary formulas. Implement and validate the
  mu boundary closure first, prioritizing electron-ion (`20.7238`) and
  electron-electron (`24.7037`) pair norms, then port the general interior-mu
  and vpar-boundary branches. A four-run `mu_operator` by `nuxfac` factorial
  split sharpens this further: pure mu diffusion is `32.2473` (`88.19%` of
  full) and exclusively diagonal-block work; its mixed branch is `0.444517`.
  Pure vpar diffusion is only `0.0846490`, whereas the vpar-path mixed branch
  is `14.4028`. The first slice is now closed: the public, differentiable
  two-node constructor evaluates half-mu ordered collision frequencies and
  both default ghost-cell closures, matching all 1,248 native rows at
  `1.10e-15` relative L2 and `7.99e-15` maximum absolute error. Next implement
  the vpar-path mixed branch, followed by general interior-mu grids.
  The public two-mu mixed-vpar constructor now closes that next branch as well,
  including both vpar ghost boundaries and all interior blocks. Native parity
  is `9.11e-16` relative L2 and `2.66e-15` maximum absolute error. The compact
  remainder is the pure-vpar (`0.0846490`) and mu-path mixed (`0.444517`)
  components; general `nmu>2` interior formulas remain a separate requirement.
  The general-mu pure-vpar half-grid builder now matches its native component
  at `1.35e-15` relative L2 and `1.47e-17` maximum absolute error. The only
  remaining compact component was the mu-path mixed branch (`0.444517`). Its
  local two-node builder now matches at `2.80e-15` relative L2 and `4.32e-16`
  maximum absolute error. Summing all four local components reproduces the
  complete native trace at `1.08e-15` relative L2 and `7.99e-15` maximum
  absolute error. The compact matrix is closed; general `nmu>2` interior-mu
  formulas and production-residual selection remain before closing this item.
  The dominant general-grid pure-mu branch is now closed on a fresh pinned
  `nmu=4` trace: its nonuniform interior and boundary rows agree at `1.42e-15`
  relative L2 and `1.53e-16` maximum absolute error. The general mixed-vpar
  and mixed-mu rows now match the same trace at `1.30e-15` and `6.48e-16`
  relative L2. Their complete local sum matches all 4,992 native rows at
  `9.62e-16` relative L2 and `1.80e-16` maximum absolute error. General-grid
  differential coefficients are closed. The production matrix builder now
  combines those blocks with the identity and mode-dependent gyro diagonal;
  its pinned zero-`kperp` layout matches at `6.39e-17` relative L2 and
  `3.33e-16` maximum absolute error. A complete precompute joins that matrix
  to the local Laguerre--Legendre field-particle coefficients, and a public
  backward-Euler step is selectable from the fixed-step split integrator.
  Together with the earlier independent electromagnetic parity gates, this
  closes the combined item.
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
- Historical open item: demonstrate statistically stationary heat flux with resolution/domain
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
  and stationary finest-pair convergence. Historical `jax_fluxtube_gk_native`
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
  The combined campaign now requires at least three unique initialization
  roots, producer-accepted stationarity for every lineage, and no more than
  15% maximum deviation from the ensemble mean. This closes the possibility
  of promoting one cherry-picked chaotic realization. The next numerical gate
  is obtaining three passing coarse lineage reports before spending resources
  on resolved/domain ladders. An attempted second-seed run exposed that direct
  invocation without `JAX_ENABLE_X64=1` silently selected float32. The producer
  now fails before grid construction unless x64 is active, records the choice
  in its report, and binds checkpoint state dtype into the restart contract.
  The first independent x64 lineage (seed 18) has now been extended through
  time 300 with the same `0->60->140->220->300` schedule. It is not stationary:
  the time-220-to-300 segment's candidate window has mean `-14.3748`, block
  relative error `25.61%`, drift `0.5378`, and fitted nonzonal growth
  `-5.89e-3`; its full-segment nonzonal RMS ratio is `0.750`. The earlier
  seed-17 candidate is therefore not robust across initialization. Seed 18
  has now been extended through time 400. Its new half-window remains
  nonstationary with mean `-21.367`, block relative error `9.76%`, drift
  `-0.888`, and nonzonal growth `1.21e-2`. Merging time 220-to-400 gives mean
  `-21.031`, block relative error `6.57%`, drift `-0.300`, and nonzonal growth
  `3.26e-3`; the drift still exceeds the unchanged `0.2` gate. Extending seed
  18 through time 500 produces a stationary time-220-to-500 merge with mean
  `-24.5555`, `5.42%` block error, drift `0.1836`, and nonzonal growth
  `-3.40e-4`. Fresh x64 seed-19 and seed-20 runs to time 500 are also
  stationary, with means `-24.3764` and `-23.7248`. The three-root ensemble
  passes: mean `-24.2189`, maximum relative deviation `2.04%`, and
  between-lineage standard error `0.2524`. The next required evidence is the
  stationary resolution/domain ladders and pinned GX comparison. The combined
  evaluator now fails closed on fake ladders: resolution rungs must refine only
  phase-space resolution at a fixed Fourier/physics contract, while domain
  rungs must enlarge the physical box without losing retained spectral
  bandwidth at fixed phase-space resolution and physics. This pins the current
  domain ladder to `9x5, ky_min=0.1 -> 17x9, ky_min=0.05` rather than accepting
  duplicated reports or a changed dissipation model as convergence evidence.
  A pinned GX preparation helper now writes a caller-owned matched nonlinear
  input and manifest with source revision, input hash, physical case contract,
  and exact run/summarization commands. The summarizer verifies that manifest,
  and campaign acceptance rejects a GX report whose geometry, profiles, box,
  boundary, electrostatic setting, or hyperdiffusion differs from the finest
  local rung. The CUDA run itself remains outstanding. The adaptive producer
  now computes compact flux and potential diagnostics online and retains only
  initial/final phase-space states. This removes the roughly 20 GiB full-state
  history that a `16x16x8`, `9x5`, 14k-step rung would otherwise allocate,
  without changing the default every-accepted-step statistics.
  The first bounded-memory `16x16x8`, `9x5`, seed-19 rung is now complete
  through time 500. Its merged time-200-to-500 window is stationary with mean
  `-29.6930`, `4.40%` block error, `0.1041` drift, and `-1.15e-4` nonzonal
  growth. It remains fail-closed on resolution: the change from the stationary
  `12x12x6` seed-19 mean is `17.91%`, above the unchanged `15%` gate. The next
  fixed-Fourier rung therefore refined to `20x20x10`. Its time-400-to-500
  segment is stationary with mean `-30.8928`, `3.51%` block error, `0.0428`
  drift, and `-7.69e-4` nonzonal growth. The stationary `16x16x8 -> 20x20x10`
  finest pair changes by `3.88%`, so the resolution sub-gate now passes. Domain
  expansion and pinned GX parity remain open. The first `17x9, ky_min=0.05`
  wide-box rung is complete through time 500. Its merged time-200-to-500
  window is stationary with mean `-7.0770`, `8.46%` block error, `0.1754`
  drift, and `-2.70e-4` nonzonal growth. It differs from the stationary `9x5,
  ky_min=0.1` mean by `244%` when normalized to the wide result, so the domain
  gate fails decisively. A `33x17, ky_min=0.025` rung preserving the same
  `kx/ky` bandwidth is now required; the wide result must not be promoted as
  domain-converged.
  That second expansion has reached its first caller-owned time-100 checkpoint
  in 2,867 steps. It remains transient (mean `-10.2580`, `39.0%` block error,
  drift `-0.0729`, field growth `0.0308`) and costs about 31 CPU minutes per
  100-unit segment. The contract-identical time-100-to-200 continuation is also
  nonstationary after 2,856 steps: mean `-3.3568`, `7.91%` block error, drift
  `0.6739`, and field growth `-6.05e-3`. Merging time 0-to-200 still fails on
  drift (`0.3990`, with mean `-3.4244` and `5.99%` block error), so this is not
  a short-window false negative. The next time-200-to-300 segment is the first
  stationary window for this rung: mean `-5.4858`, `5.61%` block error, drift
  `0.0922`, and field growth `1.41e-3`. It differs from the stationary `17x9`
  mean `-7.0770` by `29.0%` when normalized to the finer result, above the
  unchanged `15%` gate. Domain convergence therefore still fails and requires
  a bandwidth-preserving `65x33, ky_min=0.0125` rung. That rung now has a valid
  complex128 time-20 bootstrap checkpoint after 572 steps. Its short startup
  window is intentionally nonstationary (mean `-3.70e-5`, `10.17%` block
  error, drift `-2.10`, field growth `0.103`) and is restart/feasibility
  evidence only. A contract-identical time-20-to-40 continuation adds 572
  steps and a valid finite checkpoint. It remains strongly growing (mean
  `-0.00776`, `12.76%` block error, drift `-2.615`, field growth `0.1438`), and
  the merged time-0-to-40 trace is also nonstationary. The time-40-to-60
  continuation adds another 572 steps and remains strongly growing (mean
  `-2.7114`, `12.80%` block error, drift `-2.624`, field growth `0.1471`, RMS
  ratio `4.29`). The time-60-to-80 continuation reaches the first turnover:
  its mean `-8.6117`, `4.00%` block error, and `0.1601` drift pass their limits,
  but the RMS ratio `0.603` and field growth `-0.0389` correctly reject a
  decaying post-burst window. The time-80-to-100 segment flattens substantially:
  mean `-5.7446`, `1.30%` block error, RMS ratio `0.867`, and field growth
  `-0.0134` pass, but drift `0.2251` narrowly exceeds the unchanged `0.20`
  limit. The time-100-to-120 segment then satisfies every numerical statistic
  (mean `-4.1290`, `0.409%` block error, drift `0.0739`, RMS ratio `0.953`,
  field growth `-0.00472`) but is correctly rejected because its candidate
  window has only 37 samples and duration `9.914`, below the unchanged minima
  of 100 samples and duration 10. A longer uninterrupted late segment at the
  same diagnostic cadence is required; the sampling gate will not be lowered.
  The time-120-to-140 continuation remains bounded in amplitude but is
  intermittent rather than stationary: mean `-3.3663`, `1.65%` block error,
  RMS ratio `0.899`, and field growth `-0.0119` pass, while drift rises to
  `0.3385`. Merging time 100-to-140 also fails drift (`0.3349`) and contains
  only 73 candidate samples. The time-140-to-200 continuation completes in
  1,714 steps and is bounded: its final half has mean `-4.4622`, `3.28%` block
  error, drift `0.1567`, candidate RMS ratio `0.937`, and field growth
  `-2.35e-3`. It has 108 samples over 29.74 time units but only five complete
  physical-time blocks, so the unchanged six-block minimum correctly rejects
  the segment. Merging time 100-to-200 produces the first stationary result
  for this rung: mean `-4.4118`, 180 samples, 49.91 time units, nine blocks,
  `4.84%` block error, drift `-0.0962`, candidate RMS ratio `1.218`, and field
  growth `2.24e-3`. Its `24.34%` change from the stationary `33x17` mean
  `-5.4858`, normalized to the finer result, still exceeds the unchanged 15%
  domain gate. A `129x65, ky_min=0.00625` rung is therefore required.
  CUDA/GX execution is explicitly deferred for now and is not counted as a
  passing independent parity result.
- Historical open item: extend to full equilibrium-shape optimization only after the standalone
  geometry, W7-X parity, convergence, timing, and gradient gates pass.

### Historical 2026-08-10 future-work handoff

The remaining Priority 5 electrostatic acceptance work is scientific
validation, not a missing local operator or API. A 2026-08-10 fail-closed audit
now independently verifies report statistics and stationarity controls,
candidate-window field behavior, restart/segment lineage, shared ladder cases,
local/reference producer identities, finite CLI controls, and pinned GX source,
artifact, numerical-resolution, and physics provenance. Keep both checkboxes
above open until the following evidence exists:

1. **Finish the CPU-local domain ladder.** The caller-owned
   `/private/tmp/p5-domain-finest-seed19-t100-t200-merged.json` is a passing
   stationary `65x33, ky_min=0.0125` report, and
   `/private/tmp/p5-domain-finest-seed19-t200.npz` is its latest valid
   checkpoint. Domain convergence nevertheless fails at `24.34%`, so start the
   next bandwidth-preserving `129x65, ky_min=0.00625` rung from the same seed
   and physics. First establish a finite complex128 time-20 bootstrap in
   caller-owned storage:

   ```console
   JAX_ENABLE_X64=1 uv run python examples/run_nonlinear_heat_flux.py \
     --output /tmp/p5-domain-next-seed19-t0-t20.json \
     --checkpoint-output /tmp/p5-domain-next-seed19-t20.npz \
     --final-time 20 --n-z 12 --n-vpar 12 --n-mu 6 \
     --n-kx 129 --n-ky 65 --ky-min 0.00625 \
     --flux-moment gx_total_energy --seed 19 --diagnostic-stride 8
   ```

   The `129x65` time-0-to-20 bootstrap completed on 2026-08-10 in 572
   adaptive steps. Its caller-owned checkpoint
   `/private/tmp/p5-domain-next-seed19-t20.npz` contains a finite complex128
   `(12,6,12,129,65)` state with exact seed-19 lineage. The startup window is
   intentionally nonstationary: mean `-1.6364e-4`, `10.50%` block error,
   drift `-2.166`, candidate RMS ratio `2.914`, field growth `0.1094`, 37
   samples, and one physical-time block. Continue the identical trajectory to
   time 40 rather than interpreting the transient:

   ```console
   JAX_ENABLE_X64=1 uv run python examples/run_nonlinear_heat_flux.py \
     --output /tmp/p5-domain-next-seed19-t20-t40.json \
     --restart-from /private/tmp/p5-domain-next-seed19-t20.npz \
     --checkpoint-output /tmp/p5-domain-next-seed19-t40.npz \
     --final-time 40 --n-z 12 --n-vpar 12 --n-mu 6 \
     --n-kx 129 --n-ky 65 --ky-min 0.00625 \
     --flux-moment gx_total_energy --seed 19 --diagnostic-stride 8
   ```

   The contract-identical time-20-to-40 continuation also completed in 572
   steps and produced a finite complex128 time-40 checkpoint with lineage
   `[20,40]`. It remains strongly transient: mean `-0.03929`, `12.97%` block
   error, drift `-2.653`, candidate RMS ratio `4.265`, and field growth
   `0.1464`. The merged time-0-to-40 trace likewise fails (mean `-0.02061`,
   drift `-3.852`, growth `0.1426`). Continue the same trajectory to time 60:

   ```console
   JAX_ENABLE_X64=1 uv run python examples/run_nonlinear_heat_flux.py \
     --output /tmp/p5-domain-next-seed19-t40-t60.json \
     --restart-from /private/tmp/p5-domain-next-seed19-t40.npz \
     --checkpoint-output /tmp/p5-domain-next-seed19-t60.npz \
     --final-time 60 --n-z 12 --n-vpar 12 --n-mu 6 \
     --n-kx 129 --n-ky 65 --ky-min 0.00625 \
     --flux-moment gx_total_energy --seed 19 --diagnostic-stride 8
   ```

   The time-40-to-60 continuation completed in 572 steps and wrote a finite
   complex128 checkpoint with exact `[20,40,60]` seed-19 lineage. It remains
   strongly transient: mean `-13.1572`, `12.13%` block error, drift `-2.498`,
   candidate RMS ratio `4.063`, and field growth `0.1420`. The contiguous
   time-0-to-60 merge is also nonstationary: mean `-4.63125`, 109 candidate
   samples over 29.91 time units, five blocks, `78.09%` block error, drift
   `-4.442`, candidate RMS ratio `77.75`, and field growth `0.1471`. Continue
   the exact trajectory through the first nonlinear turnover:

   ```console
   JAX_ENABLE_X64=1 uv run python examples/run_nonlinear_heat_flux.py \
     --output /tmp/p5-domain-next-seed19-t60-t80.json \
     --restart-from /private/tmp/p5-domain-next-seed19-t60.npz \
     --checkpoint-output /tmp/p5-domain-next-seed19-t80.npz \
     --final-time 80 --n-z 12 --n-vpar 12 --n-mu 6 \
     --n-kx 129 --n-ky 65 --ky-min 0.00625 \
     --flux-moment gx_total_energy --seed 19 --diagnostic-stride 8
   ```

   The time-60-to-80 continuation completed in 573 steps and produced a finite
   complex128 checkpoint with exact `[20,40,60,80]` lineage. Its late
   time-70-to-80 candidate passes every scalar gate: mean `-8.96128`, `0.535%`
   block error, drift `-0.0447`, candidate RMS ratio `0.970`, and field growth
   `-0.00249`. The producer correctly remains fail-closed because the candidate
   has only 37 samples, one block, and duration `9.9765`, short of the unchanged
   100-sample, six-block, and duration-10 minima. The broader time-40-to-80
   merge includes the turnover and also fails. Continue the now-bounded state:

   ```console
   JAX_ENABLE_X64=1 uv run python examples/run_nonlinear_heat_flux.py \
     --output /tmp/p5-domain-next-seed19-t80-t100.json \
     --restart-from /private/tmp/p5-domain-next-seed19-t80.npz \
     --checkpoint-output /tmp/p5-domain-next-seed19-t100.npz \
     --final-time 100 --n-z 12 --n-vpar 12 --n-mu 6 \
     --n-kx 129 --n-ky 65 --ky-min 0.00625 \
     --flux-moment gx_total_energy --seed 19 --diagnostic-stride 8
   ```

   The time-80-to-100 continuation completed in 572 steps with a finite
   complex128 checkpoint and exact `[20,40,60,80,100]` lineage. It remains
   bounded, but its late candidate is not stationary: mean `-6.51294`, `1.67%`
   block error, drift `0.3376`, candidate RMS ratio `0.873`, and field growth
   `-0.0148`. The time-60-to-100 merge likewise fails on drift (`0.573`) and
   still has only 73 candidate samples and four blocks. Continue the exact
   intermittent saturated trajectory rather than using a premature mean:

   ```console
   JAX_ENABLE_X64=1 uv run python examples/run_nonlinear_heat_flux.py \
     --output /tmp/p5-domain-next-seed19-t100-t120.json \
     --restart-from /private/tmp/p5-domain-next-seed19-t100.npz \
     --checkpoint-output /tmp/p5-domain-next-seed19-t120.npz \
     --final-time 120 --n-z 12 --n-vpar 12 --n-mu 6 \
     --n-kx 129 --n-ky 65 --ky-min 0.00625 \
     --flux-moment gx_total_energy --seed 19 --diagnostic-stride 8
   ```

   Continue through bounded caller-owned checkpoints until a merged late window
   satisfies producer stationarity with at least 100 samples, duration 10,
   relative block error at most `0.10`, absolute drift at most `0.20`, RMS
   ratio at least `0.8`, and absolute field growth at most `0.02`. Compare its
   mean with `-4.4118`; the relative finest-pair change must be at most `15%`.
   A failure requires another bandwidth-preserving domain rung rather than a
   tolerance change. Do not commit checkpoints or reports.
2. **Run independent nonlinear parity when CUDA/native capacity is available.**
   Use the revision-pinned GX preparation and schema-v1 summarization workflow;
   do not treat the current deferral as a pass or commit solver output.
3. **Attempt unrestricted equilibrium-shape optimization only after** the
   nonlinear domain/parity gates and the existing geometry, W7-X parity,
   convergence, timing, and gradient gates all pass. Preserve the fixed
   topology/remeshing contract and add a checkpointed end-to-end design run.

Acceptance status: **collision and linear-electromagnetic implementation
complete; nonlinear electrostatic and unrestricted-design scientific gates
remain open**. The suite covers the production collision operator,
nonlinear discretization, adaptive control, and diagnostics. Priority 5 cannot
be marked complete until the stationary nonlinear-flux and unrestricted
shape-optimization claims named above pass.

Known capabilities outside this acceptance claim remain explicitly
unsupported rather than silently approximated: the nonlinear residual rejects
electromagnetic field models, and the implicit parallel-response shortcut is
single-species only. These are later physics extensions, not blockers for the
current electrostatic nonlinear campaign.

</details>

## Priority 6: Maintainability After the Standalone Boundary Is Green

- [ ] Split the 16k-line `benchmarks.py` into focused fixture I/O, Cyclone/GKW,
  geometry parity, and W7-X validation modules. The public split is in place as
  lazy `validation.fixture_io`, `validation.cyclone_gkw`,
  `validation.geometry_parity`, and `validation.w7x` namespaces, and the shared
  scalar target contract has moved physically to `targets.py`. The densely
  coupled legacy implementation still needs to be split behind those facades.
  Future-work completion means: move fixture records/readers first, geometry
  parity second, then Cyclone/GKW workflows and shared numerics; reduce
  `benchmarks.py` to a compatibility facade under 500 lines; introduce no
  import cycles; preserve report/fixture schemas; and pass the standalone suite.
- [x] Keep benchmark-only symbols out of the default top-level import path;
  expose a compact solver API and a separate validation namespace. A fresh
  `import jax_fluxtube_gk` no longer loads `jax_fluxtube_gk.benchmarks`, benchmark
  symbols are absent from `__all__`, and legacy attribute imports resolve lazily
  while callers migrate to `jax_fluxtube_gk.validation`.
- [x] Replace historical `Phase N` docstrings with subsystem descriptions and
  document which APIs are stable at version `0.1.x`. The stable solver,
  provider/schema, opt-in validation, compatibility, and experimental surfaces
  are defined in `docs/api_stability.md` and linked from the README.
- [x] Decide which large GKW traces are essential compact regression contracts,
  regenerate smaller selected slices where possible, and move archival raw
  traces out of the source distribution. The retained selected-ky potential,
  velocity-slice, multitime, state/RHS/matrix, time, and input contracts total
  `178,763` bytes and are directly consumed by regression tests. No raw run is
  tracked. `docs/fixture_policy.md`, ignore/manifest rules, and a `200 kB`
  aggregate/`64 KiB` per-file regression budget enforce the decision.

Priority 6 status: **three maintainability items complete; the physical split
of the legacy benchmark implementation is explicit future work behind stable,
lazy public facades**. Repository-wide verification after these changes passes
with **572 tests passed and 25 external tests deselected**.

## Project Rules

- `jax-fluxtube-gk` must never require a particular sibling-directory layout.
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
