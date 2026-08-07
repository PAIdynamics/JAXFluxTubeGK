# STATUS

Last updated: 2026-08-07

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
`wout` file. The externally validated W7-X scientific gate passes for the
converged unstable `ky=0.3` branch. Priority 3 is complete; unconverged low-`ky`
stella windows remain diagnostics rather than grounds for relaxing the gate.
Priority 4 is also complete at the reduced fixed-topology boundary: stable
design objectives, gradient audits, topology guards, robust multi-sample
aggregation, reproducible checkpoints, and a real VMEC++ W7-X outer loop are
implemented. End-to-end VMEC++ autodiff and full-boundary optimization are not
claimed.

### 2026-08-07: Nonlinear Priority 5 Acceptance Gate

- The shared nonlinear loader now requires each local or GX producer's explicit
  stationarity decision. Resolution and parity checks cannot reinterpret a
  report rejected for duration, amplitude retention, or field growth as
  stationary merely because its flux drift and uncertainty are small.
- Added one fail-closed campaign evaluator for separate resolution and domain
  ladders plus independent-reference parity. Every rung and the reference must
  be producer-accepted, and local/GX normalization requires an explicit
  positive conversion factor.
- Updated the confidence ledger with the accepted coarse `12x12x6`, `9x5`
  time-220-to-300 candidate (`-25.7841` native mean flux, `0.462%` relative
  error, `0.0572` drift, `-1.93e-3` nonzonal growth). This is one local rung,
  not resolution/domain convergence or GX parity; those evidence gates remain
  open.

### 2026-08-07: Production Electromagnetic Resolution Gate

- Added a named on-demand production ladder with `16x16x8`, `24x24x12`, and
  `32x32x16` rungs, preserving the reduced ladder as the routine default.
- The full beta-0.01, final-time-10 ladder passes against pinned Gyaradax at
  revision `8d9dc2d205e8993ae9e43e6e1e82ec1ea2875234`. The finest local
  growth/frequency changes are `2.196%`/`1.206%`; independent-reference changes
  are `2.203%`/`1.206%`, all below the 5% convergence threshold. Finest-rung
  growth, frequency, and complex mode errors are `1.72e-5`, `1.08e-9`, and
  `7.58e-10`, with late growth drift `4.90e-7`.
- Generated references and the schema-v1 summary were written only to
  `/tmp/optimal-fusion-em-production-20260807`; no equilibrium, state, or
  reference artifact was added to the repository. The production
  electromagnetic resolution sub-gap is closed; inter-species Landau
  field-particle parity remains open.

### 2026-08-06: Priority 5 Implementation Campaign Verification

- Corrected the kinetic-electron timestep estimate, integrated a conservative
  BGK collision foundation, and added the dealiased nonlinear ExB residual,
  combined adaptive CFL driver, and saturated-flux window statistics.
- The complete standalone x64 suite passes with **485 passed and 25 external
  tests deselected in 474.40 s**. Repository-wide Ruff and whitespace checks
  also pass. The recurring missing `bitx` hook message is unrelated and does
  not affect these results.
- Priority 5 is not declared scientifically complete: production
  inter-species Landau/Fokker--Planck parity,
  stationary nonlinear heat-flux parity, and unrestricted full-shape
  optimization remain evidence-gated work.

### 2026-08-06: Priority 5 Confidence Ledger

- Added a schema-versioned machine-readable ledger for the open full GKW
  selected-mode history, long-time velocity-slice, and low-`ky` Cyclone/GX
  branch-shape gaps. The ledger now also blocks unsupported production claims
  for collisional/electromagnetic physics, nonlinear turbulence, and
  unrestricted equilibrium-shape optimization. The TEM entry now records a
  passed gate.
- Broad GKW/GX parity claims now fail an explicit readiness guard with the
  blocking gap identifier. The converged W7-X linear claim records that its
  independent stella gate narrowly supersedes the two GKW gaps without closing
  or hiding them.
- The ledger retains the observed GKW velocity-slice growth from `3.99e-3` at
  step 20 to `3.67e-2` at step 800 against the unchanged `2e-2` tolerance.
- No external state/history fixture was added; reports are generated on demand
  and may be written to caller-selected scratch storage.

### 2026-08-06: Priority 5 Velocity Backend Decision

- Selected collocation as the production CPU velocity representation; the
  reduced Hermite-Laguerre moment path is not a prerequisite for production
  use of the current linear kinetic solver.
- Added a claim-scoped policy API. Chebyshev supports general reduced and
  differentiable linear CPU runs, finite differences support the GKW
  term/operator convention claim, and midpoint/Gauss-Laguerre supports the
  validated W7-X linear recipe.
- Arbitrary native grids remain plumbing-only until a named gate passes.
  Hermite-Laguerre remains an experimental GX discriminator because it is not
  integrated with the full stellarator residual/field/design path and has no
  equivalent convergence, external-parity, or production timing gate.
- Focused policy, Hermite-Laguerre, and spectral-grid verification passes with
  34 tests.

### 2026-08-06: Kinetic-Electron TEM Preflight

- Added a TEM-favorable electrostatic preflight with kinetic deuterium and
  heavy electrons (`m_e/m_i=0.01`), `R/L_Ti=0`, `R/L_Te=6.9`, and
  `R/L_n=2.2` on s-alpha geometry.
- The preflight passes charge neutrality, fully kinetic quasineutrality,
  coupled JIT residual finiteness, positive CFL, and the expected 10:1
  electron/ion streaming scale. The current reduced case reports zero field
  residual and an estimated explicit CFL timestep of `3.54e-3` after recurrence
  and algebraic-field-response stiffness are included.
- Added `kinetic_electron_tem_external_parity` to the confidence ledger. It was
  initially open after algebraic preflight and is now closed by the later
  pinned growth, frequency, and complex mode-structure gate.
- Added a normalized multi-window TEM discriminator. Short runs are visibly
  transient; longer reduced runs recover an electron-direction branch but do
  not agree with the reference growth and change with resolution.
- Fixed the explicit CFL estimator to include matrix-backend parallel and
  velocity recurrence operators plus a kinetic-quasineutrality field-response
  row-sum bound. The previous estimate omitted these active fourth-order and
  algebraic-coupling terms and became increasingly unsafe as resolution
  increased.
- Repository-wide verification after these Priority 5 changes passes: Ruff is
  clean and the x64 standalone suite reports 430 passed with 20 external tests
  deselected.

### 2026-08-06: Pinned TEM Target Reproduced

- Added an explicit Gyaradax TEM producer requiring a caller-supplied checkout,
  optional exact revision, and scratch output path. No generated state or
  reference artifact is committed.
- At pinned revision `8d9dc2d205e8993ae9e43e6e1e82ec1ea2875234`, the exact
  notebook configuration (`32x32x16`, 200 windows of 20 steps, `dt=0.01`)
  reproduces `gamma=0.66370834`, `omega=-1.02976757`, and final time `40`.
- Added a local reference-matched TEM profile with GKW cell-centered parallel
  and velocity grids, zero-incoming boundaries, Gyaradax's separable upwind
  streaming/trapping backend, and velocity-independent `cosine2`
  initialization. The fused `gkw_igh` backend remains the separate GKW
  convention path.
- Corrected the explicit wave-number contract: public GKW `kthrho=0.7` maps to
  internal `krho=0.56548668` through `kthnorm=q/(2*pi*eps)` for s-alpha. The
  producer now records both values.
- The final-time-40 local result is `gamma=0.6637083371`,
  `omega=-1.0297675735`. Relative growth and frequency errors are `2.23e-9`
  and `1.02e-9`; the phase-aligned complex `phi(z)` error is `8.84e-10`, and
  late-window growth drift is `2.35e-14`. These pass the declared
  10%/20%/25% and `1e-3` gates, closing the kinetic-electron TEM validation
  gap with substantially tighter observed parity.

### 2026-08-06: Electromagnetic Field Foundations

- Added a differentiable GKW-normalized mixed-variable `A_parallel` solve with
  the numerical velocity-grid skin term used by the `g`-to-`f` cancellation.
- Discrete tests cover field-residual closure, odd-`v_parallel` current
  selection, the electrostatic beta-zero limit, and differentiation through
  the solve.
- An opt-in test matches source weights, diagonal, and `k_perp^2` against the
  pinned Gyaradax checkout. The source/diagonal tolerance accounts for the
  documented difference between optimal-fusion's differentiable Cephes `J0`
  approximation and Gyaradax's `jax.scipy` implementation.
- Added the coupled kinetic `phi`/`B_parallel` field solve, gyrokinetic
  `B_parallel` response factor, beta-zero limit, and differentiation tests.
  All coupled coefficients independently match pinned Gyaradax.
- Added the complete mixed-variable field contract: solve `A_parallel` from
  evolved `g`, transform reversibly to physical `f`, then solve coupled
  `phi/B_parallel`. The `g`-to-`f` and generalized-potential coefficients match
  pinned Gyaradax, while the exact beta-zero path recovers kinetic
  electrostatics including constant-mode gauge regularization.
- The new parity test exposed and fixed the large-argument `J1` approximation
  incorrectly erasing physical oscillatory sign changes; a SciPy comparison
  now covers both signs beyond the asymptotic threshold.
- This is an algebraic field foundation, not full electromagnetic evolution.
  The subsequent implementation now couples physical `f`, generalized
  `chi=J0*phi+chi_A+chi_B`, and the two explicit `B_parallel` compression terms
  into the linear RHS. The exact beta-zero residual, JIT, and beta-gradient
  tests pass.
- Fields, mixed-to-physical state recovery, the isolated finite-beta increment,
  and the complete one-state RHS match pinned Gyaradax on the same mixed state.
  The former full-RHS mismatch was an operator-routing error: Gyaradax's JAX
  backend uses separable upwind streaming and trapping, whereas `gkw_igh` is
  the fused GKW convention path. One RK4 step and a five-step unnormalized
  trajectory also match, covering repeated self-consistent field solves.
  Added on-demand electromagnetic controls to the revision-pinned producer and
  local linear smoke runner. A reduced `8x8x4`, beta-0.01 final-time-10 run
  passes the growth/frequency/mode/drift gate: relative errors are `2.30e-3`,
  `2.23e-9`, and `6.82e-10`, with late growth drift `7.23e-5`. No generated
  reference is stored in the repository. Production-grid resolution
  convergence remains open.
- Added an on-demand electromagnetic resolution-ladder runner. It regenerates
  independent references in caller-selected scratch storage and requires both
  per-rung Gyaradax parity and finest-pair convergence. The measured
  `12x12x6 -> 16x16x8` pair passes the 5% gate: local relative growth and
  frequency changes are `4.64%` and `3.93%`; per-rung growth parity errors are
  `8.95e-4` and `2.43e-4`, while frequency and mode errors remain near
  `1e-9`. The later production ladder closes the higher-grid evidence gate.
- Added a differentiable electromagnetic CFL bound covering the mixed-state
  amplification and `A_parallel`, `phi`, and `B_parallel` feedback. It tightens
  at finite beta and conservatively dominates the exact row sum of a small
  dense coupled operator.

### 2026-08-06: Conservative Collision Foundation

- Added an optional species-local linearized BGK operator to the collocation
  residual. Its discrete Maxwellian projection preserves density, parallel
  momentum, and energy independently at each parallel point and Fourier mode.
- Collision frequencies remain differentiable solver inputs, and their
  stiffness is included in the explicit timestep estimate. Multispecies and
  single-species conservation, null-space, residual-integration, CFL, JIT, and
  gradient tests pass.
- This closes the model-collision implementation slice only. A validated
  Landau/Fokker--Planck operator with inter-species exchange remains an open
  Priority 5 physics gate; electromagnetic field evolution now passes its
  production ladder.
- Added a standalone nine-point GKW-style test-particle Fokker--Planck
  foundation for the finite-difference velocity grid. It includes pitch-angle
  scattering, energy diffusion, friction, target/background mass and thermal
  speed scaling, boundary flux closure, a differentiable frequency, and a
  conservative row-sum stiffness bound. The single-species stencil and action
  match pinned Gyaradax to `2e-12`.
- The differential operator is integrated into the linear residual through
  `collision_model="fokker_planck"`; BGK remains the backward-compatible
  default. Its exact stencil row-sum bound contributes to the explicit CFL.
  The reciprocal field-particle term needed for exact inter-species
  momentum/energy exchange remains open, so the production Landau claim stays
  blocked.
- Added an opt-in conservative field-particle completion through
  `collision_conserve_exchange=True`. A discrete global projection preserves
  each species density and combined physical parallel momentum and energy to
  roundoff, while allowing nonzero exchange between species. The correction's
  induced norm is included conservatively in the CFL bound. Independent
  Landau field-particle parity is still required before calling this a
  production Landau operator.
- Added the distinct `collision_conservation_model="xu_species_local"` option.
  Its species-local momentum/energy factors, quadrature weights, and corrected
  action match the pinned Gyaradax/GKW Xu implementation to `2e-12`; local
  momentum and energy defects vanish to roundoff and the correction enters the
  CFL bound. This reference-parity model does not supply reciprocal
  inter-species field-particle exchange, so that production gate remains open.
- Added the experimental `collision_conservation_model="pairwise_exchange"`
  path. The precompute retains all ordered target/background stencils; each
  unordered self or cross-species pair is corrected separately, and the
  directed contributions are exposed for diagnostics. Focused JIT tests verify
  per-pair density and combined physical momentum/energy conservation, true
  off-diagonal distribution coupling, summed-action consistency, and a finite
  conservative CFL contribution. GKW cannot validate this slice because its
  source explicitly states that collision conservation has not been made
  global. Independent stella Laguerre--Legendre coefficient/action parity is
  still required, so this remains an experimental architecture rather than a
  production Landau operator.

### 2026-08-06: Nonlinear ExB Foundation

- Added the gyroaveraged nonlinear term `-{J0 phi,f}` using Hermitian
  half-spectrum reconstruction, 3/2-padded two-dimensional FFTs, real-space
  bracket evaluation, and truncation to retained modes.
- Nonlinear grids now fail early unless they contain centered, uniform `kx`
  modes and a uniform nonnegative `ky` half spectrum beginning at zero. The
  operator is integrated with the electrostatic residual and has a padded-grid
  ExB advective CFL estimator.
- Manufactured Fourier coefficients, antisymmetry, constant-field invariance,
  JIT, amplitude/CFL scaling, grid rejection, and residual integration pass.
- Added candidate saturated-flux statistics reporting the mean, sample
  deviation, standard error, and relative window drift. An adaptive nonlinear
  RK4 driver now selects the minimum of the coupled linear and instantaneous
  nonlinear CFL bounds and records every accepted step. Adaptive accept
  decisions are explicitly outside the differentiable fixed-step path.
- Added `gyrokinetic_heat_response`, the per-species collocation quadrature of
  `J0 T_s (E_s-3/2) f_s`. It is JIT-compatible and agrees with a direct weighted
  velocity integral, closing the missing evolved-state-to-heat-flux diagnostic
  connection.
- Added an artifact-free, revision-pinned GX nonlinear heat-flux summarizer for
  the independently maintained Cyclone benchmark. It reads GX's
  `Diagnostics/HeatFlux_st`, validates finite strictly increasing traces, and
  writes a schema-v1 caller-selected JSON report with mean, sample deviation,
  standard error, relative drift, window bounds, species index, and source
  revision. Synthetic stationary and invalid-trace contracts pass. No GX
  NetCDF output is tracked by this repository.
- Added a deterministic local nonlinear heat-flux producer joining the driven
  s-alpha ITG case, fourth-order Fourier hyperdiffusion, adaptive CFL evolution,
  self-consistent potential history, gyrokinetic heat response, radial-flux
  trace, and explicit drift/uncertainty stationarity decision. A tiny
  final-time-0.2 end-to-end smoke writes finite caller-owned JSON, takes two
  accepted steps, and correctly remains nonstationary. Longer resolution and
  box-size acceptance trajectories are still evidence-gated.
- Added schema-v1 local/GX report loading plus shared stationarity, uncertainty,
  mean-parity, and finest-pair convergence decisions. All ladder rungs must be
  stationary. Native local and GX `Q/Q_GB` normalizations cannot be compared
  implicitly; a declared positive conversion factor is mandatory. Focused
  tests cover passing parity/convergence, drift failure, and normalization
  mismatch. The conversion factor itself remains physics evidence to derive.
- Ran the first reduced driven local discriminator at `8x8x4` phase space and
  `5x3` Fourier modes. Final-time-20 uses 261 accepted steps and reports mean
  native heat flux `-3.28e-8`, `1.42%` relative standard error, and drift
  `0.335`; final-time-100 uses 1,303 steps and reports `-3.21e-8`, `1.56%`, and
  drift `-0.523`. Both fail the 0.2 drift gate. The longer run rules out a
  merely undersampled window; growth, damping, domain, or saturation is the
  active nonlinear discriminator. Producer reports now include initial/final
  state and potential RMS plus maximum absolute heat flux for that audit.
- Adding `ky=0.3` made the old flux-only stationarity decision produce a false
  positive: drift `0.185` and relative error `1.42%` passed even though potential
  RMS dropped from `4.17e-4` to `2.19e-4`. The producer now requires a minimum
  final/initial potential-RMS ratio, default `0.8`; the same 284-step run
  correctly reports nonstationary at ratio `0.525`. This narrows the blocker to
  decay of the nominally driven reduced setup rather than noisy diagnostics.
- The apparent decay was a GX-to-local profile-normalization bug. The producer
  had used GX `fprim=0.8` and `tprim=2.49` directly as `R/L` gradients; it now
  applies `Rmaj/Lref=2.77778`, giving `R/Ln=2.222224` and `R/LT=6.9166722`, and
  records the source and derived controls. Repeating the `12x12x6`, `5x4`,
  hyperdiffusion-0.005 run to final time 20 gives nonzonal potential growth
  `12.59x`; `ky=0.2`/`0.3` grow `13.40x`/`18.59x`. Flux drift `-3.23` rejects
  the still-unsaturated trajectory. The stationarity guard uses nonzonal RMS
  and reports total, nonzonal, and per-`ky` amplitudes. Long-time saturation is
  now the next discriminator before a resolution and box ladder. In the
  `5x4` box neither hyperdiffusion `0.005` nor `0.05` plateaus by time 60.
  Expanding to `9x5` delays the growth but still gives nonzonal RMS `365x` and
  flux drift `-4.48` at time 40. Its random initial condition also produced a
  zonal potential roughly 16 times larger than a typical nonzonal branch due
  to the weak zonal polarization denominator. The producer now defaults the
  zonal seed to zero and offers an explicit opt-in fraction, ensuring zonal
  flow in the acceptance trajectory is generated by nonlinear transfer.
- The zero-zonal `9x5` run still grows at time 60: nonzonal RMS increases
  `5822x`, peak native flux reaches `43.2`, and flux drift is `-3.74`, despite
  strong self-generated zonal flow. The producer now exposes an opt-in
  conserving-BGK collision frequency with adaptive-CFL accounting to test
  whether unresolved velocity-space cascade is the limiter. It remains off by
  default and is not presented as GX hypercollision or Landau parity.
- A `nu=0.1` BGK discriminator barely changes the time-60 result: nonzonal RMS
  is `5664x` and flux drift is `-3.68`. The audit therefore moves the active
  blocker to flux-tube topology. The current producer uses independent periodic
  `kx` chains; production Cyclone acceptance requires shear-consistent `kx`
  spacing plus twist-and-shift parallel connectivity. README and TODO now
  classify the periodic-chain runs as numerical discriminators only.
- Added a public cell-centered GKW parallel-grid builder and made the nonlinear
  producer default to the GKW upwind twist-and-shift boundary stencil. Its
  Fourier grid now uses the shear relation
  `dkx=q*shat*ky_min/(eps*ikxspace)`, and multi-`ky` connectivity scales the
  boundary jump by `ky/ky_min`. The old periodic-chain model remains an
  explicit diagnostic option. Focused Fourier, nonlinear, and benchmark tests
  pass; long-time stationarity on this corrected topology is the next gate.
- The first corrected-topology `12x12x6`, `9x5`, hyperdiffusion-0.05 run reaches
  time 60 in 1,714 adaptive steps. Relative to the periodic-chain run, peak
  native flux drops from `43.2` to `0.314` and final nonzonal RMS from `1.33`
  to `0.102`, demonstrating that boundary topology materially controls the
  nonlinear trajectory. The window still grows with drift `-4.71`, so it does
  not pass stationarity; longer-time and wider-box convergence remain open.
- Added caller-owned nonlinear restart checkpoints containing complex state,
  absolute time, and a schema-versioned topology/physics contract. Changed
  grids, profiles, damping, collision settings, or boundaries are rejected.
  Resumed statistics cover the new segment, allowing the transient to be
  excluded without storing it in the repository. A restart smoke advances
  time `0.2 -> 0.4` exactly. It exposed and fixed a short-window false positive:
  stationarity now additionally requires 100 samples and duration 10 by
  default, so the two-step segment correctly fails.
- Compiled the nonlinear CFL evaluation and complete RK4 accepted step while
  leaving adaptive termination and acceptance host-controlled. Focused time-
  advance/nonlinear tests pass (`31 passed`). A `12x12x6`, `9x5` corrected-
  topology segment reaches time 20 in 572 steps and `49.36 s` wall time while
  writing a contract-checked checkpoint. Statistical gates, not bitwise trace
  identity, remain the nonlinear acceptance contract.
- Contract-checked checkpoints extended the corrected run through time 220.
  It develops repeatable nonlinear bursts rather than continued exponential
  growth, but the final 40-unit window remains nonstationary: mean flux
  `-30.80`, relative error `0.895%`, drift `0.633`, and half-window means
  `-35.46`/`-26.16`. Added a candidate-window nonzonal-potential log-growth
  fit with default magnitude limit `0.02`, closing the possibility that a
  slowly growing/decaying field passes during a coincidentally flat flux span.
- The time-220-to-300 checkpoint segment is the first local stationary
  candidate to pass every declared gate. Over its final 40 time units, mean
  native flux is `-25.7841`, relative standard error `0.462%`, drift `0.0572`,
  and fitted nonzonal-potential growth `-1.93e-3`; half-window means are
  `-27.235` and `-24.336`, and the potential RMS ratio is `0.879`. This is a
  coarse `12x12x6`, `9x5` rung only. Resolution/domain convergence and the
  independent GX normalization/parity gate remain open.
- The following time-300-to-400 window has mean `-26.256`, only `1.8%` from
  the first candidate, but fails its trend gate (`-0.450`; half means
  `-23.145`/`-29.363`). Combining time 220-to-400 gives mean `-27.100`; its
  final 90-unit half has mean `-25.251`, relative error `0.658%`, and drift
  `-0.068`. Reports now include compact nonzonal-potential RMS histories as
  well as flux histories so merged-window acceptance can be independent of
  checkpoint boundaries without retaining phase-space state histories.
- Added a schema-v1 nonlinear segment merger. It validates matching
  normalization and every grid/physics contract field, requires contiguous
  absolute times, drops duplicate checkpoint-boundary samples, and recomputes
  flux uncertainty/drift plus nonzonal amplitude ratio/growth over one merged
  window. Passing and contract-change/gap rejection tests bring the focused
  nonlinear reporting suite to **12 passed**.
- Converged stationary heat flux and external nonlinear parity remain open
  before claiming nonlinear turbulence readiness.

### 2026-08-06: Priority 4 Reduced Design Integration Complete

- Added `DesignObjectiveSpec`/`design_objective` as the stable public contract
  for selected, hard-max, or smooth-max growth; real-frequency targeting;
  phase-invariant complex mode structure; and quasilinear proxy terms.
- Added reverse-mode-versus-central-difference audits through geometry and time
  advance. Branch-gap diagnostics expose near-degenerate `ky` modes, and the
  smooth aggregation path has a finite gradient at exact degeneracy.
- Added a schema-versioned `OptimizationTopologyContract` over velocity and
  parallel nodes/backends, Fourier modes, connectivity, and provider linking.
  Restarts and compiled objectives reject topology changes and require rebuild.
- Ran `examples/vmecpp_w7x_design_loop.py` locally against VMEC++'s installed
  `w7x-standard` input. One iteration completed three fresh equilibrium solves,
  in-memory provider conversion, fixed-topology checks, and an outer central
  finite-difference step; its JSON record stayed under `/private/tmp`.
- Added fixed `(rho, alpha, ky)` robust aggregation with normalized weights and
  weighted-mean, hard-worst-case, and differentiable soft-worst-case results.
- Added schema-v1 optimization checkpoints carrying sample axes and values,
  objective/aggregation policy, topology fingerprints, design parameters,
  provider provenance, code/dependency revisions, command, seed, and history.
- Verified the exact standalone boundary after closure: Ruff passes and the
  x64 default suite reports 408 passed with 20 external tests deselected.
- Priority 4's acceptance gate is reduced design integration. VMEC++ and GVEC
  remain non-differentiable providers, while full equilibrium-shape AD and
  nonlinear heat-flux optimization remain Priority 5 work.

### 2026-08-06: Priority 3 Closure

- Traced the complete stella production step and corrected the solver ordering
  to SSP RK3 explicit stages, a full direction-aware cubic mirror
  characteristic, and a full implicit streaming/quasineutrality response.
- Replayed the exact native input in JAX: RK3, mirror, final distribution, and
  potential errors are `7.19e-6`, `1.34e-4`, `2.17e-5`, and `1.34e-4`.
- Identified the long-time blocker as the initial perturbation. The scan now
  offers an analytic `stella_maxwellian` initializer; it does not store a W7-X
  distribution or equilibrium artifact.
- The converged `t=500`, 256×32×8, `dt=0.1`, `ky=0.3` comparison passes with
  growth/frequency/profile errors `3.20e-4/2.18e-3/1.08e-2` against tolerances
  of 0.02. Across `ky=0.1,0.2,0.3`, growth and profiles pass, but low-`ky`
  frequency is not accepted because stella's own omega window remains
  unconverged even at `t=1000`.
- Added a guarded `stella-production` CPU timing path that uses disposable
  scratch output and the validated advance. A 100-step run took 13.34 s end to
  end, including geometry load, JAX compilation, and diagnostics; estimated
  memory was 5.13 MiB.
- This validated linear branch is the basis for Priority 4. It does not claim
  end-to-end MHD gradients, nonlinear turbulence, or CUDA/GX parity.

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
- Corrected the opt-in stella response to use zero-incoming boundaries for the
  nonzonal one-segment W7-X modes, then matched stella's separate cell
  centering of the streaming coefficient and Maxwellian and its single
  full-step post-explicit response ordering. Focused algebra, boundary, and
  integration tests pass. Controlled `t=200` runs leave the `ky=0.3`
  phase-aligned profile error near `0.169`, so none of these partial source
  corrections closes the gate. The next discriminator is a stage-resolved
  one-step trace of stella's inhomogeneous solve, field response, and final PDF
  sweep; further long-time tuning is deferred until that replay agrees.

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
- stable differentiable design objectives, branch/gradient audits,
  fixed-topology guards, robust multi-sample aggregation, and reproducible
  optimization checkpoints;
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

The scientific W7-X gate now passes on its converged acceptance branch:

- matched stella geometry, initial distribution, 256×32×8 grid, `ky=0.3`,
  `kx=0`, `dt=0.1`, and `t=500` controls close growth, frequency, and profile;
- exact stage replay validates the explicit, mirror, and implicit response
  ordering independently of the long-time eigenmode diagnostic;
- the source-matched production CPU timing path passes after external parity;
- low-`ky` omega results remain non-acceptance diagnostics until the independent
  stella windows converge;
- production MHD design optimization remains a Priority 4 gradient task, not a
  remaining Priority 3 solver-parity blocker.

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
