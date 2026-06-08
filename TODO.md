# TODO: Differentiable Flux-Tube Stellarator Gyrokinetic Solver

Last planned: 2026-06-08

## Project Goal

Build a differentiable local flux-tube stellarator gyrokinetic turbulence code, based on the physical structure and benchmark conventions of Gyaradax and GKW, and informed by GX's modern flux-tube geometry, Fourier pseudo-spectral layout, Laguerre-Hermite velocity-space formulation, closures, diagnostics, and benchmarks. The initial focus remains linear electrostatic gyrokinetics, Boozer/field-aligned stellarator geometry, CPU-usable performance, spectral discretization along the magnetic field and velocity-space coordinates, and gradients of growth-rate and quasilinear transport proxies with respect to equilibrium and flux-tube parameters.

The working order is:

1. Write the physical model and detailed numerical schemes in `main.tex`.
2. Implement the differentiable solver in `src/`.
3. For each newly implemented function, add or update a unit test in `tests/`.
4. Update `STATUS.md` during every implementation round.

## Source Material Read

- `task.tex`: thesis scope, deliverables, six-month roadmap, focus on local linear electrostatic stellarator flux tubes, differentiability, Gyaradax/GKW basis, Boozer surface import, and CPU-oriented runtime.
- `papers/gyaradax-paper/main.tex`: Gyaradax model, GKW term mapping, JAX differentiability, RK4, validation against Rosenbluth-Hinton and Cyclone Base Case, diagnostics, performance notes, and limitations.
- `papers/gkw/GKW.pdf`: primary GKW paper reference; treat this as the authoritative visual/paper source.
- `papers/gkw/GKW_rebuilt.tex`: editable, searchable reconstruction of the GKW paper text; builds to `papers/gkw/GKW_rebuilt.pdf` without embedding `GKW.pdf`.
- `papers/gkw/GKW_manual_0.4-b1.pdf`: GKW manual reference for code usage, options, and implementation details.
- `papers/gx-paper/main.tex`: GX paper source; use for the GPU-native nonlinear flux-tube formulation, Fourier-Laguerre-Hermite pseudo-spectral velocity method, geometry discussion, closure strategy, diagnostics, and benchmark hierarchy.
- `relevant-codes/gkw/README`: GKW package overview, build notes, sample descriptions, and source-file map.
- `relevant-codes/gkw/src/`: original Fortran GKW simulation source code; use as the direct reference for normalization, geometry, modes, RHS terms, timestep logic, diagnostics, and benchmarks.
- `relevant-codes/gkw/samples/`: GKW sample input cases, including `cyclone`, `simple_example`, `simple_itg`, `STD`, and `STD_kinetic`.
- `relevant-codes/gx/README.md`: GX package overview, target use case, GPU-native implementation goals, and high-level numerical method summary.
- `relevant-codes/gx/docs/Numerics.rst`, `Geometry.rst`, `Inputs.rst`, and `Nonlinear.rst`: GX user-facing descriptions of pseudo-spectral numerics, linked/twist-and-shift domains, geometry options, Hermite/Laguerre resolution, closures, hypercollisions, nonlinear dealiasing, and diagnostics.
- `relevant-codes/gx/include/` and `relevant-codes/gx/src/`: GX source-level references for grids, geometry arrays, linked parallel derivatives, moments, Laguerre transforms, closures, fields, nonlinear terms, and diagnostics.
- `relevant-codes/gx/unit_tests/` and `relevant-codes/gx/docs/inputFiles/`: GX test/input conventions to mine for future benchmark fixtures, with the caveat that GX notes some unit tests are old and may require verification before use as strict oracles.
- `relevant-codes/gyaradax/docs/NOTES.md`: detailed GKW-to-Gyaradax mapping, normalization, RHS terms I-VIII, field solver, CFL, geometry, tests, known limitations, validation metrics.
- `relevant-codes/gyaradax/gyaradax/*.py`: reference implementation for parameters, geometry, stencils, integrals, solver, diagnostics, simulation interface, and backend split.
- `relevant-codes/gyaradax/tests/*`: test strategy for geometry parity, phi/flux integrals, solver contracts, gradients, analytical benchmarks, and GKW reference parity.
- `papers/arXiv-2301.09356v2/main.tex`: direct microstability optimization with linear gyrokinetics, Boozer/Clebsch flux-tube coordinates, quasilinear heat-flux proxy, field-line/radius scans.
- `papers/arXiv-2310.18842v2/opt_of_nonlinear_turbulence.tex`: turbulence optimization loop, DESC/GX geometry quantities, multiple surfaces/field lines, noisy heat-flux objective treatment.
- `relevant-codes/DESC/README.rst`: DESC as a differentiable stellarator equilibrium and optimization source for geometry and shape parameters.

Note: local `pdftotext` is not installed. Use `papers/gkw/GKW_rebuilt.tex` for searchable paper text and `papers/gkw/GKW.pdf` as the reference document. Use `relevant-codes/gkw/src/` as the authoritative implementation reference.

## Reference Hierarchy

Use references in this order when behavior is ambiguous:

1. `papers/gkw/GKW.pdf` for the published equations, numerical method descriptions, and benchmark statements.
2. `relevant-codes/gkw/src/` for exact legacy implementation behavior.
3. `papers/gkw/GKW_rebuilt.tex` for searchable extracted GKW paper text.
4. `relevant-codes/gkw/samples/` for runnable GKW input conventions and benchmark/sample cases.
5. `relevant-codes/gyaradax/` for the modern differentiable/JAX mapping of GKW behavior.
6. `papers/gx-paper/` and `relevant-codes/gx/` for GX's Fourier-Laguerre-Hermite algorithm, linked flux-tube geometry conventions, closure ideas, nonlinear/dealiasing strategy, diagnostics, and benchmark/input conventions.
7. `relevant-codes/DESC/` and stellarator optimization papers for equilibrium, Boozer/field-line geometry, and optimization interfaces.

Important GKW source files to consult during implementation:

- `relevant-codes/gkw/src/linart.f90`: main program and run flow.
- `relevant-codes/gkw/src/normalise.F90`: normalization and unit conventions.
- `relevant-codes/gkw/src/geom.F90`: geometry, metric tensors, drift coefficients.
- `relevant-codes/gkw/src/mode.F90`: spectral modes and flux-tube mode connectivity.
- `relevant-codes/gkw/src/grid.F90`: grid-size setup.
- `relevant-codes/gkw/src/velocitygrid.F90`: velocity-space grids and weights.
- `relevant-codes/gkw/src/components.F90`: species parameters and Maxwellian setup.
- `relevant-codes/gkw/src/functions.F90` and `specfun.f90`: special functions, including Bessel/Gamma-related support.
- `relevant-codes/gkw/src/linear_terms.F90`: linear RHS terms, Poisson/quasineutrality pieces, and term-level sign conventions.
- `relevant-codes/gkw/src/non_linear_terms.F90`: nonlinear ExB bracket, FFT/dealiasing, and nonlinear timestep support.
- `relevant-codes/gkw/src/exp_integration.F90`: explicit RK time integration and field update sequence.
- `relevant-codes/gkw/src/matdat.F90`: matrix construction/compression and timestep estimates.
- `relevant-codes/gkw/src/diagnostic.F90`: growth rates, frequencies, fluxes, spectra, and output conventions.
- `relevant-codes/gkw/src/collisionop.F90` and `rotation.F90`: later extensions beyond the linear electrostatic baseline.

Important GX source files to consult during implementation:

- `relevant-codes/gx/include/grids.h` and `parameters.h`: domain sizes, Fourier mode arrays, Hermite/Laguerre resolution, domain lengths, linked-boundary controls, and runtime options.
- `relevant-codes/gx/include/geometry.h`: geometry array contract, including `bmag`, `gradpar`, `gds2`, `gds21`, `gds22`, drift arrays, Jacobian, and operator arrays.
- `relevant-codes/gx/include/grad_parallel.h`: periodic, linked, non-twist-and-shift, and local parallel derivative implementations.
- `relevant-codes/gx/include/moments.h`: Hermite-Laguerre moment layout `G(l,m)[ky,kx,z]` and fluid-moment accessors.
- `relevant-codes/gx/include/laguerre_transform.h`: Laguerre grid/spectral transform interface for nonlinear pseudo-spectral velocity-space evaluation.
- `relevant-codes/gx/include/closures.h` and `smith_par_closure.h`: Beer/Smith closure models and moment-hierarchy closure structure.
- `relevant-codes/gx/include/nonlinear.h`, `fields.h`, and `diagnostic_classes.h`: pseudo-spectral nonlinear terms, field solve structure, and spectra/diagnostic output patterns.
- `relevant-codes/gx/docs/Inputs.rst`: practical input conventions for `nhermite`, `nlaguerre`, `y0`, `jtwist`, `boundary`, `geo_option`, closure models, hypercollisions, and spectra.

## GX Comparison and Reuse Decisions

GX is an algorithmic and benchmark reference, not the exact target architecture.

- Different target: GX is a GPU-native C++/CUDA nonlinear turbulence code optimized for time-to-solution on one or more GPUs. This project is a JAX-first differentiable solver aimed at CPU-usable optimization loops, with linear electrostatic flux-tube physics as the first production milestone.
- Different default velocity representation today: the current implementation uses Chebyshev-Lobatto collocation in `v_parallel` and `mu` because it is direct, differentiable, and easy to test against manufactured functions. GX uses a Fourier-Laguerre-Hermite pseudo-spectral formulation in `kx`, `ky`, `z`, `mu B`, and `v_parallel`, with evolved Hermite/Laguerre moments.
- Different first validation target: GKW/Gyaradax remain the source of truth for term-by-term linear electrostatic parity. GX will be used for geometry conventions, velocity-space methods, diagnostics, nonlinear extensions, and cross-code benchmark cases.
- Already aligned with GX: local flux-tube ordering, field-aligned coordinates, perpendicular Fourier modes, `v_parallel`/`mu` velocity coordinates, precomputed geometry arrays, `k_perp^2` metric contraction, linked/twist-and-shift mode connectivity, and a spectral-first philosophy.
- Planned reuse from GX: Hermite-Laguerre velocity backend, moment diagnostics, closure/hypercollision models, nonlinear pseudo-spectral ExB evaluation and dealiasing, GX-style input/benchmark fixtures, and geometry adapter parity checks for `eik`/GS2/stella-like arrays.
- Not adopted for now: GPU-native kernel design, production nonlinear turbulence performance targets, and closure-dependent reduced gyrofluid operation are deferred until the differentiable linear solver is stable.

## Development Rules

- Keep `main.tex`, `TODO.md`, and `STATUS.md` current. Before coding a subsystem, document its model and numerical scheme in `main.tex`.
- Every new public function in `src/` must have a test in `tests/`. Prefer one small unit test per function plus integration tests for coupled physics.
- Use JAX-compatible pure functions for differentiable paths. Keep integer topology, file I/O, and non-differentiable indexing outside gradient-traced functions.
- Keep geometry transforms and solver kernels modular enough to test separately: grids, geometry, Bessel/Maxwellian factors, phi solve, RHS terms, time/eigen diagnostics.
- Match Gyaradax/GKW physics, normalization, signs, and benchmarks first, then extend. The target discretization follows `task.tex`: Fourier perpendicular directions plus spectral magnetic-field and velocity-space operators. GKW finite differences are retained as a fallback/parity backend, not the default numerical target.
- Use GX as a method source for the Fourier-Laguerre-Hermite extension, linked parallel-gradient checks, nonlinear pseudo-spectral/dealiasing design, and diagnostic/benchmark conventions. Keep the current APIs broad enough that the Chebyshev-collocation backend and a future GX-style moment backend can share the same physics primitives where practical.
- Maintain CPU viability as a first-class constraint. Optimize for vectorized JAX/XLA on CPU before adding optional GPU-specific code.
- Each implementation round must update `STATUS.md` with:
  - date/time,
  - goal,
  - files changed,
  - tests run and results,
  - next action,
  - open risks or blockers.
- Current Phase 2 test/lint commands:
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`

## Phase 0: Repository Bootstrap

- [x] Decide package name and layout: `src/stellarator_gk/`.
- [x] Add project metadata: `pyproject.toml`, Python version, runtime dependencies, test dependencies, lint/format configuration.
- [x] Create `src/` and `tests/` package structure.
- [x] Add a minimal import smoke test.
- [x] Add CI-friendly test command documentation.
- [x] Decide whether to vendor/adapt selected Gyaradax code or reimplement cleanly with references: clean reimplementation with GKW/Gyaradax references.
- [ ] Decide how to cite/use the original GKW Fortran source in code comments and documentation.
- [x] Create a GKW source crosswalk document or section in `main.tex` mapping equations/operators to `relevant-codes/gkw/src/*`.
- [ ] Add a small `examples/` or `configs/` directory for standard test cases.
- [ ] Port or adapt reduced input fixtures from `relevant-codes/gkw/samples/simple_example`, `simple_itg`, and `cyclone`.

Suggested initial layout:

```text
src/stellarator_gk/
  __init__.py
  types.py
  params.py
  grids.py
  geometry/
    __init__.py
    circular.py
    boozer.py
    fieldline.py
  numerics/
    __init__.py
    stencils.py
    spectral.py
    quadrature.py
  physics/
    __init__.py
    maxwellian.py
    bessel.py
    quasineutrality.py
    rhs_terms.py
  solver.py
  diagnostics.py
  objectives.py
```

## Phase 1: Write `main.tex` Model and Numerics

- [x] Add a clear scope statement: local flux-tube, linear electrostatic, collisionless baseline, adiabatic electrons first, kinetic electrons later.
- [x] Add a source-of-truth paragraph explaining that `GKW.pdf`, `GKW_rebuilt.tex`, and `relevant-codes/gkw/src/` define the GKW reference behavior for this project.
- [x] Add a GX comparison note explaining what differs from GX, which GX methods can be reused, and which GX-inspired pieces are already present.
- [x] Define normalization: length, velocity, time, potential, magnetic field, distribution function, species parameters, gradient lengths.
- [x] Cross-check normalization against `relevant-codes/gkw/src/normalise.F90` and the GKW paper.
- [x] Define phase-space coordinates:
  - parallel velocity `v_parallel`,
  - magnetic moment `mu`,
  - field-line coordinate `z` or `s`,
  - perpendicular Fourier modes `k_x`, `k_y`,
  - species index.
- [x] Define field-aligned stellarator coordinates:
  - flux label `psi` or `rho`,
  - field-line label `alpha`,
  - Boozer angles `(theta, phi)`,
  - Clebsch form `B = grad psi x grad alpha`,
  - rotational transform `iota`.
- [x] Write the linear electrostatic gyrokinetic equation in GKW/Gyaradax term convention:
  - Term I: parallel streaming.
  - Term II: magnetic drift advection.
  - Term IV: mirror/trapping term.
  - Term V: equilibrium density/temperature-gradient drive.
  - Term VII: parallel field drive.
  - Term VIII: drift field drive.
  - Term III documented as omitted initially for linear runs, retained for future nonlinear extension.
- [x] Write quasineutrality for adiabatic electrons.
- [x] Add kinetic-electron quasineutrality as planned extension.
- [x] Define finite-Larmor-radius factors `J_0` and `Gamma_0`.
- [x] Define Maxwellian and gradient-drive factors.
- [x] Document required stellarator geometry quantities:
  - `B`,
  - `b dot grad z`,
  - `|grad psi|^2`,
  - `|grad alpha|^2`,
  - `grad psi dot grad alpha`,
  - `(B x grad B) dot grad psi`,
  - `(B x grad B) dot grad alpha`,
  - `(b x kappa) dot grad alpha`,
  - mapping to Gyaradax-style arrays `little_g`, `ffun`, `gfun`, `efun`, `dfun`.
- [x] Document numerical grids and quadrature:
  - spectral velocity-space collocation and weights,
  - spectral field-line collocation and weights,
  - centered Fourier grid in `k_x`,
  - nonnegative half-spectrum in `k_y`.
- [x] Document parallel boundary conditions:
  - periodic zonal modes,
  - magnetic-shear mode connectivity for nonzonal modes,
  - stellarator field-line twist/ballooning boundary extension.
- [x] Cross-check mode connectivity conventions against `relevant-codes/gkw/src/mode.F90`.
- [x] Document derivative schemes:
  - spectral derivative matrices in the magnetic-field coordinate,
  - spectral derivative matrices in velocity-space coordinates,
  - optional spectral filters/dissipation,
  - GKW/Gyaradax fourth-order finite differences as a fallback backend.
- [x] Cross-check fallback stencil, dissipation, and RHS sign conventions against `linear_terms.F90`, `matdat.F90`, and Gyaradax tests.
- [x] Document solver modes:
  - matrix-free RHS residual `R(f; geometry, params)`,
  - RK4 initial-value growth-rate extraction,
  - later matrix-free eigenvalue interface.
- [x] Document diagnostics:
  - `phi`,
  - growth rate `gamma`,
  - real frequency `omega`,
  - mode structure,
  - quasilinear proxy `sum_k gamma(k) / <k_perp^2>`,
  - finite-difference and AD gradient checks.
- [x] Document benchmark hierarchy:
  - algebraic/unit tests,
  - reduced GKW sample cases from `relevant-codes/gkw/samples/`,
  - circular/s-alpha GKW parity,
  - Rosenbluth-Hinton residual,
  - Cyclone Base Case ITG,
  - Boozer/DESC geometry consistency,
  - stellarator ITG scans against GS2/GX/GKW-style references where available.

## Phase 2: Core Types, Parameters, and Grids

- [x] Implement immutable parameter dataclasses/PyTrees:
  - solver controls,
  - species parameters,
  - grid parameters,
  - geometry scalar parameters,
  - differentiability/static-field split.
- [x] Implement spectral velocity grids, quadrature weights, derivative matrices, and modal/filter transforms.
- [x] Implement spectral field-line grids, integration weights, derivative matrices, and modal/filter transforms.
- [x] Implement Fourier `kx`/`ky` grids.
- [x] Implement GKW-style finite-difference derivative backend as an optional fallback/parity path.
- [x] Implement mode-label and mode-connectivity construction outside gradient-traced code.
- [x] Unit tests:
  - shapes/dtypes,
  - grid centering,
  - spectral integration weights,
  - manufactured spectral derivative convergence,
  - finite-difference fallback parity on small fixtures,
  - PyTree flatten/unflatten,
  - mode connectivity for zonal and nonzonal cases.

## Phase 3: Baseline Analytic Geometry

- [x] Implement circular and s-alpha geometry to reproduce Gyaradax behavior before stellarator geometry.
- [x] Compute:
  - `B(s)`,
  - metric components for `k_perp^2`,
  - `b dot grad s` / parallel streaming factor,
  - mirror-force factor,
  - ExB tensor,
  - magnetic drift tensor.
- [x] Keep continuous geometry differentiable with respect to `q`, `shat`, and `eps`.
- [x] Keep mode topology non-differentiable and shape/static.
- [x] Unit tests:
  - field/metric shape and finite values,
  - symmetry and positivity checks,
  - AD gradients versus finite differences,
  - parity against selected Gyaradax analytic geometry outputs.

## Phase 4: Boozer/Stellarator Flux-Tube Geometry

- [x] Define `BoozerSurface`, `PhysicalFluxTubeGeometry`, and `FluxTubeGeometry` data models.
- [x] Support loading geometry from one or more practical sources:
  - [x] DESC-style precomputed flux-tube geometry arrays,
  - [x] direct DESC equilibrium object/example extraction script,
  - [x] canonical DESC DSHAPE `.npz` fixture generated from `relevant-codes/DESC`,
  - [x] direct DESC HDF5/path reader,
  - [ ] SIMSOPT/Boozer surface objects if available,
  - [x] precomputed arrays for tests,
  - [ ] later VMEC/booz_xform data.
- [x] Implement field-line sampling:
  - choose surface `rho` or `psi`,
  - choose field-line label `alpha`,
  - trace `theta`, `phi` along the field line,
  - support multiple poloidal/toroidal turns.
- [x] Compute the geometry set used in GX/GS2 stellarator flux tubes:
  - `B`,
  - `b dot grad z`,
  - `|grad psi|^2`,
  - `|grad alpha|^2`,
  - `grad psi dot grad alpha`,
  - drift quantities involving `B x grad B` and curvature.
- [x] Map these quantities into the solver's internal GKW/Gyaradax coefficient interface.
- [x] Decide and document radial coordinate convention: default to `rho`, with `psi` and minor-radius-normalized `x` carried as metadata options.
- [x] Unit tests:
  - [x] constant/axisymmetric toy geometry recovers circular limit,
  - [x] `k_perp^2 >= 0`,
  - [x] field-line periodicity/twist consistency,
  - [x] AD gradients through geometry arrays for differentiable inputs,
  - [x] extracted DESC DSHAPE fixture loads through the solver geometry contract,
  - [x] solver-produced DESC/GX eik-convention geometry matches the DSHAPE `eik.out` fixture field by field,
  - [ ] finite differences against DESC/SIMSOPT geometry quantities for a small fixture.

## Phase 5: Physics Primitives

- [x] Implement stable `J_0(x)` and `Gamma_0(b) = I_0(b) exp(-b)`.
- [x] Implement Maxwellian and energy variable.
- [x] Implement species-dependent FLR/Bessel factors.
- [x] Implement density and temperature gradient drive coefficients.
- [x] Implement magnetic drift and mirror coefficients.
- [x] Keep physics primitive APIs velocity-backend neutral so the present Chebyshev collocation arrays and a future GX-style Hermite-Laguerre moment backend can both call them.
- [x] Unit tests:
  - small-argument limits,
  - shape broadcasting for one and multiple species,
  - finite values over representative grids,
  - AD gradients versus finite differences for scalar losses.

## Phase 5A: GX-Informed Hermite-Laguerre Velocity Backend

This is a parallel extension path, not a blocker for the first Chebyshev-collocation linear solver.

- [x] Extract the Fourier-Laguerre-Hermite formulas from `papers/gx-paper/main.tex` and the implementation contracts from `relevant-codes/gx/include/moments.h`, `laguerre_transform.h`, and `closures.h`.
- [x] Design a `VelocityBasisSpec`/backend interface that can represent:
  - current Chebyshev collocation in `v_parallel` and `mu`,
  - future Hermite moments in `v_parallel`,
  - future Laguerre moments in `mu B`.
- [x] Implement Hermite basis utilities:
  - normalization conventions,
  - recurrence relations,
  - derivative/moment coupling matrices,
  - quadrature or modal transforms needed for diagnostics.
- [x] Implement Laguerre basis utilities:
  - roots/weights or transform matrices,
  - gyroaverage/Bessel coupling coefficients,
  - moment-to-grid and grid-to-moment transforms.
- [x] Add GX-style moment diagnostics:
  - density, parallel flow, parallel/perpendicular temperature, heat-flux moments,
  - free-energy spectra versus Hermite and Laguerre index.
- [x] Add closure hooks for later reduced-moment runs:
  - truncation-only baseline,
  - hypercollisions,
  - damping-rate/apply hooks tested on modal arrays.
- [ ] Add production closure models after the collisionless linear baseline is validated:
  - Beer/Smith/Hammett-type closures.
- [x] Tests:
  - Hermite and Laguerre orthogonality,
  - Maxwellian moment normalization,
  - transform round trips,
  - parity of low-order moments with collocation integrals,
  - small fixtures inspired by GX's `test_laguerre_transform`, `test_moments`, and closure input cases.

## Phase 6: Quasineutrality and Diagnostics

- [x] Implement adiabatic-electron electrostatic phi solve.
- [x] Add zonal-mode correction if the chosen formulation requires it.
- [x] Implement kinetic-electron phi solve as second milestone.
- [x] Implement diagnostic integrals:
  - density response,
  - heat flux / quasilinear ingredients,
  - spectra,
  - mode amplitude.
- [x] Add GX-style spectra hooks for later output:
  - `ky`,
  - `kx`,
  - `kxky`,
  - `z`,
  - Hermite/Laguerre free-energy spectra when Phase 5A exists.
- [x] Unit tests:
  - zero distribution gives zero `phi`,
  - no-zonal and zonal paths,
  - quasineutrality residual near machine precision,
  - algebraic small-fixture checks against the Gyaradax/GKW phi-solve convention,
  - flux/spectrum normalization checks.
- [x] Add a direct reduced Gyaradax/GKW phi/RHS parity fixture using the Phase 7 coupled precompute inputs.

## Phase 7: Linear RHS Residual

- [x] Implement a public matrix-free residual:

```python
rhs = linear_residual(df, geometry, params, precomputed)
```

- [x] Implement RHS terms in isolated, testable functions:
  - `parallel_streaming`,
  - `magnetic_drift_advection`,
  - `mirror_force`,
  - `equilibrium_drive`,
  - `parallel_field_drive`,
  - `drift_field_drive`,
  - dissipation.
- [x] Implement precomputation for geometry/species coefficients.
- [x] Keep RHS differentiable with respect to `df`, continuous geometry arrays, and physical parameters.
- [x] Unit tests:
  - each term has expected shape and zero-input behavior,
  - isolated manufactured-solution derivative checks,
  - full RHS linearity in `df` for fixed geometry,
  - `jax.jit` compatibility,
  - reverse-mode gradients versus finite differences for `df`, continuous geometry arrays, and species parameters.

Phase 7 baseline note: dissipation is implemented as optional linear perpendicular damping with zero default. Spectral/modal damping and benchmark-tuned hyperdissipation should remain inactive until needed by Phase 8/10 tests.

## Phase 8: Time Advancement and Growth Rates

- [x] Implement RK4 single-step and multi-step scan.
- [x] Implement fixed timestep first.
- [x] Add CFL estimate only after fixed-step tests pass.
- [x] Implement per-`ky` amplitude normalization for linear growth-rate extraction.
- [x] Keep the time-advance interface compatible with later GX-style nonlinear pseudo-spectral runs, including explicit dealias/filter hooks that can be no-ops in the linear solver.
- [x] Compute:
  - growth rate `gamma`,
  - real frequency `omega`,
  - mode amplitude,
  - mode structure.
- [x] Unit tests:
  - zero-input invariance,
  - RK4 order test on a scalar linear ODE,
  - growth-rate recovery for a known scalar mode,
  - JIT and gradient checks through a short solve.

Phase 8 baseline note: the integrator is fixed-step RK4 with an explicit post-step filter hook. Per-`ky` amplitude normalization is exposed as a separate utility so production runs can choose diagnostic-window cadence before applying it.

## Phase 9: Eigenvalue and Objective Interfaces

- [x] Expose residual for external nonlinear/eigenvalue solvers.
- [x] Add matrix-free linear operator wrapper for one `ky` or a mode chain.
- [x] Add optional Arnoldi/eigensolver path if useful for faster linear growth rates.
- [x] Implement differentiable objective functions:
  - max growth rate,
  - selected `ky` growth rate,
  - weighted quasilinear proxy `sum gamma / <k_perp^2>`,
  - mode-structure penalties.
- [x] Unit tests:
  - objective shapes and finite values,
  - AD gradients with respect to `R/L_T`, `R/L_n`, `q`, `shat`, and continuous geometry controls,
  - finite-difference agreement on reduced grids.

Phase 9 baseline note: dense eigensystem helpers are intentionally limited to small validation problems. Production linear scans should continue using matrix-free residual actions, RK4 growth extraction, or a future sparse/Arnoldi adapter.

## Phase 10: Benchmarks and Validation

- [x] Reproduce reduced Gyaradax/GKW circular and s-alpha geometry tests.
- [x] Add a direct reduced Gyaradax/GKW-style phi/RHS parity fixture using the Phase 7 coupled precompute inputs.
- [x] Add a reduced zonal-flow invariant test: constant zonal mode is stationary in a flat flux tube.
- [x] Add a true long-time Rosenbluth-Hinton zonal-flow plateau gate that passes the documented \(t>80\) residual using GKW finite-difference fallback stencils and late-window mean convergence.
- [x] Add Cyclone Base Case selected-`ky` linear ITG growth-rate test. The
  matched 48/32/8 production-control regression now passes against the
  \(0.179\) GKW/Gyaradax scalar target; full ITG scan parity remains separate.
- [x] Add a first GX-inspired benchmark fixture from `relevant-codes/gx/benchmarks/linear/ITG_cyclone/` after verifying TOML compatibility.
- [x] Add named scalar benchmark targets for the documented Rosenbluth-Hinton residual and Cyclone Base Case growth-rate references.
- [x] Load GX NetCDF growth/frequency reference curves from `omega_kxkyt` and convert selected `ky` points into optimization targets.
- [x] Load local GX/GS2 eik-style geometry tables for future stellarator geometry parity checks.
- [x] Add executable reduced RH and CBC validation gates that report observed values, normalized residuals, and pass/open status against production targets.
- [x] Add a GX/GS2 eik metric gate verifying the imported table maps into the solver `k_perp^2` contract.
- [x] Add a late-window least-squares growth diagnostic for CBC/GX/GKW-style growth-rate gates.
- [x] Add a calibrated reduced RH crossing gate as a deterministic regression hook while keeping the production long-time RH plateau open.
- [x] Replace the calibrated RH crossing in the active validation path with a true long-time plateau metric over the \(t>80\) window.
- [x] Add reusable spectral modal damping filters for benchmark-controlled recurrence studies.
- [x] Add solver-geometry-to-GX/GS2-eik field-by-field parity reports and a corresponding geometry gate.
- [x] Add a reduced validation-gate example that writes CSV summaries and a paper figure for the RH endpoint, RH plateau, Cyclone, and GX/eik gates.
- [x] Add a solver-produced DESC fixture geometry export gate for GX/GS2 eik-compatible fields, including \(B\), \(\nabla_\parallel\), metric elements, summed magnetic drifts, and \(\kperp^2\).
- [x] Correct GIST/GS2 eik drift-column ordering and add a three-fixture GX/VMEC GIST external eik-suite gate.
- [x] Replace the RH/CBC post-step parallel modal-damping default with an in-residual GKW/Gyaradax-scaled `disp_par` recurrence-control term.
- [x] Correct the Cyclone selected-`ky` gate to use the GKW cell-centered `s` grid, `nperiod=5`, the single-mode `ky=0.5` convention, and a production-control wrapper that stores only late-window amplitudes.
- [x] Close the active RH late-time plateau gate by adding GKW finite-difference parallel/velocity fallbacks, direct fourth-difference `disp_par`/`disp_vp` recurrence operators, exact zonal initialization, and a late-window mean-convergence check.
- [x] Harden the production-control Cyclone selected-`ky` gate with GKW finite-difference velocity and zero-boundary parallel fallback backends, jitted window advancement, and documented GKW/GX tolerance reporting.
- [x] Add the GKW/Gyaradax sign-dependent upwind parallel fallback for CBC Term I/Term VII, including open-boundary shift maps and fused `disp_par` recurrence control.
- [x] Add a CBC term-level parity audit for magnetic drift, equilibrium drive, drift-field drive, GKW boundary maps, grid/velocity normalization, and assembled RHS conventions.
- [x] Add a reduced CBC trace report that records selected-mode raw/physical amplitude, per-window and fitted growth, phi/state/RHS norms, and log-normalization after fixed windows.
- [x] Add an external Gyaradax reduced trace exporter and compare physical selected-`ky` amplitude/growth fields directly against `CycloneTrace`.
- [x] Add normalization-equivalent physical phi/state/RHS norm comparisons to the reduced Gyaradax trace report.
- [x] Add a production-control-grid Gyaradax smoke trace profile for `CycloneTrace` using \(N_z=48\), \(N_{v_\parallel}=32\), \(N_\mu=8\), 20 steps per window, and four windows.
- [x] Extend the Gyaradax trace comparison to the full 80-window production-control run.
- [x] Add a GKW `time.dat` diagnostic loader for the same `CycloneTrace` schema.
- [x] Build the local serial/no-FFT GKW executable, run the bundled linear `simple_example`, and convert its real `time.dat` output into `figures/gkw_simple_example_time_trace.csv`.
- [x] Generate a matched linear selected-`ky` GKW `time.dat` trace at the production-control grid/window settings and convert it into the `CycloneTrace` CSV schema.
- [x] Add explicit production Cyclone growth-diagnostic selection between selected-mode late-fit and GKW-style late mean of per-window growth values.
- [x] Add a GKW `parallel_phi.dat` loader, solver-side selected-`ky` parallel `|phi|^2` trace, and row-normalized profile comparison for the matched production-control Cyclone run.
- [x] Add a `parallel_phi.dat` alignment/normalization audit showing that GKW-unweighted field normalization removes the raw total-power scale, while reversal/circular shifts do not reduce the remaining profile error.
- [x] Add localized `parallel_phi.dat` profile-mismatch diagnostics for peak-position, profile-width, and worst `(time,z)` signed shape error.
- [x] Add a selected-mode Cyclone profile-operator audit showing that the field solve and RHS assembly are exact at the worst profile row, while the local matrix-versus-GKW-upwind streaming delta is non-negligible and the local field-drive delta is negligible.
- [x] Add a source-reconstructed GKW Fortran Term I audit showing that the selected-mode parallel streaming/upwind stencil, sign choice, and `idisp=2` recurrence normalization match `linear_terms.F90::vpar_grad_df_4d_testnewbc` to roundoff.
- [x] Add a source-sequence Cyclone time-update/normalization audit showing that the GKW RK4 staging and `normalise.F90` unweighted field normalization agree with the solver window trace to roundoff.
- [x] Add a GKW diagnostic/field-packing audit showing that `dist.F90::get_phi`, the `index_function.F90` field layout, `diagnostic.F90::parallel_phi`, and the GKW `ky`/`kx` field spectra match the solver field diagnostics to roundoff.
- [x] Add a reduced GKW `matdat.F90` sparse matrix/source convention audit showing that homogeneous source handling, duplicate triplet compression, explicit `dtim*(source+mat*fdis_tmp)`, and `complex-real` splitting match the matrix-free residual to roundoff.
- [x] Add a production-control selected-`ky` source-level coefficient audit for GKW Terms II, IV, V, VII, and VIII, reconstructing `linear_terms.F90::{vdgradf,trapdf_4d,ve_grad_fm,vpgrphi_3_newbc,vd_grad_phi_fm}` and matching the current coefficient/action insertion to roundoff.
- [x] Add a production-control GKW `ltrapping_arakawa` fused `igh` audit comparing the combined Term I/IV Hamiltonian stencil and `disp_vp=0.2` diffusion against the solver's separated streaming/mirror fallback.
- [x] Add an optional matrix-free GKW `ltrapping_arakawa` fused `igh` RHS backend that precomputes the combined Term I/IV Hamiltonian stencil plus in-operator `disp_par`/`disp_vp` diffusion and matches the source-reconstructed `igh` action in a direct JIT-tested reduced fixture.
- [x] Rerun the production selected-`ky` growth/profile gates with `parallel_derivative_model="gkw_igh"` and compare against the matched GKW `time.dat`/`parallel_phi.dat` traces.
- [x] Add a non-destructive GKW `cosine2` patch path. The helper copies GKW to a scratch tree, adds a six-character `finit='cosin2'` branch implementing \(1+\cos(2\pi s)\), and leaves `relevant-codes/gkw/` untouched.
- [x] Build and run the patched serial/no-FFT GKW selected-`ky` production-control case, then store the raw `cosin2` `time.dat` and `parallel_phi.dat` fixtures.
- [x] Compare the solver `cosine2`/`gkw_igh` production-control growth and profile traces against the patched GKW `cosin2` fixtures.
- [x] Add a combined selected-`ky` gap audit that aligns solver post-window samples with the patched GKW `cosin2` `time.dat`/`parallel_phi.dat` fixtures and reports late-fit growth, late-window mean, final-window growth, profile-shape error, worst `(t,z)`, and total-power normalization.
- [x] Document that the direct compact GKW trace path is `finit='cosine'`; `cosine2` would require a GKW source patch or restart/state-injection path because the original `finit` selector is a six-character Fortran option and has no native `cosine2` branch.
- [x] Diagnose the remaining `cosine2`/`cosin2` growth and parallel-profile gap after the initialization mismatch is removed: aligned late-fit growth remains low by \(2.26\times10^{-2}\), late-window mean by \(2.18\times10^{-2}\), and row-normalized profile error peaks at \(2.96\times10^{-2}\), while total-power normalization agrees to \(2\times10^{-6}\).
- [x] Compare the evolved GKW velocity-space/distribution diagnostics (`distr*.dat` or an explicit restart/state dump) against the solver state at the same selected-`ky` time, since field normalization, output ordering, initialization, and source-level term conventions are now ruled out as dominant causes.
- [x] Add a loader for GKW peak-\(\phi\) `distr*.dat` velocity-space slices and a solver-side final-slice diagnostic with the same \(f\,\Delta\mu\,\Delta v_\parallel/\phi\) normalization.
- [x] Run the patched `cosin2` velocity-slice audit: velocity grids and final time match GKW (`v_\parallel` error \(5.0\times10^{-5}\), \(v_\perp\) error \(4.7\times10^{-5}\), time error zero), but the normalized distribution remains open with complex max error \(3.53\times10^{-2}\) and relative \(L^2\) error \(3.78\times10^{-1}\).
- [x] Add a velocity-slice convention audit that explicitly respects GKW's Fortran layout: `global_vpar_mu(n_vpar,n_mu)` is written as `mu` rows and \(v_\parallel\) columns by `output_slice_2d`.
- [x] Use the convention audit to identify that reversing the \(v_\parallel\) columns is the best simple convention, reducing the complex max error from \(3.53\times10^{-2}\) to \(1.85\times10^{-2}\).
- [x] Add even/odd \(v_\parallel\) decomposition to the convention audit; the odd opposite-sign \(L^2\) error is \(1.90\times10^{-3}\), much lower than the odd same-sign \(6.67\times10^{-3}\).
- [x] Add a controlled \(v_\parallel\)-odd RHS sign audit for the evolved state, keeping GKW's 1-based `k=1,\ldots,nvpar` ordering and `vpgr=-v_{\max}+(k-1/2)\Delta v` explicit.
- [x] Use the sign audit to rule out a global fused `ltrapping_arakawa`/`igh` sign reversal: flipping the fused Term I/IV block blows up the final-slice error to \(\sim 17\).
- [x] Use the sign audit to narrow the remaining issue to the parallel field-drive path: flipping only `vpgrphi_3_newbc`/Term VII makes the direct Fortran layout the best layout and reduces the direct complex max error from \(3.53\times10^{-2}\) to \(2.01\times10^{-2}\), but this is diagnostic only and is not yet a physics change.
- [x] Add a Term VII field-variable convention audit across Terms V, VII, and VIII. It rules out a global `phi` sign/conjugation convention: global sign, global conjugation, and global negative-conjugation all worsen the direct slice error. The best diagnostic variant is Term VII-only conjugation, reducing the direct max error from \(3.53\times10^{-2}\) to \(1.60\times10^{-2}\), with Term VII-only negative-conjugation close at \(1.70\times10^{-2}\).
- [x] Inspect the immediate Term VII insertion/output phase path enough to rule out the simplest hidden conventions: `vpgrphi_3_newbc` inserts real scalar coefficients into `iphi`, `add_element` maps that directly to the connected `iphi` column, `put_element` stores the coefficient unchanged in `complex` or `complex-real` format, `dist.F90::get_phi` copies `fdis(iphi)` directly, and `diagnostic.F90::velocity_space_output` normalizes by `phi`, not `conjg(phi)`.
- [x] Add a velocity-slice phase-alignment audit. Optimal unit phase does not collapse the error: the best unit-phase variant remains `reverse_vpar_columns:identity` with max error \(1.89\times10^{-2}\), and even an unconstrained complex scale only reaches \(1.80\times10^{-2}\). This rules out a pure eigenvector/output phase explanation for the final-slice mismatch.
- [x] Continue the true GKW Term VII complex phase path across \(k_y\)/Fourier signs, field-equation packing, and final-state evolution. The new source-path audit shows the selected positive-\(k_y\) `mode.F90`/`dist.F90::get_phi`/`vpgrphi_3_newbc` path is direct, not conjugated or sign-flipped; the Term VII-only conjugation improvement is therefore diagnostic and not a source convention.
- [x] Add a non-destructive GKW multi-time velocity-slice dump to localize the full evolved-distribution mismatch beyond the final peak-\(\phi\) `distr*.dat` slice. The copied GKW `diagnostic.F90` patch writes `distr1_00000020.dat`--`distr4_00001600.dat` at every diagnostic output window while preserving the unsuffixed final `distr*.dat` files.
- [x] Add a public loader for suffixed GKW `distr*_<ntotstep>.dat` series. The patched selected-`ky` `cosin2` run produced 320 files, read as 80 snapshots with shape `(80, 8, 32)`, matching the production-control time history from step 20 through 1600.
- [x] Use the multi-time GKW velocity-slice series to compare solver and GKW peak-\(\phi\) velocity slices over early, mid, and late windows. The mismatch is already nonzero at step 20 but grows cumulatively by step 800: direct complex max errors are \(3.99\times10^{-3}\), \(3.67\times10^{-2}\), and \(3.53\times10^{-2}\) at steps 20, 800, and 1600.
- [x] Use the multi-time audit to localize the cumulative step-20-to-step-800 distribution mismatch through controlled `gkw_igh` and Term VII/field-convention variants. `flip_igh` still blows up by the final snapshot, global field conjugation worsens mid/final snapshots, Term VII sign flip is best at step 800, and Term VII conjugation is best at step 1600.
- [x] Trace the selected-mode Term VII \(k_y\)/Fourier sign and field-packing path at source level, including GKW `mode.F90`, `dist.F90::get_phi`, `linear_terms.F90::vpgrphi_3_newbc`, and the real/complex storage convention. The audit passes with direct field roundtrip error zero and direct Term VII action error \(3.47\times10^{-18}\), while conjugating or negating the packed field changes the Term VII action by \(\simeq2.89\times10^{-2}\).
- [x] Refocus the remaining selected-mode distribution/growth gap on cumulative state evolution and the fused GKW `ltrapping_arakawa`/`gkw_igh` path now that Term VII source-level mode sign, field packing, matrix insertion, and diagnostic packing are ruled out. The new multi-window `igh` audit shows the fused-vs-separated operator delta is largest at the first sampled window and decreases by windows 40 and 80, so it is an early operator-shape mismatch rather than the sole source of the step-20-to-step-800 velocity-slice growth.
- [x] Add a solver-side selected-mode source-term time trace under the promoted `gkw_igh` backend. It records raw post-window phi/state/RHS norms, term norms for fused `igh`, magnetic drift, equilibrium drive, parallel field drive, drift-field drive, and dissipation, the accumulated log-normalization, and a direct RHS reconstruction error.
- [x] Add the non-destructive GKW compact state-trace patch, loader, and solver/source-trace comparator. The copied GKW diagnostic patch writes `stellarator_gk_state_trace.dat` with `step`, `time`, `state_norm`, and `phi_norm`; the Python loader maps it into `GkwStateTrace`.
- [x] Run/capture the matched production-control GKW `cosin2` compact state trace and compare it against the solver `CycloneSourceTermTrace` with GKW's post-normalization diagnostic timing. The compact norm comparison passes at tolerance \(5\times10^{-3}\), with max state-norm error \(3.31\times10^{-3}\) and field-norm error \(6.11\times10^{-16}\).
- [x] Add and run a full selected-mode GKW state dump discriminator after the compact state-trace result. The copied GKW diagnostic patch writes `stellarator_gk_selected_state_<step>.dat` files; the sampled production-control fixtures at steps 20, 800, and 1600 load as shape `(3, 32, 8, 48)`.
- [x] Compare the sampled selected-mode state dump against the solver `cosine2`/`gkw_igh` state trace with snapshot-wise complex phase alignment. Direct layout is best, but the comparison remains OPEN: maximum phase-aligned `phi(z)` error \(5.85\times10^{-2}\), maximum phase-aligned full-state error \(4.78\times10^{-2}\), and worst state relative \(L^2\) error \(3.19\times10^{-1}\) at step 800.
- [x] Add a matched GKW RHS/source action trace for the selected mode and the same early/mid/final snapshots. The copied GKW patch now tags linear matrix/source entries by term, writes `stellarator_gk_rhs_trace_<step>.dat`, and stores sampled fixtures at steps 20, 800, and 1600 with shape `(3, 9, 32, 8, 48)`.
- [x] Compare the GKW RHS/source action trace against solver full selected-mode term-action arrays, not only solver term norms. The GKW trace is `dtim`-scaled and internally sums to the stored total with max error \(9.55\times10^{-18}\); after the GKW internal `KTHRHO/kthnorm` convention fix, the best direct solver/GKW action comparison has max error \(2.24\times10^{-5}\), dominated by `vdgradf`.
- [x] Add a GX DESC-block `eik.out` loader and compare solver-produced DESC/GX-convention geometry against the matched DSHAPE external-format fixture.
- [x] Add a reduced stellarator fixture:
  - fixed surface,
  - fixed `alpha`,
  - small `ky` grid,
  - reference geometry arrays.
- [x] Compare stellarator geometry quantities against precomputed reference data.
- [ ] Compare solver-computed ITG growth-rate scans against available GS2/GX/GKW-style references.
- [x] Compare solver-produced geometry arrays against local GX/GS2/GX-DESC-style `eik` outputs where fixtures are available.
- [x] Supplement the matched DESC/GX eik fixture with a truly independent external eik producer suite using compatible GX/VMEC GIST fixtures. A future independently generated DESC/GX-specific runner can still strengthen DESC convention parity, but the current external-producer gate is no longer the matched block-eik fixture.
- [x] Add convergence tests over:
  - [x] `N_s`,
  - [x] `N_vparallel`,
  - [x] `N_mu`,
  - [x] `ky`,
  - [x] timestep.
- [x] Record benchmark commands and tolerances in `STATUS.md` and, later, in docs.

Phase 10 baseline note: the current validation tranche covers reduced deterministic fixtures, manufactured convergence over parallel/velocity/`ky`/time resolution, a GX Cyclone input-contract fixture, named RH/CBC scalar targets, GX NetCDF growth-curve loading, GX/GS2 eik-table loading, executable validation gates, late-window growth fitting, a passing true RH late-plateau metric, in-residual GKW/Gyaradax-scaled `disp_par` recurrence control, direct GKW fourth-difference `disp_vp` velocity recurrence control for the RH fallback path, a GKW-aligned Cyclone selected-`ky` setup with a production-control memory-light gate, GKW finite-difference Cyclone fallback controls, the GKW/Gyaradax sign-dependent upwind parallel fallback for CBC Term I/Term VII, a passing CBC term-level parity audit, GKW time/profile/state/RHS loaders, selected-mode state/source traces, GKW final and multi-time `distr*.dat` velocity-slice loaders, real serial GKW run fixtures, explicit late-fit versus GKW-style late-window-mean growth diagnostics, DESC/GX/eik parity gates, an explicit independent GX/VMEC GIST external eik-producer report/gate, a direct row-normalized `parallel_phi.dat` profile-shape gate, and a validation-gate plotting example used in `main.tex`. The RH plateau, source-level algebraic audits, production-control CBC selected-`ky` scalar regression, convergence tests, eik parity gates, selected-mode RHS/action parity gates, imported-state one-window replay gates, initial/first-window selected-state contract, and production-control `parallel_phi.dat` profile-shape gate pass their current contracts. The previous `igh_or_term_i` coefficient/action gap is closed by matching GKW's discrete `geom.F90::logbderiv` `gfun` convention in the `gkw_igh` parity setup: input trace max \(5.33\times10^{-15}\), matrix trace max \(5.77\times10^{-15}\), full solver-vs-GKW RHS/action max \(6.22\times10^{-12}\), same-state replay max \(2.35\times10^{-12}\), and fused source-level `igh` max \(1.81\times10^{-18}\). The imported GKW selected-state one-window replay passes: the early 20-to-40 window has max phase-aligned gate error \(1.12\times10^{-9}\), and the mid-run 780-to-800-to-820 windows have max error \(2.03\times10^{-10}\). The initial/first-window contract is also closed: GKW pre/post initial `normalize(2)` selected states are identical, the solver initialized distribution matches GKW at \(4.88\times10^{-19}\), GKW stores zero selected \(\phi\) at step 0 while the solver's self-consistent solved field has max norm \(1.50\times10^{-2}\), and replay from GKW step 0 to step 20 closes at \(1.17\times10^{-9}\). The direct production-control `cosin2` profile-shape gate now passes with maximum row-normalized error \(3.97\times10^{-7}\); its tolerance ladder passes \(5\times10^{-2}\), \(3\times10^{-2}\), \(2\times10^{-2}\), and \(1\times10^{-2}\), while total-power ratio mean error is \(1.73\times10^{-8}\). The remaining consistency failures are sharper: the compact GKW state trace matches solver post-normalization norms at \(5\times10^{-3}\) tolerance, but the full selected-mode state dump remains OPEN after snapshot-wise phase alignment with maximum \(\phi(z)\) error \(5.85\times10^{-2}\), maximum full-state error \(4.78\times10^{-2}\), and worst state relative \(L^2\) error \(3.19\times10^{-1}\) at step 800; the multi-time velocity slices show cumulative growth from \(3.99\times10^{-3}\) at step 20 to \(3.67\times10^{-2}\) at step 800. Full ITG \(k_y\)-scan parity against GX/GKW/GS2-style references remains open and is now the next benchmark target. DESC-driven optimization examples remain labeled reduced because they are fixed-topology demonstration runs rather than full DESC optimization campaigns.

2026-06-02 update: the selected-mode Term VII mode/Fourier sign and field-packing audit now passes. It confirms `mode.F90` uses the positive single-mode `krho=0.5`, the single nonzonal \(k_x\) chain has open GKW boundary maps (`ixplus=ixminus=0`, represented as `-1` in Python), `dist.F90::get_phi` pulls `iphi` directly, and `vpgrphi_3_newbc` matches the direct packed field with action error \(3.47\times10^{-18}\). Conjugating or negating the packed field changes the Term VII action by \(\simeq2.89\times10^{-2}\), so diagnostic Term VII sign/conjugation variants are not source-level conventions.

2026-06-02 update: the multi-window `ltrapping_arakawa`/`igh` series audit samples windows 1, 40, and 80. It remains OPEN, but the worst fused-vs-separated maximum profile delta is already at window 1: \(1.40\times10^{-2}\), then \(2.80\times10^{-3}\) at window 40 and \(6.15\times10^{-4}\) at window 80. This makes the fused-operator mismatch an early operator-shape issue; the remaining mid-run GKW/solver distribution mismatch still requires state-history or source-term time-trace comparison.

2026-06-02 update: the solver-side selected-mode source-term trace is now available through `run_cyclone_base_case_source_term_trace` and `figures/cyclone_source_term_trace.csv`. The production-control `cosine2`/`gkw_igh` trace samples windows 0, 1, 40, and 80 and closes the raw term sum against the public RHS with maximum reconstruction error zero in the stored CSV rows. The matched GKW-side source/restart/state trace is still open.

2026-06-02 update: the GKW-side state-history path is now populated with `fixtures/gkw_cyclone_selected_ky_cosin2_state_trace.dat`, generated from a copied serial/no-FFT GKW tree prepared with `--state-trace`. `figures/gkw_cosin2_cyclone_state_trace_comparison.csv` compares it against the solver's post-normalization all-window `CycloneSourceTermTrace`; the compact norm comparison passes at tolerance \(5\times10^{-3}\), with max state-norm error \(3.31\times10^{-3}\), time error \(7.02\times10^{-14}\), and field-norm error \(6.11\times10^{-16}\). This is a norm-level check only; the full distribution/profile/growth gap remains open.

2026-06-01 update: the optional matrix-free `gkw_igh` residual backend now matches the source-reconstructed fused `igh` action on a reduced direct/JIT fixture. The full production-control `gkw_igh` rerun with GKW-native `finit='cosine'` remains OPEN: late-fit growth is \(0.164465\) versus matched GKW \(0.188531\), late-window mean is \(0.154739\) versus \(0.180408\), and the row-normalized `parallel_phi.dat` profile comparison has maximum error \(3.28\times10^{-2}\). A non-destructive GKW `cosine2` patch path now exists as `scripts/prepare_gkw_cosine2_run.py`; because GKW's `finit` selector is six characters, the copied source uses the branch name `cosin2` for the solver/Gyaradax \(1+\cos(2\pi s)\) profile. The patched serial/no-FFT GKW run builds and completes. Its stored `cosin2` fixtures give GKW late-fit growth \(0.187415\), late-window mean \(0.177999\), and final-window growth \(0.188741\). The matched solver `cosine2`/`gkw_igh` values are \(0.163899\), \(0.156237\), and \(0.139452\); the row-normalized `parallel_phi.dat` profile comparison improves to \(2.96\times10^{-2}\) but remains OPEN. The new aligned selected-`ky` gap audit reports solver late-fit \(0.164862\), GKW late-fit \(0.187415\), late-fit delta \(2.26\times10^{-2}\), total-power ratio mean \(0.99999998\), and maximum total-power deviation \(1.95\times10^{-6}\). The initialization and field-normalization mismatches are therefore not the dominant remaining causes.

## Phase 11: CPU Performance and Differentiability Hardening

- [x] Profile reduced and target-size linear runs on CPU:
  - reduced-grid JIT residual smoke profiler,
  - static target-grid memory estimates before allocating large arrays.
- [x] Compare algorithmic scaling against GX qualitatively, while treating GPU-native CUDA performance as a non-goal for the first differentiable CPU-oriented solver.
- [x] Remove avoidable Python loops from traced paths:
  - RK4 history path uses `jax.lax.scan`,
  - memory-sensitive endpoint path uses `jax.lax.fori_loop`.
- [x] Cache/precompute static coefficient arrays in Phase 7 precompute objects and expose precompute byte accounting.
- [x] Use `jax.jit`, `jax.vmap`, and `jax.lax.scan` carefully with static arguments:
  - public `jitted_linear_residual`,
  - dense reduced-operator construction remains `vmap`-based,
  - fixed-step integration remains scan/loop based.
- [x] Check memory footprint for target grids with dimension-only and precompute-based estimators.
- [x] Add performance smoke tests with relaxed thresholds.
- [x] Verify gradients for target objectives are finite and stable on reduced JIT/no-history objective paths.
- [x] Document which operations are differentiable and which are treated as static topology in `docs/performance_and_differentiability.md`.

Phase 11 baseline note: the current hardening pass adds CPU profiling/memory tools and differentiability smoke coverage. Production timing comparisons should still be rerun after full Rosenbluth-Hinton/Cyclone/GX validation fixtures are selected.

## Phase 12: Optimization Integration

- [x] Define geometry-parameter inputs for optimization:
  - profile gradients,
  - rotational transform / shear,
  - beta or pressure-gradient knobs,
  - placeholder boundary/equilibrium coefficients for future DESC/Boozer adapters.
- [x] Implement single-surface/single-alpha objective evaluation.
- [x] Implement scans over `rho`, `alpha`, and `ky`.
- [x] Add quasilinear objective compatible with gradient-based stellarator design loops.
- [x] Add examples showing `jax.grad`/`jax.value_and_grad` through the objective.
- [x] Add a small toy optimization example before using full DESC equilibria.
- [x] Add benchmark-target least-squares objective wrappers.
- [x] Add a reduced DESC DSHAPE fixture optimization example.

Phase 12 baseline note: the current optimization layer is fixed-topology and reduced-grid oriented, with `examples/optimization_loop.py` available for printing analytic per-iteration objective/growth diagnostics and `examples/desc_fixture_optimization_loop.py` available for the extracted DESC DSHAPE fixture. DESC/Boozer equilibrium arrays should replace the toy equilibrium-coefficient modulation before production stellarator optimization.

Phase 12 DESC-array note: the solver now supports a supplied imported geometry object in `single_surface_objective`, and `build_desc_geometry_from_arrays` maps DESC-sampled Boozer/Clebsch flux-tube arrays into the internal solver geometry. `single_surface_benchmark_objective` can compare selected reduced diagnostics against named benchmark targets on this imported geometry. `desc_geometry_arrays_from_equilibrium`, `desc_geometry_arrays_from_path`, and `scripts/extract_desc_geometry_fixture.py` provide direct DESC equilibrium/example/HDF5 extraction paths. DESC should remain the upstream equilibrium and sensitivity provider; refactoring or vendoring DESC internals into this solver is not needed for the first coupling.

## Task-Roadmap Alignment and Remaining Consistency Gaps

The current code base is consistent with the `task.tex` target in its core
architecture: it is JAX-first, differentiable, linear electrostatic, flux-tube
oriented, uses perpendicular Fourier modes, exposes time-advance and residual
interfaces, supports spectral and GKW-parity fallback operators, and consumes
stellarator geometry through Boozer/Clebsch-style flux-tube arrays. The strongest
completed validation gates are the true long-time Rosenbluth--Hinton plateau,
the production-control Cyclone selected-`ky` scalar growth regression, term-level
GKW/Gyaradax CBC algebra, DESC/GX/eik geometry contracts, independent GX/VMEC
GIST eik-producer coverage, and reduced differentiable optimization examples.

The remaining consistency failures to fix before calling the solver production
ready are:

- Selected RHS/action parity against the patched GKW trace is now closed at
  roundoff for the production selected-`ky` control path after matching
  GKW's discrete `geom.F90::logbderiv` `gfun` convention in the `gkw_igh`
  setup. The full solver-vs-GKW RHS/action trace is phase-aligned to
  \(6.22\times10^{-12}\), same-state replay is \(2.35\times10^{-12}\),
  mid-run same-state replay is \(2.36\times10^{-12}\), and the fused
  `igh_or_term_i` source audit is \(1.81\times10^{-18}\). This is no longer
  an open operator/action consistency failure.
- Full selected-mode state-history parity against GKW remains OPEN even though
  scalar growth now passes: the phase-aligned full-state error peaks at
  \(4.78\times10^{-2}\), the phase-aligned \(\phi(z)\) error at
  \(5.85\times10^{-2}\), and the worst state relative \(L^2\) error is
  \(3.19\times10^{-1}\) at step 800.
- Multi-time velocity-slice parity remains OPEN: direct complex max error grows
  from \(3.99\times10^{-3}\) at step 20 to \(3.67\times10^{-2}\) at step 800,
  so the mismatch is cumulative and not a final-output-only convention.
- The direct row-normalized GKW `parallel_phi.dat` profile-shape gate now
  passes on the current production-control `cosin2` path with maximum error
  \(3.97\times10^{-7}\); the older \(3\times10^{-2}\) mismatch is kept only as
  historical context for the pre-contract audit path.
- A public multi-`ky` Cyclone/ITG scan gate is now implemented against
  GX/GKW/GS2-style growth/frequency references, with optional profile-error
  slots. Production scan parity remains OPEN; only the selected Cyclone
  `ky=0.5` scalar gate is currently passing at production-control resolution.
- `task.tex` also asks for stellarator/TEM-style benchmark cases, production
  growth-rate/mode-frequency/mode-structure validation, CPU timing evidence on
  target-size runs, and a simplified but real DESC-driven optimization campaign;
  the current DESC examples remain reduced fixed-topology demonstrations.

## Immediate Next Round: Toward a Trusted Stellarator Simulation

Goal: move from reduced stellarator-geometry demos to a credible linear
stellarator flux-tube simulation, without weakening the already-passing
GKW/Gyaradax selected-`ky` parity gates. A "trusted" first stellarator run means:
DESC or eik-compatible geometry is audited before solving, the selected
multi-`ky` physics path has a mode-structure check, growth/frequency extraction
is converged, and the output can be reproduced by a single example script.

Use these guardrails throughout the round:

- keep the passing RH plateau, CBC selected-`ky` scalar gate, GKW RHS/action
  trace, imported-state one-window replay gates, initial-state contract,
  row-normalized `parallel_phi.dat` profile gate, DESC/GX/eik geometry gate,
  and independent GX/VMEC GIST eik-producer gate as regression tests;
- keep the current exact-grid GX scan gap visible:
  `gamma(0.3)=0.14325556436195153` versus calibrated reference
  `0.3080402563529299`, while the `ky=0.5` anchor is much closer;
- keep DESC-driven optimization examples labeled reduced until the
  multi-`ky`, mode-structure, geometry, convergence, and timing gates below
  pass together.

### 1. Close the multi-`ky` physics discriminator

- [x] Add a code-independent per-`ky` complex mode-structure fixture contract,
  CSV loader/writer, GKW selected-state-to-fixture adapter, and comparison gate.
  The gate reports scalar growth error, real-frequency error, row-normalized
  complex `phi(z)` error with phase alignment, optional state/moment error,
  and separate pass/fail masks. Its phase-aligned `phi(z)` error can feed the
  existing multi-`ky` scan gate `profile_error` field.
- [x] Wire the fixture comparison directly into the public multi-`ky` scan
  gate with `evaluate_cyclone_ky_scan_gate_from_mode_structure_fixtures`, and
  add `run_cyclone_base_case_mode_structure_fixture` so the current solver can
  emit a per-`ky` complex `phi(z)` fixture from the same single-mode scan path
  used for scalar growth/frequency diagnostics.
- [x] Add a GX external fixture reader/export path. `load_gx_mode_structure_fixture`
  reads GX `.big.nc` `Diagnostics/Phi(time,ky,kx,z,ri)` complex fields and can
  attach compact `.out.nc` `omega_kxkyt` growth/frequency references; the
  `examples/export_gx_mode_structure_fixture.py` command writes the standard
  CSV fixture for `ky=(0.3,0.5)` or any requested scan subset. The repository's
  shipped GX Cyclone files include compact `.out.nc` spectra but not the
  corresponding `.big.nc` full-field diagnostics, so a fresh GX run or external
  artifact is still needed to populate the real low-`ky` reference.
- [x] Add the one-command external-vs-solver discriminator runner
  `examples/run_cyclone_mode_structure_gate.py`. It generates the solver
  fixture, loads either a generic per-`ky` mode-structure CSV fixture, directly
  generates the reduced GX-style moment-RHS fixture, or loads a GX `.big.nc`
  reference fixture. It explicitly converts GX `theta` to either raw `theta` or
  `theta/(2*pi)`, optionally resamples the reference fixture to the solver grid
  with a declared periodic policy, and writes solver/reference fixtures plus a
  per-`ky` gate report. The runner now requires exactly one reference source:
  self-check, generic CSV fixture, reduced moment RHS, or GX `.big.nc`.
- [x] Add a reproducible GX artifact-production path for the missing full-field
  reference. `scripts/prepare_gx_mode_structure_run.py` copies and patches the
  bundled GX s-alpha Cyclone input, forces `[Diagnostics]` `omega=true`,
  `fields=true`, `moments=true`, and `nwrite_big=1000`, and writes
  `fixtures/gx_cyclone_mode_structure_run/` with the prepared input, metadata,
  and exact external GX/export/gate command sequence.
- [ ] Obtain/export a true external per-`ky` complex mode-structure fixture for
  the problematic branch, especially `ky=0.3`, using GX `.big.nc`, GKW,
  Gyaradax, GS2, stella, or a local diagnostic patch. The fixture should store
  at least `ky`, `z`, complex `phi(z)`, growth rate, real frequency,
  normalization convention, and enough metadata to identify geometry,
  resolution, field-line length, and time/growth window. For the GX route, the
  remaining action is to run the prepared external GX command and feed the
  resulting `.big.nc` through the exporter/gate.
- [x] If no reliable external mode-structure fixture is available, assemble a
  minimal GX-style Hermite-Laguerre linear moment RHS for the s-alpha
  adiabatic-electron ITG case. Reuse the source-matched linked `k_z`
  hypercollision contribution and keep moment layout, linked-boundary chains,
  normalization, and field solve conventions explicit. This is now implemented
  through `GxMomentRHSParams`, `gx_moment_adiabatic_phi`,
  `gx_moment_linear_rhs`, `run_gx_salpha_moment_rhs_mode_structure_fixture`,
  and `examples/export_gx_moment_rhs_fixture.py`. The reduced RHS includes
  linked Hermite streaming, an adiabatic-electron moment field solve, projected
  density/temperature-gradient drive, a simple s-alpha curvature-drift
  surrogate, and GX linked `k_z` hypercollision. It is a discriminator fixture
  producer, not a full GX replacement. The same path is now available directly
  from `examples/run_cyclone_mode_structure_gate.py --reference-moment-rhs`.
- [x] Add a mode-structure comparison gate that reports scalar growth error,
  real-frequency error, row-normalized complex `phi(z)` error, phase-aligned
  state/moment error if available, and pass/fail status separately. The generic
  gate is implemented and wired to the scan-gate `profile_error` metric; what
  remains is feeding it with an exact external multi-`ky` fixture or a new
  moment-RHS scan output.
- [ ] Re-run the exact GX input-control scan after the external artifact exists:
  `n_z_total=96`, Hermite/Laguerre control resolution `48/16`, `nperiod=2`,
  `ky=(0.3,0.5)`, and the calibrated GX reference curve. The implementation
  path is closed; this remaining checkbox is a reference-data/validation run
  against the real GX `.big.nc`, not missing solver infrastructure.

### 2. Build the first single-command stellarator run

- [x] Add or harden an example script that runs a linear stellarator scan from
  either a DESC equilibrium/path or the existing DSHAPE fixture. Inputs should
  include `rho`, `alpha`, field-line length/periods, `ky` grid, resolution,
  species/profile gradients, derivative backend, time-step/window controls,
  and output directory. This is implemented as
  `examples/run_stellarator_linear_scan.py`. The default path uses the bundled
  DESC DSHAPE `.npz` fixture; `--geometry-source desc-path --desc-path ...`
  evaluates a real DESC equilibrium on a chosen Boozer field-line grid.
- [x] The example should write machine-readable outputs:
  geometry audit JSON/CSV, `ky` growth/frequency table, selected complex
  `phi(z)` mode structures, convergence metadata, and a quasilinear proxy. The
  current artifact set is `geometry_audit.json`, `geometry_audit.csv`,
  `ky_growth.csv`, `mode_structures.csv`, `convergence_history.csv`,
  `convergence_metadata.json`, `quasilinear_proxy.json`, and
  `run_config.json`.
- [x] The script should fail early if the imported geometry does not pass the
  internal geometry-contract checks or the available eik parity checks. The
  default preflight checks finite fields, positive \(B\), positive metric
  diagonals, representative nonnegative \(k_\perp^2\), and the solver-to-GX/eik
  export contract; a DESC-path run can additionally require a matched external
  `eik.out` fixture with `--external-eik-reference`.

### 3. Harden stellarator geometry and boundary contracts

- [ ] Promote the DESC/eik geometry preflight to the standard path for
  stellarator simulations: compare \(B\), \(\nabla_\parallel\), metric
  elements, magnetic-drift coefficients, Jacobian/weights where available, and
  representative \(k_\perp^2\) values before solving. The new
  `run_stellarator_linear_scan.py` command already does this for its reduced
  scan path; the remaining work is to make the same preflight a shared library
  entry point for future production runners and benchmark fixtures.
- [ ] Add the still-open finite-difference check against DESC/SIMSOPT geometry
  quantities for a small fixture, or document why the present DESC API makes a
  different independent check preferable.
- [ ] Validate stellarator field-line boundary behavior and mode-chain
  connectivity on a non-axisymmetric fixture: linked/twist-and-shift shifts,
  open-chain ends, zonal handling, and reproducibility under field-line-length
  changes.

### 4. Add a real stellarator benchmark fixture

- [ ] Select one small stellarator ITG or TEM-style benchmark from the local
  DESC/GX/GIST/GS2-compatible materials or generate one with an external code.
  The first fixture can be linear electrostatic and adiabatic-electron if it
  has reliable growth/frequency/mode-structure references.
- [ ] Store the benchmark metadata with the fixture: equilibrium source,
  geometry producer, field-line label, radial coordinate, normalization,
  species/profile parameters, resolution, boundary condition, and reference
  diagnostic windows.
- [ ] Add tests that run a reduced version of the benchmark and verify finite
  outputs plus tolerance-gated geometry and mode-structure diagnostics. Keep
  production-resolution checks as examples or opt-in tests if they are too
  expensive for the default suite.

### 5. Convergence, timing, and optimization readiness

- [ ] Run convergence studies for the trusted stellarator case in `n_z`,
  velocity resolution, `kx/ky` grid, field-line length, time step, and
  growth-window choice. Record the tolerance ladder in `STATUS.md` and
  `main.tex`.
- [ ] Re-run CPU memory and timing estimates at the validated production
  controls and record whether the `task.tex` target, roughly minutes on
  \(\mathcal{O}(100)\) CPUs, is credible.
- [ ] Upgrade the reduced DESC optimization demo only after the above gates
  pass: use the validated stellarator fixture, fixed topology, documented
  geometry sensitivities, and finite-difference checks of AD gradients with
  respect to profile and continuous geometry inputs.

Deferred beyond the first trusted stellarator run:

- [ ] kinetic-electron TEM production validation,
- [ ] collisions and electromagnetic perturbations,
- [ ] nonlinear ExB turbulence and nonlinear dealiasing,
- [ ] full DESC shape-optimization campaigns with topology/remeshing changes.
