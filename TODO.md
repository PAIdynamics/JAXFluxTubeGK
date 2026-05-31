# TODO: Differentiable Flux-Tube Stellarator Gyrokinetic Solver

Last planned: 2026-05-31

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
- [ ] Add Cyclone Base Case linear ITG growth-rate test.
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
- [ ] Extend Gyaradax/GKW trace comparison to production-control resolution.
- [ ] Add a GKW `time.dat`/diagnostic trace exporter or loader for the same `CycloneTrace` schema.
- [x] Add a GX DESC-block `eik.out` loader and compare solver-produced DESC/GX-convention geometry against the matched DSHAPE external-format fixture.
- [x] Add a reduced stellarator fixture:
  - fixed surface,
  - fixed `alpha`,
  - small `ky` grid,
  - reference geometry arrays.
- [x] Compare stellarator geometry quantities against precomputed reference data.
- [ ] Compare solver-computed ITG growth-rate scans against available GS2/GX/GKW-style references.
- [x] Compare solver-produced geometry arrays against local GX/GS2/GX-DESC-style `eik` outputs where fixtures are available.
- [ ] Add a truly independently generated DESC/GX eik fixture from an external runner when a compatible GX/DESC script or GS2/stella export path is available.
- [ ] Add convergence tests over:
  - [x] `N_s`,
  - [x] `N_vparallel`,
  - [x] `N_mu`,
  - [x] `ky`,
  - [x] timestep.
- [x] Record benchmark commands and tolerances in `STATUS.md` and, later, in docs.

Phase 10 baseline note: the current validation tranche covers reduced deterministic fixtures, manufactured convergence over parallel/velocity/`ky`/time resolution, a GX Cyclone input-contract fixture, named RH/CBC scalar targets, GX NetCDF growth-curve loading, GX/GS2 eik-table loading, executable validation gates, late-window growth fitting, a passing true RH late-plateau metric, in-residual GKW/Gyaradax-scaled `disp_par` recurrence control, direct GKW fourth-difference `disp_vp` velocity recurrence control for the RH fallback path, a GKW-aligned Cyclone selected-`ky` setup with a production-control memory-light gate, GKW finite-difference Cyclone fallback controls, the GKW/Gyaradax sign-dependent upwind parallel fallback for CBC Term I/Term VII, a passing CBC term-level parity audit, a reduced CBC trace CSV/API, a reduced Gyaradax physical trace comparison with normalization-equivalent phi/state/RHS norm fields, spectral modal damping hooks for experiments only, solver-to-eik field parity reports, a DESC fixture eik-export contract gate, a matched DESC/GX block-`eik.out` geometry parity gate, corrected GIST drift-column handling, a three-fixture GX/VMEC GIST external eik-suite gate, and a validation-gate plotting example used in `main.tex`. The RH plateau, CBC term audit, reduced Gyaradax physical trace comparison, GX/eik imported metric, field-parity gate, DESC eik-export gate, DESC/GX block-eik gate, and GX/GIST suite pass their current contracts. The reduced RH endpoint still runs as a deliberately short smoke gate. The production-control CBC selected-`ky` gate now uses the GKW finite-difference velocity and zero-boundary GKW-upwind parallel fallback path and observes \(\gamma=0.164715\) for the medium documented run against the \(0.179\) GKW/Gyaradax target, but it remains open against the \(0.01\) tolerance. The term audit reports zero stored error in drift, drive, drift-field, boundary-map, grid-normalization, and RHS assembly conventions; the reduced Gyaradax comparison reports max physical trace error \(1.24\times10^{-2}\) across time, physical amplitude, window growth, fitted growth, physical phi norm, physical state norm, and physical RHS norm at tolerance \(2\times10^{-2}\). Raw amplitudes and raw log-normalization remain convention-dependent because the two codes store per-window normalization differently. The next Cyclone work should extend the trace comparison to production-control resolution or add a direct GKW diagnostic export.

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

## Immediate Next Round

Use the new validation-hardening tools to close the remaining production gaps:

- compare the Cyclone selected-`ky` time evolution directly against a Gyaradax/GKW trace at matched reduced and production-control resolution, focusing on phi solve history, RK4/window normalization, initialization amplitude/envelope, and late-window growth diagnostic conventions;
- extend the reduced Gyaradax physical trace comparison to production-control resolution and decide whether remaining differences are transient, timestep/window, or normalization-convention effects;
- export an equivalent GKW diagnostic trace and load it into the same `CycloneTrace` schema;
- add a production-resolution Cyclone selected-`ky` regression once the observed growth is within the GKW/GX tolerance ladder;
- replace the matched DESC/GX block-eik fixture with or supplement it by a truly independent external eik producer when a compatible GX/DESC, GS2, stella, or VMEC/GIST path is available;
- keep DESC-driven optimization examples labeled reduced until the CBC selected-`ky` gate passes alongside the RH plateau and eik parity gates.
