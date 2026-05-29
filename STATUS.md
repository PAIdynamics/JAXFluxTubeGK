# STATUS

Last updated: 2026-05-29

## Current State

Planning pass complete, the first physical-model/numerics specification has
been drafted, and the first Phase 4 Boozer/stellarator flux-tube geometry
adapter is implemented and tested. GX has now been added as an algorithmic
and benchmark reference for future velocity-space, geometry, nonlinear,
closure, and diagnostic extensions. Phases 5, 5A, 6, 7, 8, and 9 are implemented
and tested through quasineutrality, diagnostics, the self-consistent linear RHS
residual, fixed-step RK4 integration, linear growth-rate extraction, matrix-free
operator wrappers, and differentiable objective helpers. The first Phase 10
validation tranche is implemented with reduced parity, stellarator fixture, and
manufactured convergence tests. A second Phase 10 validation tranche now adds
velocity-space and `ky` convergence checks plus a GX Cyclone input-contract
fixture mapped into the solver's public grid, geometry, species, and
Hermite-Laguerre interfaces. Phase 11 CPU performance and differentiability
hardening is implemented with endpoint-only RK4 integration, a public jitted
linear residual, memory/profiling helpers, performance smoke tests, and
static-vs-differentiable documentation.

The repository currently contains:

- `task.tex`: project thesis description and six-month roadmap.
- `main.tex`: physical model and numerical scheme specification for the first solver implementation.
- `TODO.md`: project implementation plan.
- `STATUS.md`: this progress ledger.
- `docs/performance_and_differentiability.md`: Phase 11 CPU scaling, memory, and AD/topology notes.
- `pyproject.toml`: root Python package metadata for the `stellarator_gk` package.
- `uv.lock`: resolved project dependency lock file.
- `src/stellarator_gk/`: Phase 2 core types/grids, Phase 3 analytic geometry, Phase 4 flux-tube geometry adapters, the public linear residual wrapper, Phase 8 fixed-step time advancement, and Phase 9 objective/operator interfaces.
- `src/stellarator_gk/geometry/`: circular/\(s\)-alpha analytic geometry plus Boozer/precomputed flux-tube geometry scaffolding.
- `src/stellarator_gk/physics/`: Phase 5 Bessel/FLR, Maxwellian, drive, drift, mirror, streaming primitives, Phase 5A Hermite-Laguerre velocity-moment utilities, Phase 6 quasineutrality solvers, and Phase 7 linear RHS terms.
- `src/stellarator_gk/diagnostics.py`: Phase 6 diagnostic reductions, spectra, and quasilinear flux ingredients.
- `src/stellarator_gk/operators.py`: Phase 9 matrix-free residual actions, mode-chain projection helpers, dense reduced-operator construction, and tiny eigensystem helpers.
- `src/stellarator_gk/objectives.py`: Phase 9 growth-rate, selected-mode, quasilinear-proxy, mode-structure, and short initial-value objective helpers.
- `src/stellarator_gk/performance.py`: Phase 11 reduced-grid profiler, memory estimators, PyTree byte accounting, and byte-format helpers.
- `src/stellarator_gk/time_advance.py`: Phase 8 RK4 stepping, fixed-step scan integration, CFL estimate, per-`ky` normalization, and growth/frequency diagnostics.
- `tests/`: Phase 2 through Phase 11 unit, validation, performance-smoke, and differentiability tests.
- `papers/`: Gyaradax paper sources, stellarator microstability/optimization papers, and GKW paper materials.
- `papers/gkw/`: GKW reference PDF, rebuilt extracted TeX, GKW manual PDF, and related paper material.
- `papers/gx-paper/`: GX paper source for the Fourier-Laguerre-Hermite flux-tube formulation and benchmark discussion.
- `relevant-codes/`: Gyaradax, DESC, GKW, and GX source trees.
- `relevant-codes/gkw/`: original GKW Fortran simulation source, sample inputs, and helper scripts.
- `relevant-codes/gx/`: GX source tree, docs, input files, and tests for method/benchmark reference.

## Inputs Reviewed

- Read `task.tex`.
- Read Gyaradax README, notes, paper source, core modules, and test files.
- Read stellarator optimization/microstability paper TeX sources in `papers/arXiv-2301.09356v2` and `papers/arXiv-2310.18842v2`.
- Read DESC README for equilibrium/optimization context.
- Attempted PDF text extraction for the GKW PDFs, but `pdftotext` is not installed in this environment.
- Built and checked `papers/gkw/GKW_rebuilt.tex` against `papers/gkw/GKW.pdf` as a non-embedded paper reconstruction.
- Inventoried `relevant-codes/gkw/README`, `relevant-codes/gkw/src/`, and `relevant-codes/gkw/samples/` as the direct GKW implementation reference.
- Read `papers/gx-paper/main.tex` for GX's local flux-tube model, Fourier-Laguerre-Hermite pseudo-spectral velocity formulation, geometry discussion, closures, nonlinear numerics, and benchmark strategy.
- Read `relevant-codes/gx/README.md`, `docs/Numerics.rst`, `docs/Geometry.rst`, `docs/Inputs.rst`, `docs/Nonlinear.rst`, and `docs/Citing.rst`.
- Skimmed GX source contracts in `include/grids.h`, `geometry.h`, `grad_parallel.h`, `moments.h`, `laguerre_transform.h`, and `closures.h`.
- Inspected GX unit-test/input structure, noting that GX's own unit-test README says many tests are old and need updating before being treated as strict oracles.

## Reference Sources

The project reference hierarchy is now:

- Primary paper specification: `papers/gkw/GKW.pdf`.
- Editable paper reconstruction: `papers/gkw/GKW_rebuilt.tex`, which builds to `papers/gkw/GKW_rebuilt.pdf` without embedding the reference PDF. Use it for quick searching/editing of the GKW paper text, while treating `GKW.pdf` as the visual/reference source.
- Direct implementation reference: `relevant-codes/gkw/src/`, the original Fortran GKW simulation source.
- GKW sample/reference cases: `relevant-codes/gkw/samples/`, including `cyclone`, `simple_example`, `simple_itg`, `STD`, and `STD_kinetic`.
- Modern differentiable implementation reference: `relevant-codes/gyaradax/`, especially its GKW source mapping, JAX solver implementation, and tests.
- GX method and benchmark reference: `papers/gx-paper/` and `relevant-codes/gx/`, especially the Fourier-Laguerre-Hermite velocity-space formulation, moment layout, linked/twist-and-shift parallel derivative machinery, geometry array contract, closure models, diagnostics, nonlinear/dealiasing strategy, and example input files.
- Stellarator geometry/optimization reference: `relevant-codes/DESC/` plus the stellarator optimization papers in `papers/`.

For implementation work, use the GKW source modules as the authoritative source for legacy conventions and term-level behavior:

- `src/linart.f90`: main program.
- `src/normalise.F90`: normalization.
- `src/geom.F90`: geometry and metric quantities.
- `src/mode.F90`: spectral mode setup and flux-tube mode connectivity.
- `src/grid.F90` and `src/velocitygrid.F90`: grid definitions.
- `src/components.F90`: species setup.
- `src/linear_terms.F90`: linear gyrokinetic RHS terms and Poisson-related pieces.
- `src/non_linear_terms.F90`: nonlinear ExB terms and FFT-based operations.
- `src/exp_integration.F90`: explicit integration and field update flow.
- `src/matdat.F90`: matrix/timestep preparation, including timestep estimates.
- `src/diagnostic.F90`: growth rates, fluxes, spectra, and output conventions.
- `src/collisionop.F90`, `src/rotation.F90`, and electromagnetic/collision-related modules: future physics extensions after the linear electrostatic baseline.

## Decisions

- Use Gyaradax/GKW physics, normalization, term, sign, diagnostic, and benchmark conventions as the baseline.
- Use the `task.tex` discretization target: Fourier modes in perpendicular directions, spectral operators along the magnetic-field coordinate, and spectral operators in velocity space.
- Keep GKW/Gyaradax finite-difference stencils as an optional fallback/parity backend, not as the default numerical target.
- Use GX as an algorithmic reference, not as the exact architecture target: GX is GPU-native and nonlinear-first, while this project is JAX-first, differentiable, CPU-oriented, and linear electrostatic first.
- Keep the present Chebyshev-collocation velocity grids as the first implemented spectral backend, while planning a GX-style Hermite-Laguerre moment backend as a later extension.
- Reuse GX ideas where they fit the design: flux-tube geometry quantities, linked mode chains, Fourier pseudo-spectral nonlinear/dealiasing strategy, Hermite/Laguerre moments, moment closures/hypercollisions, and spectra/benchmark conventions.
- Document physical model and numerical scheme in `main.tex` before coding the corresponding solver components.
- Begin with linear electrostatic, collisionless, adiabatic-electron flux-tube gyrokinetics.
- Keep nonlinear ExB, kinetic electrons, collisions, and electromagnetic effects as later extensions.
- Keep differentiable continuous geometry separate from non-differentiable integer topology and file I/O.
- Require tests for every new function added under `src/`.
- Update this file during every implementation round.

## Next Implementation Round

Goal: begin Phase 12 optimization integration:

- define differentiable geometry/profile input knobs for optimization,
- implement a single-surface/single-alpha objective wrapper using the Phase 9/11 no-history path,
- add a small toy `jax.value_and_grad` optimization example,
- keep DESC equilibrium coupling as the next step after the toy objective is stable.

Expected file changes:

- optimization-facing helper module or additions to `objectives.py`,
- focused tests for single-surface/single-alpha objective values and gradients,
- example or docs snippet for the toy optimization path,
- `TODO.md`,
- `STATUS.md`

Expected tests:

- objective shape/value smoke tests,
- `jax.grad`/`jax.value_and_grad` finite-gradient checks,
- finite-difference agreement on selected optimization knobs,
- no-history integration parity against the default history path on reduced grids.

## Round Log

### 2026-05-29: Implemented Phase 11 CPU Performance and Differentiability Hardening

- Added `src/stellarator_gk/performance.py` with:
  - `LinearMemoryEstimate`,
  - `LinearResidualBenchmark`,
  - `pytree_nbytes`,
  - `estimate_linear_memory_from_dimensions`,
  - `estimate_linear_memory_from_precompute`,
  - `benchmark_linear_residual`,
  - `format_bytes`.
- Added `jitted_linear_residual` as the public fixed-topology JIT residual entry point.
- Extended `integrate_fixed_step` with `store_history=False`:
  - default path still uses `jax.lax.scan` and stores all snapshots,
  - endpoint-only path uses `jax.lax.fori_loop` and stores initial/final states only.
- Added `docs/performance_and_differentiability.md` documenting:
  - CPU execution strategy,
  - memory scaling and endpoint-history savings,
  - qualitative GX comparison,
  - differentiable continuous quantities versus static topology/file-I/O.
- Added `tests/test_performance_hardening.py` covering:
  - dimension-only target memory estimates,
  - assembled-precompute byte accounting,
  - eager versus jitted residual parity,
  - reduced-grid residual benchmark smoke timing,
  - finite/stable gradients through a jitted no-history objective path.
- Extended `tests/test_time_advance.py` with endpoint-only RK4 parity.
- Updated `TODO.md` to mark Phase 11 complete and set Phase 12 as the next implementation round.
- Commands run:
  - `uv run --extra dev python -m pytest tests/test_performance_hardening.py tests/test_time_advance.py`
  - `uv run --extra dev ruff check src tests`
  - `uv run --extra dev python -m pytest`
- Verification:
  - `10 passed` for the Phase 11 targeted tests.
  - `ruff check src tests` passed.
  - `95 passed` for the full pytest suite.

### 2026-05-29: Extended Phase 10 Benchmark and Convergence Tests

- Committed the first Phase 10 baseline validation tranche:
  - commit `6fd5db0` (`Add baseline benchmark validation tests`).
- Extended `tests/test_benchmark_validation.py` with:
  - Chebyshev `v_parallel` derivative convergence from `N_vparallel=8` to `N_vparallel=16`,
  - Chebyshev `mu` derivative convergence from `N_mu=8` to `N_mu=16`,
  - a manufactured `ky` growth-scan convergence check from `N_ky=9` to `N_ky=65`,
  - a GX Cyclone s-alpha input fixture parsed from `relevant-codes/gx/benchmarks/linear/ITG_cyclone/itg_salpha_adiabatic_electrons.in`.
- The GX fixture check maps the local TOML input into:
  - `FourierGridSpec`/`build_fourier_grid`,
  - `VelocityBasisSpec`/`build_hermite_laguerre_basis`,
  - `GeometryScalarParams`/`build_s_alpha_geometry`,
  - `SpeciesParams`.
- Updated `TODO.md`:
  - marked the stale coupled phi/RHS parity checkbox complete,
  - marked the first GX input fixture complete,
  - marked convergence over `N_vparallel`, `N_mu`, and `ky` complete.
- Commands run:
  - `uv run --extra dev python -m pytest tests/test_benchmark_validation.py`
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`
- Verification:
  - `9 passed` for `tests/test_benchmark_validation.py`.
  - `90 passed` for the full pytest suite.
  - `ruff check src tests` passed.

### 2026-05-29: Implemented Phase 10 Baseline Benchmarks and Validation

- Added `tests/test_benchmark_validation.py`.
- Implemented reduced Gyaradax/GKW analytic geometry validation for both circular and s-alpha models:
  - `B`,
  - `F`,
  - `G`,
  - `E_y`,
  - perpendicular metric components,
  - magnetic drift coefficients.
- Added a direct reduced phi/RHS parity fixture:
  - builds the coupled Phase 7 precompute,
  - solves adiabatic quasineutrality,
  - checks the field residual to `2e-12`,
  - compares `linear_residual` against an explicit GKW/Gyaradax-term formula to `3e-12`.
- Added a reduced zonal-flow invariant:
  - flat Boozer flux tube,
  - `kx=ky=0`,
  - constant distribution,
  - self-consistent residual remains zero to `3e-12`.
- Added a reduced stellarator fixture:
  - fixed Boozer surface,
  - fixed field-line label `alpha`,
  - small `ky` grid,
  - precomputed physical arrays mapped to internal geometry,
  - drift, metric, `B`, source label, and nonnegative `k_perp^2` checked.
- Added convergence validation:
  - periodic parallel spectral derivative convergence from `N_s=12` to `N_s=24`,
  - fixed-step RK4 growth-rate convergence from 10 to 20 steps.
- Updated `TODO.md` to mark the completed reduced Phase 10 baseline tasks and leave full Rosenbluth-Hinton, Cyclone, GX/eik, velocity-resolution, and `ky` scans open.
- Commands run:
  - `uv run --extra dev python -m pytest tests/test_benchmark_validation.py`
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`
- Verification:
  - `6 passed` for `tests/test_benchmark_validation.py`,
  - `87 passed` for the full pytest suite,
  - `ruff check src tests` passed.

### 2026-05-29: Implemented Phase 9 Eigenvalue and Objective Interfaces

- Added `src/stellarator_gk/operators.py`.
- Implemented matrix-free/eigensolver helpers:
  - `mode_chain_mask`,
  - `project_to_ky`,
  - `project_to_mode_chain`,
  - `linear_operator_action`,
  - `flatten_state`,
  - `unflatten_state`,
  - `dense_matrix_from_action`,
  - `dense_linear_operator_matrix`,
  - `dense_eigensystem`.
- Added `src/stellarator_gk/objectives.py`.
- Implemented differentiable objective containers and helpers:
  - `LinearObjectiveValues`,
  - `max_growth_objective`,
  - `selected_growth_objective`,
  - `weighted_quasilinear_proxy`,
  - `kperp2_weighted_average`,
  - `mode_structure_penalty`,
  - `linear_growth_objectives`,
  - `initial_value_growth_objectives`,
  - `solve_field_from_state`.
- Exported Phase 9 APIs through the top-level `stellarator_gk` package.
- Added `tests/test_objectives_operators.py` covering:
  - mode-chain and one-`ky` projection helpers,
  - dense matrix reconstruction and tiny eigensystem helpers,
  - restricted matrix-free residual actions,
  - objective shapes, values, and penalties on manufactured mode histories,
  - short reduced-grid initial-value objective gradients with respect to `R/L_n`, `R/L_T`, `q`, `shat`, and continuous geometry scaling,
  - finite-difference agreement for representative objective gradients.
- Committed completed Phase 7/8 work before starting Phase 9:
  - `a324343 Implement linear RHS and RK4 time advance`.
- Updated `TODO.md` to mark Phase 9 complete and set Phase 10 as the next project phase.
- Commands run:
  - `git add ...`
  - `git commit -m "Implement linear RHS and RK4 time advance"`
  - `uv run --extra dev python -m pytest tests/test_objectives_operators.py`
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`
- Verification:
  - `4 passed` for `tests/test_objectives_operators.py`,
  - `81 passed` for the full pytest suite,
  - `ruff check src tests` passed.

### 2026-05-29: Implemented Phase 8 Time Advancement and Growth Rates

- Added `src/stellarator_gk/time_advance.py`.
- Implemented Phase 8 time-advance containers:
  - `TimeAdvanceResult`,
  - `LinearGrowthDiagnostics`,
  - `KyNormalizationResult`.
- Implemented fixed-step time advancement:
  - `rk4_step`,
  - `integrate_fixed_step` using `jax.lax.scan`,
  - explicit post-step `filter_fn` hook for later pseudo-spectral filtering/dealiasing.
- Implemented linear diagnostics:
  - `mode_chain_amplitude` using the connected `kx` chain containing `kx=0`,
  - `growth_rate`,
  - `real_frequency` with the GKW/Gyaradax sign convention from `main.tex`,
  - `linear_growth_diagnostics` returning amplitudes, growth rates, frequencies, and normalized mode structures.
- Implemented per-`ky` amplitude normalization:
  - `normalize_by_ky_amplitude`,
  - accumulated logarithmic normalization factors for diagnostic bookkeeping.
- Implemented `estimate_linear_cfl_dt`, a conservative row-sum/RK4-radius estimate using Phase 7 RHS precompute coefficients.
- Exported Phase 8 APIs through the top-level `stellarator_gk` package.
- Added `tests/test_time_advance.py` covering:
  - zero-input RK4 invariance,
  - fixed-step history and times,
  - fourth-order RK4 convergence on a complex scalar ODE,
  - mode-chain amplitude, growth-rate, frequency, and normalization recovery,
  - JIT compatibility and reverse-mode gradients through a short fixed-step solve,
  - CFL estimate formula behavior.
- Updated `TODO.md` to mark Phase 8 complete and set Phase 9 as the next project phase.
- Commands run:
  - `uv run --extra dev python -m pytest tests/test_time_advance.py`
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`
- Verification:
  - `5 passed` for `tests/test_time_advance.py`,
  - `77 passed` for the full pytest suite,
  - `ruff check src tests` passed.

### 2026-05-29: Implemented Phase 7 Linear RHS Residual

- Added `src/stellarator_gk/physics/rhs_terms.py`.
- Implemented `LinearRHSPrecompute`, combining:
  - spectral derivative matrices,
  - Fourier `ky`,
  - geometry fields,
  - species FLR factors,
  - Maxwellian and thermodynamic drive factors,
  - parallel streaming, mirror, and magnetic-drift coefficients,
  - charge-over-temperature factors,
  - optional perpendicular damping with zero default.
- Implemented isolated RHS terms:
  - `parallel_streaming`,
  - `magnetic_drift_advection`,
  - `mirror_force`,
  - `equilibrium_drive`,
  - `parallel_field_drive`,
  - `drift_field_drive`,
  - `dissipation`.
- Added `linear_residual_from_phi` for supplied-field residual assembly.
- Added a finite zero-mode JVP for the nonnegative square root used in FLR Bessel arguments, avoiding `sqrt(0)` AD NaNs for exact zero Fourier/Larmor-radius modes.
- Added `src/stellarator_gk/solver.py` with:
  - `LinearResidualPrecompute`,
  - `build_linear_residual_precompute`,
  - public `linear_residual` supporting both explicit `phi` through `LinearRHSPrecompute` and self-consistent adiabatic/kinetic phi solves through `LinearResidualPrecompute`.
- Exported Phase 7 APIs through `stellarator_gk.physics` and the top-level `stellarator_gk` package.
- Added `tests/test_linear_rhs.py` covering:
  - precompute shapes,
  - zero-input behavior for every term and the assembled residual,
  - manufactured spectral derivative checks for streaming and mirror terms,
  - magnetic drift and field-drive formula checks,
  - full self-consistent residual linearity,
  - `jax.jit` compatibility,
  - reverse-mode gradient versus finite difference,
  - geometry-array and species-parameter gradients through RHS precomputation,
  - multi-species residual shape and finite-value behavior.
- Updated `TODO.md` to mark Phase 7 complete and set Phase 8 as the next project phase.
- Commands run:
  - `uv run --extra dev python -m pytest tests/test_linear_rhs.py`
  - `uv run --extra dev python -m pytest tests/test_linear_rhs.py tests/test_physics_primitives.py`
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`
- Verification:
  - `7 passed` for `tests/test_linear_rhs.py`,
  - `17 passed` for `tests/test_linear_rhs.py tests/test_physics_primitives.py`,
  - `72 passed` for the full pytest suite,
  - `ruff check src tests` passed.

### 2026-05-29: Implemented Phase 6 Quasineutrality and Diagnostics

- Added `src/stellarator_gk/physics/quasineutrality.py`.
- Implemented adiabatic-electron electrostatic quasineutrality:
  - `AdiabaticElectronParams`,
  - `AdiabaticQuasineutralityPrecompute`,
  - default electron-density choice from background ion charge neutrality,
  - precomputed velocity/quasineutrality weights,
  - density numerator reduction,
  - local phi solve,
  - explicit ky=0 zonal adiabatic correction with kx=0 gauge left on the local path,
  - residual evaluators for solved/trial fields.
- Implemented fully kinetic electrostatic quasineutrality:
  - `KineticQuasineutralityPrecompute`,
  - kinetic density numerator,
  - kinetic phi solve,
  - explicit constant-mode gauge regularization,
  - kinetic residual evaluators.
- Added `src/stellarator_gk/diagnostics.py` with:
  - velocity-space integrals,
  - mode amplitudes,
  - `kxky` and `ky` spectra,
  - radial flux spectrum and total radial flux quasilinear ingredients.
- Exported Phase 6 APIs through `stellarator_gk.physics` and the top-level package.
- Added `tests/test_quasineutrality_diagnostics.py` covering:
  - default adiabatic electron response,
  - zero-distribution phi,
  - local adiabatic formula without zonal correction,
  - flux-surface-corrected zonal equation residuals,
  - multi-species adiabatic solve,
  - kinetic solve and constant-mode regularization,
  - JIT and AD finite-difference checks,
  - diagnostic integral/spectrum/flux normalization,
  - input validation.
- Updated `TODO.md` to mark Phase 6 complete and set Phase 7 as the next round.
- Commands run:
  - `uv run --extra dev python -m pytest tests/test_quasineutrality_diagnostics.py`
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`
- Verification:
  - `9 passed` for `tests/test_quasineutrality_diagnostics.py`,
  - `65 passed` for the full pytest suite,
  - `ruff check src tests` passed.

### 2026-05-29: Implemented Phase 5A GX-Informed Hermite-Laguerre Velocity Backend

- Added `src/stellarator_gk/physics/velocity_moments.py`.
- Implemented GX-convention velocity basis pieces:
  - `VelocityBasisKind` and `VelocityBasisSpec`,
  - `HermiteLaguerreBasis` PyTree container,
  - Gauss-Hermite and Gauss-Laguerre grids/weights,
  - probabilists' Hermite basis `He_m/sqrt(m!)`,
  - signed Laguerre basis `(-1)^l L_l`,
  - moment-to-grid and grid-to-moment transforms,
  - Hermite derivative, `v_parallel`, and `v_parallel^2` coupling matrices,
  - Laguerre `mu B` multiplication matrix.
- Added GX gyroaverage utilities:
  - `gyroaverage_laguerre_coefficients`,
  - `truncated_gamma0_from_laguerre`,
  - `gamma0_limit_error`.
- Added low-order moment diagnostics:
  - density,
  - parallel flow,
  - parallel/perpendicular temperature,
  - parallel/perpendicular heat-flux-like moments,
  - fluid-moment dictionary helper,
  - free-energy spectra.
- Added closure hooks:
  - modal hypercollision damping rates,
  - modal hypercollision RHS application,
  - truncation-only baseline represented by no extra closure RHS.
- Exported Phase 5A utilities through `stellarator_gk.physics` and the top-level `stellarator_gk` package.
- Added `tests/test_hermite_laguerre_basis.py` covering:
  - PyTree/static spec behavior,
  - Hermite/Laguerre polynomial conventions,
  - transform orthonormality,
  - spectral/grid round trips,
  - modal coupling matrices against projected grid multiplication,
  - low-order moment diagnostics against quadrature,
  - GX gyroaverage coefficients against quadrature,
  - truncated Laguerre sum convergence to `Gamma_0`,
  - gyroaveraged moment diagnostics,
  - hypercollision damping hooks,
  - JIT and AD gradient smoke tests.
- Updated `TODO.md` to mark the implemented Phase 5A foundation complete while leaving production Beer/Smith/Hammett closures as future work.
- Commands run:
  - `uv run --extra dev python -m pytest tests/test_hermite_laguerre_basis.py`
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`
- Verification:
  - `10 passed` for `tests/test_hermite_laguerre_basis.py`,
  - `56 passed` for the full pytest suite,
  - `ruff check src tests` passed.
- Scope note:
  - This is a tested Hermite-Laguerre velocity backend foundation. It does not yet implement the full GX moment-space gyrokinetic RHS, field equations in moment variables, or production Beer/Smith/Hammett closures.

### 2026-05-29: Implemented Phase 5 Physics Primitives

- Added `src/stellarator_gk/physics/` with backend-neutral primitives:
  - stable differentiable `bessel_j0`,
  - `gamma0(b) = I_0(b) exp(-b)` using the scaled Bessel function,
  - normalized energy,
  - Maxwellian,
  - thermodynamic drive factor,
  - equilibrium-gradient drive coefficient,
  - species FLR/Bessel/polarization factors,
  - magnetic drift frequency,
  - mirror-force coefficient,
  - parallel-streaming coefficient.
- Added `FLRFactors` as a JAX PyTree data container.
- Exported the new physics primitives through `stellarator_gk.physics` and the top-level `stellarator_gk` package.
- Added validation that `SpeciesParams.charge` is nonzero.
- Added `tests/test_physics_primitives.py` covering:
  - Bessel and \(\Gamma_0\) values and small-argument limits,
  - Bessel AD gradient against \(-J_1\),
  - one-species and multi-species broadcasting,
  - zero-\(k_\perp\) FLR limits,
  - gradient, drift, mirror, and streaming coefficient formulas,
  - JIT compatibility and AD gradients versus finite differences.
- Updated `TODO.md` to mark Phase 5 complete and set Phase 6 as the immediate next round.
- Commands run:
  - `uv run --extra dev python -m pytest tests/test_physics_primitives.py`
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`
- Verification:
  - `10 passed` for `tests/test_physics_primitives.py`,
  - `46 passed` for the full pytest suite,
  - `ruff check src tests` passed.

### 2026-05-28: Added GX Reference Comparison and Reuse Plan

- Added GX paper/source references to `TODO.md`, `STATUS.md`, and `task.tex`.
- Compared this project against GX:
  - GX is a GPU-native C++/CUDA nonlinear turbulence code optimized for fast production runs.
  - This project targets differentiable JAX-first linear electrostatic flux-tube physics for CPU-usable optimization loops.
  - The current velocity backend is Chebyshev collocation in `v_parallel` and `mu`; GX's distinctive velocity method is Fourier-Laguerre-Hermite with evolved moment coefficients.
- Recorded what is already GX-aligned: local flux-tube ordering, field-aligned geometry, perpendicular Fourier modes, `v_parallel`/`mu` coordinates, precomputed geometry arrays, linked/twist-and-shift connectivity, and spectral-first operators.
- Added future GX-inspired tasks:
  - Hermite-Laguerre velocity backend,
  - moment diagnostics,
  - closure/hypercollision hooks,
  - nonlinear pseudo-spectral ExB/dealiasing hooks,
  - GX input/benchmark fixture mining,
  - GX/GS2/stella-style `eik` geometry parity checks.
- Commands run:
  - documentation/source inspection only; no tests were run in this documentation update.

### 2026-05-25: Implemented Phase 4 Boozer/Stellarator Flux-Tube Geometry

- Added `BoozerSurface`, `FieldLineSpec`, `BoozerFieldLine`, `PhysicalFluxTubeGeometry`, and `FluxTubeGeometry` data models.
- Implemented Boozer toroidal-angle grids, field-line tracing with \(\alpha=\theta-\iota\phi-\alpha_0\), and simple Boozer Fourier magnetic-field evaluation.
- Added the first supported imported-geometry source: precomputed physical arrays on a sampled field line.
- Implemented the adapter from physical Boozer/GX/GS2-like arrays to the internal solver geometry contract: \(B,F,G,E_y,D_x,D_y,g^{xx},g^{xy},g^{yy}\).
- Chose and documented `rho` as the default radial coordinate convention in `main.tex`, with `psi` and minor-radius-normalized `x` carried as metadata options.
- Updated `TODO.md` to mark the implemented Phase 4 pieces complete and leave DESC/SIMSOPT/VMEC source adapters plus DESC/SIMSOPT finite-difference fixture checks as future source integrations.
- Commands run:
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Verification:
  - `36 passed` in pytest,
  - `ruff check src tests` passed,
  - `main.pdf` built successfully with 15 pages.

### 2026-05-25: Implemented Phase 3 Baseline Analytic Geometry

- Added `src/stellarator_gk/geometry/` with circular and \(s\)-alpha analytic geometry builders.
- Implemented the internal geometry contract from `main.tex`: \(B\), parallel streaming factor \(F\), mirror factor \(G\), ExB coefficient \(E_y\), magnetic drift coefficients \(D_x,D_y\), perpendicular metric coefficients \(g^{xx},g^{xy},g^{yy}\), weights, ExB tensor, drift tensor, and `k_perp_squared`.
- Kept continuous geometry differentiable with respect to `GeometryScalarParams(q, shat, eps)` and kept mode topology in the existing static Phase 2 maps.
- Added direct parity tests against `relevant-codes/gyaradax/gyaradax/geometry.py` for both \(s\)-alpha and circular geometry outputs.
- Updated `TODO.md` to mark Phase 3 complete.
- Commands run:
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`
- Verification:
  - `30 passed` in pytest,
  - `ruff check src tests` passed.

### 2026-05-25: Implemented Phase 2 Core Types, Parameters, and Grids

- Created the root Python package `stellarator_gk` with `pyproject.toml`, `uv.lock`, source package files, tests, and `.gitignore`.
- Implemented frozen dataclass PyTrees for species parameters, solver controls, geometry scalar parameters, grid specs, grid outputs, Fourier grids, mode connectivity, and finite-difference fallback operators.
- Implemented Chebyshev-Lobatto spectral velocity grids, Chebyshev open parallel grids, Fourier periodic parallel grids, Clenshaw-Curtis weights, barycentric derivative matrices, modal transforms, perpendicular Fourier grids, GKW-style shear spacing, static mode labels/connectivity, and GKW-style finite-difference fallback derivative matrices.
- Kept topology construction outside gradient-traced code by building labels/connectivity with NumPy and converting maps to JAX integer/bool arrays.
- Updated `TODO.md` to mark the Phase 2 implementation and related bootstrap tasks complete.
- Commands run:
  - `uv run --extra dev python -m pytest`
  - `uv run --extra dev ruff check src tests`
- Verification:
  - `24 passed` in pytest,
  - `ruff check src tests` passed.

### 2026-05-25: Aligned main.tex With Spectral Discretization Target

- Reviewed `task.tex` and confirmed the intended numerical target is Fourier discretization in the perpendicular directions plus spectral methods along the magnetic-field and velocity-space coordinates.
- Updated `main.tex` so GKW supplies physics conventions, signs, normalization, benchmarks, and a finite-difference fallback/parity path, while the primary implementation target is spectral.
- Added spectral velocity-space collocation/quadrature, spectral parallel operators, modal filters/damping, spectral timestep estimates, derivative backend contracts, and tests for spectral convergence.
- Updated `TODO.md` to make the same spectral-first decision explicit in the project goal, development rules, Phase 1 notes, and Phase 2 implementation tasks.
- Built the document with:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Verification:
  - build completed successfully,
  - `main.pdf` generated with 15 pages,
  - no LaTeX errors or overfull-box warnings remained after the final polish pass.
- Tests run: LaTeX build only. No code tests yet because no solver code has been created.

### 2026-05-25: Drafted main.tex Model and Numerics

- Replaced the root `main.tex` placeholder with a concise implementation specification for the differentiable flux-tube stellarator gyrokinetic solver.
- Covered the reference hierarchy, GKW source crosswalk, normalization, field-aligned/Boozer geometry contract, flux-tube mode connectivity, Maxwellian/FLR factors, linear electrostatic gyrokinetic RHS terms, quasineutrality, discrete grids and derivative stencils, residual/precompute interfaces, RK4 stepping, diagnostics, quasilinear objective, differentiability contract, extensions, implementation tasks, and benchmark ladder.
- Updated `TODO.md` to mark the Phase 1 `main.tex` documentation tasks and the GKW source crosswalk as complete.
- Built the document with:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Verification:
  - build completed successfully,
  - `main.pdf` generated with 14 pages,
  - no LaTeX errors or overfull-box warnings remained after the final polish pass.
- Tests run: LaTeX build only. No code tests yet because no solver code has been created.

### 2026-05-25: Updated TODO With GKW References

- Added `papers/gkw/GKW.pdf`, `papers/gkw/GKW_rebuilt.tex`, `papers/gkw/GKW_manual_0.4-b1.pdf`, `relevant-codes/gkw/src/`, and `relevant-codes/gkw/samples/` to `TODO.md`.
- Added a reference hierarchy and GKW source-file crosswalk to guide future implementation.
- Added TODO tasks to cross-check normalization, mode connectivity, stencils, RHS signs, and benchmark cases against the GKW paper/source.
- Tests run: none. Documentation update only.

### 2026-05-25: Rebuilt GKW Paper TeX

- Goal: build `papers/gkw/GKW_rebuilt.tex` and make its output resemble `papers/gkw/GKW.pdf` without embedding or importing the reference PDF.
- Changed `papers/gkw/GKW_rebuilt.tex` from a one-block extracted-text draft into a page-preserving XeLaTeX reconstruction:
  - one `Verbatim` block per extracted original page,
  - explicit `\newpage` boundaries from extraction form feeds,
  - custom paper size matching the rendered reference first page,
  - small monospaced layout to preserve extracted two-column spacing,
  - removed invalid PDF-extraction control characters,
  - normalized `fi`/`fl` ligatures and two unsupported glyphs.
- Built `papers/gkw/GKW_rebuilt.pdf` with:
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error GKW_rebuilt.tex`
- Verification:
  - build completed successfully,
  - rebuilt output is 23 pages,
  - reference PDF has 23 page markers,
  - source contains no `GKW.pdf`, `pdfpages`, `includepdf`, or `includegraphics` reference,
  - first-page raster render now matches the reference page size and preserves the extracted paper layout.
- Caveat: this is a faithful text/layout reconstruction, not a publisher-perfect clone; it does not reproduce Elsevier logo artwork or all exact typography.

### 2026-05-25: Planning

- Created the initial project backlog in `TODO.md`.
- Created this `STATUS.md` progress ledger.
- Tests run: none. This was a documentation/planning update only.

## Open Risks

- PDF comparison tooling is limited: `pdftotext`, Ghostscript, and Poppler tools are unavailable. Current GKW comparison used the extracted TeX, page counts, source checks, and first-page raster rendering via `sips`.
- Boozer/DESC geometry import needs an early design choice: direct DESC API, SIMSOPT/Boozer objects, precomputed array fixtures, or multiple adapters.
- The current Phase 2 spectral operators are dense matrices, which is appropriate for early tests but may need matrix-free/modal application before larger production grids.
- Velocity-space strategy now has an explicit branch: Chebyshev collocation should carry the first linear solver, while the GX-style Hermite-Laguerre backend currently provides tested basis/transforms/diagnostics but not a full moment-space RHS.
- GX is a valuable method reference, but its GPU-native assumptions should not leak into the first CPU-oriented differentiable design unless a feature directly benefits the JAX implementation.
- `bessel_j0` currently uses a differentiable Cephes-style approximation rather than a JAX built-in because `jax.scipy.special` does not provide a reliable differentiable `j0` path in this environment. Keep the SciPy comparison tests as guards.
- The Phase 5A Hermite-Laguerre backend establishes transforms and moment diagnostics, but the primary near-term solver should still proceed through the collocation linear RHS path unless we explicitly decide to switch to a GX-style moment RHS.
- The Phase 6 zonal adiabatic correction is algebraically tested, but should still be parity-checked against reduced Gyaradax/GKW fixtures once Phase 7 couples the full RHS.
