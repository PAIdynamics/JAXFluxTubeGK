# STATUS

Last updated: 2026-05-31

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
static-vs-differentiable documentation. Phase 12 optimization integration now
has a fixed-topology single-surface objective layer, differentiable profile and
geometry knobs, reduced `rho`/`alpha`/`ky` scans, and a toy gradient-descent
example before full DESC coupling. The first DESC coupling path is now
implemented as a sampled-array adapter: DESC remains the equilibrium/geometry
provider, while this solver consumes the required flux-tube geometry arrays in
the fixed-topology objective. A direct optional DESC extraction adapter and CLI
script now evaluate those arrays from a DESC equilibrium/example object onto
`parallel_grid.z`. DESC dependencies are installed in the project `.venv`, and a
canonical DSHAPE `.npz` fixture generated through the DESC HDF5/path loader now
loads through the solver geometry contract. The current benchmark-informed
optimization pass adds named RH/CBC scalar targets, GX NetCDF growth-curve
loading, GX/GS2 eik-table loading, least-squares benchmark objective wrappers,
and a reduced DESC DSHAPE fixture optimization example. The immediate
validation-gate pass now adds executable RH/CBC gates and GX/GS2/DESC eik
metric gates. A follow-up hardening pass adds late-window growth fitting, a
calibrated reduced RH crossing regression hook, and
solver-geometry-to-GX/GS2-eik field parity reports. The active RH path now
passes a true late-time plateau gate over `t>80` by using GKW
finite-difference fallback stencils in `s` and `v_parallel`, exact zonal
initialization, direct fourth-difference `disp_par`/`disp_vp` recurrence
operators inside the residual, and a late-window mean-convergence check. The
passing CPU gate observes `0.07041301423095102` against the GKW/Gyaradax RH
target `0.0711`. The Cyclone selected-`ky` gate now uses the GKW
cell-centered `s` grid, `nperiod=5`, single-mode `ky=0.5` convention, GKW
finite-difference velocity fallback, a zero-boundary finite-difference
parallel fallback, an optional GKW/Gyaradax sign-dependent upwind parallel
stencil for Term I/Term VII, and a jitted production-control amplitude-window
runner. The medium production-control gate observes `0.16471456525100867`
against the GKW/Gyaradax target `0.179`, so it is narrowed but still OPEN
against the `0.01` tolerance. A CBC term-level audit now passes with zero
stored error for magnetic drift, equilibrium drive, drift-field drive, GKW
boundary maps, grid/velocity normalization, and assembled RHS conventions; the
new reduced CBC trace diagnostic records selected-`ky` raw and physical
amplitudes, per-window and fitted growth, phi/state/RHS norms, and
log-normalization for direct external comparison. Gyaradax runtime dependencies
are enabled through the optional `reference` extra and installed in the local
`.venv`. The Gyaradax trace exporter now supports named profiles and compares
normalization-independent physical fields against `CycloneTrace`; the reduced
comparison passes with maximum error `1.23687934e-02`, the
production-control-grid smoke comparison passes with maximum error
`1.32907879e-03`, and the full 80-window production-control comparison passes
with maximum error `1.01865677e-02`, all at tolerance `2.0e-02`, for time,
physical amplitude, window growth, fitted growth, physical phi norm, physical
state norm, and physical RHS norm. Raw amplitudes and raw log-normalization
are still normalization-convention dependent and are not production parity
gates yet. A GKW `time.dat` loader now maps GKW linear time/growth diagnostics
into the same `CycloneTrace` schema by reconstructing relative physical
amplitude from the reported growth-rate increments; field/state/RHS norms are
marked unavailable for that compact GKW format. The local serial/no-FFT GKW
build now succeeds with `gfortran`. A real GKW `simple_example` run has been
converted to `figures/gkw_simple_example_time_trace.csv`; it contains 50 time
samples with final GKW-reported window growth `0.184492` and
loader-reconstructed full-history fitted growth `0.16090982345149119`. A
matched linear selected-`ky` GKW run at the production-control grid/window
settings is now stored in `fixtures/gkw_cyclone_selected_ky_time.dat` and
`figures/gkw_cyclone_selected_ky_time_trace.csv`; its late-window mean growth
is `0.180407525`, close to the `0.179` target, while its reconstructed late-fit
growth is `0.18853144053590817`. The solver production gate now exposes both
`late_fit` and `late_mean_window` diagnostics plus explicit `cosine2`/`cosine`
initial-profile controls. The corresponding solver values remain open:
`0.1647145652510088` (`cosine2`, late fit), `0.15674153067144372`
(`cosine2`, late mean), `0.1659730160275755` (`cosine`, late fit), and
`0.15572083125648728` (`cosine`, late mean). A richer GKW `parallel_phi.dat`
diagnostic from the same matched run is now stored in
`fixtures/gkw_cyclone_selected_ky_parallel_phi.dat` and loaded through the
public `ParallelPhiTrace` API. The solver-side selected-`ky` parallel
`|phi|^2` trace uses the same production-control grid/window settings and the
GKW-native `cosine` initialization. The trace now supports GKW's unweighted
field normalization, which removes the raw total-power scale mismatch: the
mean solver/GKW total-power ratio is `1.0000000565887992` with maximum
deviation below `1.7e-06`. The row-normalized profile comparison is still
OPEN: maximum profile-shape error `3.38801745e-02`, mean row error
`1.84122540e-02`, and final-row error `2.20871902e-02` at exploratory
tolerance `2.0e-02`. Reversing the output order and circularly shifting the
GKW rows do not reduce the best-aligned error (`best_shift=0`). The localized
audit now identifies the worst signed shape error at `t=3.72`, `z=0.09375`:
the solver normalized value is `0.35697806498567025` while GKW's is
`0.3230978904417056`, with signed error `3.38801745e-02` and negative
profile-width error in that row. This confirms that the remaining CBC gap
includes a real central-curvature/width mismatch in the parallel mode
structure, not only a compact `time.dat` growth-window, scalar normalization,
output-ordering convention, center-of-power shift, or boundary-edge
concentration.
A reduced validation-gate example now writes CSV summaries and a paper figure
that show the current RH, Cyclone, CBC-term, GX/eik, DESC/eik, DESC/GX eik, and
GX/GIST gate status in `main.tex`, plus a reduced CBC trace CSV for the current
windowed selected-`ky` evolution and Gyaradax comparison CSVs. The
stellarator-geometry path now includes a
solver-produced DESC fixture export gate for GX/GS2 eik-compatible fields, so
DESC arrays can be audited through the same metric/drift and `k_perp^2`
contract before they are used in optimization. The external stellarator eik
path now also checks three independent GX/VMEC GIST fixtures, uses the correct
GIST drift-column order, and includes a matched DESC/GX block-`eik.out` DSHAPE
fixture with zero residual against the solver-produced DESC/GX-convention
geometry.

The repository currently contains:

- `task.tex`: project thesis description and six-month roadmap.
- `main.tex`: physical model and numerical scheme specification for the first solver implementation.
- `TODO.md`: project implementation plan.
- `STATUS.md`: this progress ledger.
- `docs/performance_and_differentiability.md`: Phase 11 CPU scaling, memory, and AD/topology notes.
- `docs/optimization_integration.md`: Phase 12 fixed-topology optimization and toy-gradient example.
- `examples/optimization_loop.py`: runnable reduced optimization loop that prints objective/growth diagnostics and knob values at each iteration.
- `examples/desc_fixture_optimization_loop.py`: runnable reduced benchmark-target optimization loop on the extracted DESC DSHAPE fixture.
- `examples/run_validation_gates.py`: runnable report for RH, CBC, and GX/eik validation gate status.
- `examples/generate_validation_gate_figures.py`: runnable reduced validation-gate figure and CSV generator for the `main.tex` result section.
- `examples/compare_gkw_parallel_phi_profile.py`: matched production-control GKW `parallel_phi.dat` versus solver selected-`ky` parallel-profile comparison generator.
- `scripts/export_gyaradax_cyclone_trace.py`: optional Gyaradax trace exporter with reduced, production-control-smoke, full production-control, and explicit `finit` profiles.
- `figures/validation_gate_status.pdf`, `figures/rh_plateau_demo.csv`, `figures/validation_gate_summary.csv`, `figures/cyclone_trace_reduced.csv`, `figures/gyaradax_cyclone_trace_reduced.csv`, `figures/gyaradax_cyclone_trace_comparison.csv`, `figures/gyaradax_cyclone_trace_production_control_smoke.csv`, `figures/gyaradax_cyclone_trace_production_control_smoke_comparison.csv`, `figures/gyaradax_cyclone_trace_production_control.csv`, `figures/gyaradax_cyclone_trace_production_control_comparison.csv`, `figures/gyaradax_cyclone_trace_production_control_gkw_cosine.csv`, `figures/gyaradax_cyclone_trace_production_control_gkw_cosine_comparison.csv`, `figures/gkw_simple_example_time_trace.csv`, `figures/gkw_cyclone_selected_ky_time_trace.csv`, `figures/gkw_cyclone_selected_ky_time_comparison.csv`, `figures/gkw_cyclone_parallel_phi_profile_comparison.csv`, and `figures/cyclone_growth_diagnostic_convention_comparison.csv`: current reduced validation-gate and CBC trace result artifacts.
- `fixtures/gkw_cyclone_selected_ky_linear_input.dat`, `fixtures/gkw_cyclone_selected_ky_time.dat`, and `fixtures/gkw_cyclone_selected_ky_parallel_phi.dat`: matched GKW selected-`ky` linear input, compact time diagnostic, and parallel `|phi|^2` diagnostic.
- `scripts/extract_desc_geometry_fixture.py`: optional DESC example-equilibrium geometry fixture extractor.
- `fixtures/desc_geometry_dshape_rho05_alpha0.npz`: small sampled DESC DSHAPE flux-tube geometry fixture.
- `fixtures/gx_desc_dshape_rho05_alpha0.eik.out`: matched GX DESC-convention block eik fixture for DSHAPE geometry parity.
- `pyproject.toml`: root Python package metadata for the `stellarator_gk` package.
- `uv.lock`: resolved project dependency lock file.
- `src/stellarator_gk/`: Phase 2 core types/grids, Phase 3 analytic geometry, Phase 4 flux-tube geometry adapters, the public linear residual wrapper, Phase 8 fixed-step time advancement, and Phase 9 objective/operator interfaces.
- `src/stellarator_gk/benchmarks.py`: named validation targets and lightweight GX/GX-eik reference loaders.
- `src/stellarator_gk/geometry/`: circular/\(s\)-alpha analytic geometry plus Boozer/precomputed/DESC flux-tube geometry scaffolding.
- `src/stellarator_gk/physics/`: Phase 5 Bessel/FLR, Maxwellian, drive, drift, mirror, streaming primitives, Phase 5A Hermite-Laguerre velocity-moment utilities, Phase 6 quasineutrality solvers, and Phase 7 linear RHS terms.
- `src/stellarator_gk/diagnostics.py`: Phase 6 diagnostic reductions, spectra, and quasilinear flux ingredients.
- `src/stellarator_gk/operators.py`: Phase 9 matrix-free residual actions, mode-chain projection helpers, dense reduced-operator construction, and tiny eigensystem helpers.
- `src/stellarator_gk/objectives.py`: Phase 9 growth-rate, selected-mode, quasilinear-proxy, mode-structure, and short initial-value objective helpers.
- `src/stellarator_gk/optimization.py`: Phase 12 optimization knobs, single-surface objectives, scan helpers, and toy gradient-descent step.
- `src/stellarator_gk/performance.py`: Phase 11 reduced-grid profiler, memory estimators, PyTree byte accounting, and byte-format helpers.
- `src/stellarator_gk/time_advance.py`: Phase 8 RK4 stepping, fixed-step scan integration, CFL estimate, per-`ky` normalization, and growth/frequency diagnostics.
- `tests/`: Phase 2 through Phase 12 unit, validation, performance-smoke, optimization, and differentiability tests.
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
- Couple to DESC through a sampled flux-tube geometry-array contract first; do not refactor or vendor DESC internals into this solver unless a later direct adapter proves that shared source code is necessary.
- Require tests for every new function added under `src/`.
- Update this file during every implementation round.

## Next Implementation Round

Goal: use the matched selected-`ky` GKW `time.dat` and `parallel_phi.dat`
traces, the passing Gyaradax/solver physical trace checks, and the explicit
growth-diagnostic selector to isolate the remaining production Cyclone
growth-history and parallel mode-structure gap while keeping DESC optimization
examples labeled as reduced until CBC parity passes:

- compare the localized GKW `parallel_phi.dat` central-curvature/width
  mismatch against the selected-mode parallel derivative, GKW-upwind boundary,
  and field-solve assembly,
- only patch GKW or add a restart/state-injection path for `cosine2` if the
  native `cosine` profile comparison cannot isolate the discrepancy,
- retain both `late_fit` and `late_mean_window` production-gate diagnostics
  until the GKW/Gyaradax selected-mode history gap is isolated,
- promote the production-control Cyclone growth-rate gate to PASS only after it
  is within the documented GKW/GX tolerance ladder,
- supplement the matched DESC/GX block-eik fixture with a truly independent
  external eik producer when a compatible GX/DESC, GS2, stella, or VMEC/GIST
  path is available.

Expected file changes:

- GKW/Gyaradax/solver parallel profile or amplitude-history comparison report updates,
- any selected-mode initialization or diagnostic-window adjustment needed for
  the scalar Cyclone gate,
- any independent eik producer/fixture discovered for DESC/GX geometry,
- `TODO.md`,
- `STATUS.md`

Expected tests:

- Cyclone selected-`ky` growth-rate tolerance test promoted from OPEN to PASS
  when the remaining physics gap is closed,
- existing matched DESC solver-produced geometry parity tolerance test retained,
- continued reduced DESC objective and gradient checks.

## Round Log

### 2026-06-01: Added GKW Parallel-Phi Alignment Audit

- Committed the previous parallel-phi profile tranche as:
  - `3ab8279 Add GKW parallel phi profile comparison`.
- Added `ParallelPhiProfileAudit` and
  `audit_parallel_phi_profile_alignment`.
- Extended the audit with localized profile diagnostics:
  - peak-position error,
  - profile-width/second-moment error,
  - global worst `(time,z)` signed shape error and corresponding solver/GKW
    profile values.
- Added `normalization_model='gkw_unweighted'` to
  `run_cyclone_base_case_parallel_phi_trace` so the solver trace can mimic the
  unweighted GKW field norm used by `normalise.F90`.
- Regenerated
  `figures/gkw_cyclone_parallel_phi_profile_comparison.csv` with the
  GKW-unweighted normalization model.
- Main findings:
  - total-power ratio is now near unity: mean `1.0000000565887992`, maximum
    `1.0000016391456865`, minimum `0.9999986687417727`,
  - direct, reversed, and best circular-shift profile errors are identical to
    stored precision,
  - the global best circular shift is `0`,
  - center-of-power error is tiny, with mean `-1.0985865346851282e-08`,
  - edge-fraction error is small, with mean `-2.496145394523138e-04`,
  - the worst localized signed profile error is `3.38801745e-02` at `t=3.72`,
    `z=0.09375`, where the solver value is `0.35697806498567025` and the GKW
    value is `0.3230978904417056`,
  - the second-moment/profile-width error is negative in that row
    (`-2.0768273362321468e-02`), indicating a central-width/curvature
    mismatch rather than a boundary-localized discrepancy,
  - the profile comparison remains OPEN with maximum error
    `3.38801745e-02`.
- Interpretation: the remaining Cyclone selected-`ky` profile gap is not
  primarily caused by raw field normalization, reversed/shifted output order,
  center-of-profile displacement, or boundary-edge concentration. The next
  check should compare the selected-mode parallel derivative and field-solve
  assembly at the central profile locations against the GKW/Gyaradax
  convention.
- Verification run this round:
  - `uv run --extra dev ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/compare_gkw_parallel_phi_profile.py`
  - `uv run --extra dev ruff check src tests examples scripts`
  - `uv run --extra dev pytest tests/test_benchmark_references.py::test_parallel_phi_profile_audit_detects_output_order_shift tests/test_benchmark_references.py::test_gkw_parallel_phi_trace_loader_compares_row_normalized_profiles -q`
  - `uv run --extra dev pytest tests/test_benchmark_references.py::test_cyclone_parallel_phi_trace_records_gkw_style_profiles tests/test_benchmark_references.py::test_parallel_phi_profile_audit_detects_output_order_shift -q`
  - `uv run --extra dev pytest tests/test_benchmark_references.py -q`
  - `uv run --extra dev pytest -q`
  - `uv run --extra dev python examples/compare_gkw_parallel_phi_profile.py`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

### 2026-05-31: Added GKW Parallel-Phi Profile Comparison

- Committed the previous matched GKW Cyclone diagnostics tranche as:
  - `ec26345 Add matched GKW Cyclone diagnostics`.
- Added public benchmark/profile objects:
  - `ParallelPhiTrace`,
  - `ParallelPhiTraceComparisonReport`.
- Added public helpers:
  - `load_gkw_parallel_phi_trace`,
  - `run_cyclone_base_case_parallel_phi_trace`,
  - `compare_parallel_phi_traces`.
- Stored the richer matched GKW selected-`ky` diagnostic:
  - `fixtures/gkw_cyclone_selected_ky_parallel_phi.dat`.
- Added `examples/compare_gkw_parallel_phi_profile.py`, which reruns the solver
  at the matched production-control settings and writes:
  - `figures/gkw_cyclone_parallel_phi_profile_comparison.csv`.
- Main findings:
  - the matched GKW `parallel_phi.dat` file has 80 rows and 48 parallel-grid
    columns, matching the existing GKW `time.dat` cadence,
  - the row-normalized solver/GKW parallel `|phi|^2` profile comparison is
    OPEN with maximum profile-shape error `3.38801745e-02` at time `3.72`,
  - the mean row error is `1.84122540e-02`,
  - the final-row error is `2.20871902e-02`,
  - the GKW-native direct compact trace path remains `finit='cosine'`; a
    `cosine2` comparison would require a GKW source patch or restart/state
    injection because the original GKW `finit` selector has no native
    `cosine2` branch.
- Updated `TODO.md` and `main.tex` to record that the remaining Cyclone gap now
  has a parallel mode-structure component, not just a compact `time.dat`
  diagnostic-window component.
- Verification run this round:
  - `uv run --extra dev ruff check examples/compare_gkw_parallel_phi_profile.py src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py`
  - `uv run --extra dev ruff check src tests examples scripts`
  - `uv run --extra dev pytest tests/test_benchmark_references.py::test_gkw_parallel_phi_trace_loader_compares_row_normalized_profiles tests/test_benchmark_references.py::test_gkw_parallel_phi_loader_reads_matched_selected_ky_fixture tests/test_benchmark_references.py::test_cyclone_parallel_phi_trace_records_gkw_style_profiles -q`
  - `uv run --extra dev pytest tests/test_benchmark_references.py -q`
  - `uv run --extra dev pytest -q`
  - `uv run --extra dev python examples/compare_gkw_parallel_phi_profile.py`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  - `git diff --check`

### 2026-05-31: Added Matched GKW Selected-ky Trace Diagnostics

- Committed the previous production trace/profile and GKW loader tranche as:
  - `fe934a1 Add production trace profiles and GKW loader`.
- Added `fixtures/gkw_cyclone_selected_ky_linear_input.dat`, a reproducible
  serial/no-FFT GKW input matching the production-control selected-`ky` grid:
  \(N_z=48\), \(N_{v_\parallel}=32\), \(N_\mu=8\), \(n_{\rm period}=5\),
  \(\Delta t=0.003\), 20 steps per output window, and 80 windows.
- Ran local GKW and stored:
  - `fixtures/gkw_cyclone_selected_ky_time.dat`,
  - `figures/gkw_cyclone_selected_ky_time_trace.csv`.
- Added an explicit `initial_profile` option to the Cyclone setup, trace, and
  production gate:
  - `cosine2`, the existing solver/Gyaradax default \(1+\cos(2\pi s)\),
  - `cosine`, the native compact GKW `finit='cosine'` path.
- Added an explicit production-gate `growth_diagnostic` selector:
  - `late_fit`, the selected-mode least-squares log-amplitude fit,
  - `late_mean_window`, the GKW `time.dat`-style mean of per-window growth
    samples.
- Extended the Gyaradax trace exporter with `--finit` and generated the
  GKW-style cosine production-control comparison:
  - `figures/gyaradax_cyclone_trace_production_control_gkw_cosine.csv`,
  - `figures/gyaradax_cyclone_trace_production_control_gkw_cosine_comparison.csv`.
- Added summary comparison artifacts:
  - `figures/gkw_cyclone_selected_ky_time_comparison.csv`,
  - `figures/cyclone_growth_diagnostic_convention_comparison.csv`.
- Main findings:
  - matched GKW `time.dat` late-window mean growth is `0.180407525`, close to
    the `0.179` target,
  - matched GKW reconstructed-amplitude late fit is `0.18853144053590817`,
  - solver production gate values are `0.1647145652510088` (`cosine2`,
    `late_fit`), `0.15674153067144372` (`cosine2`, `late_mean_window`),
    `0.1659730160275755` (`cosine`, `late_fit`), and
    `0.15572083125648728` (`cosine`, `late_mean_window`),
  - Gyaradax/solver production-control `cosine` trace comparison is OPEN at
    tolerance `2.0e-02`, with max selected-field error `2.63135798e-02`,
    dominated by per-window growth; fitted growth differs by only
    `1.27907199e-03`.
- Updated `TODO.md`, `STATUS.md`, and `main.tex`.
- Commands run:
  - `git commit -m "Add production trace profiles and GKW loader"`
  - `/Users/mohsensadr/Codes/GitHub/new-plasma-code/relevant-codes/gkw/gkw.x`
  - `JAX_ENABLE_X64=1 .venv/bin/python -c "... load_gkw_time_dat_trace ..."`
  - `MPLCONFIGDIR=/tmp/stellarator_gk_matplotlib JAX_ENABLE_X64=1 .venv/bin/python scripts/export_gyaradax_cyclone_trace.py --profile production-control --finit cosine --output figures/gyaradax_cyclone_trace_production_control_gkw_cosine.csv --comparison-output figures/gyaradax_cyclone_trace_production_control_gkw_cosine_comparison.csv`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_benchmark_references.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check src tests scripts examples`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
- Verification results:
  - matched GKW selected-`ky` run completed successfully,
  - focused benchmark tests: 21 passed,
  - full ruff: all checks passed,
  - full pytest suite: 140 passed,
  - `main.tex` built successfully with existing underfull-box warnings only.

### 2026-05-31: Added Production-Control Gyaradax Trace and GKW Loader

- Committed the normalization-equivalent trace comparison tranche as:
  - `081e836 Add physical norm trace parity`.
- Added named profiles to `scripts/export_gyaradax_cyclone_trace.py`:
  - `reduced`,
  - `production-control-smoke`,
  - `production-control`.
- The `production-control-smoke` profile uses the production grid/window
  controls \(N_z=48\), \(N_{v_\parallel}=32\), \(N_\mu=8\), 20 steps per
  window, and four windows. It writes:
  - `figures/gyaradax_cyclone_trace_production_control_smoke.csv`,
  - `figures/gyaradax_cyclone_trace_production_control_smoke_comparison.csv`.
- The full `production-control` profile uses the same grid/window controls and
  80 windows. It writes:
  - `figures/gyaradax_cyclone_trace_production_control.csv`,
  - `figures/gyaradax_cyclone_trace_production_control_comparison.csv`.
- The production-control smoke comparison passes with maximum selected-field
  error `1.32907879e-03` at tolerance `2.0e-02`; the dominant errors are the
  per-window and fitted growth fields. Physical amplitude, physical phi norm,
  physical state norm, and physical RHS norm errors remain below `1.5e-05`.
- The full production-control comparison passes with maximum selected-field
  error `1.01865677e-02` at tolerance `2.0e-02`; per-window growth is the
  largest field error, fitted growth differs by `1.99908042e-03`, and physical
  norm errors remain below `1.9e-04`.
- Added `load_gkw_time_dat_trace`, which reads GKW linear `time.dat` files,
  reconstructs relative physical amplitude from the reported growth-rate
  increments, and fills unavailable field/state/RHS norm diagnostics with
  zeros under an explicit note.
- Built the local GKW reference executable with the documented serial/no-FFT
  `gfortran` path and ran the bundled linear `simple_example` in
  `/private/tmp/gkw_simple_example_run`.
- Converted that real GKW `time.dat` into
  `figures/gkw_simple_example_time_trace.csv`; the converted trace has 50
  samples, final GKW-reported window growth `0.184492`, and full-history fitted
  growth `0.16090982345149119`.
- Updated `TODO.md`, `STATUS.md`, and `main.tex`.
- Commands run:
  - `git commit -m "Add physical norm trace parity"`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check scripts/export_gyaradax_cyclone_trace.py`
  - `MPLCONFIGDIR=/tmp/stellarator_gk_matplotlib JAX_ENABLE_X64=1 .venv/bin/python scripts/export_gyaradax_cyclone_trace.py`
  - `MPLCONFIGDIR=/tmp/stellarator_gk_matplotlib JAX_ENABLE_X64=1 .venv/bin/python scripts/export_gyaradax_cyclone_trace.py --profile production-control-smoke`
  - `MPLCONFIGDIR=/tmp/stellarator_gk_matplotlib JAX_ENABLE_X64=1 .venv/bin/python scripts/export_gyaradax_cyclone_trace.py --profile production-control`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_benchmark_references.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check src tests scripts examples`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `git diff --check`
  - `make FC=gfortran FFLAGS="-O2" FFTLIB=nofft PARALLEL=nompi LDFLAGS=""`
  - `/Users/mohsensadr/Codes/GitHub/new-plasma-code/relevant-codes/gkw/gkw.x`
  - `JAX_ENABLE_X64=1 .venv/bin/python -c "... load_gkw_time_dat_trace ..."`
- Verification results:
  - full ruff: all checks passed,
  - reduced Gyaradax comparison: PASS, max error `1.23687934e-02`,
  - production-control smoke Gyaradax comparison: PASS, max error
    `1.32907879e-03`,
  - full production-control Gyaradax comparison: PASS, max error
    `1.01865677e-02`,
  - focused benchmark tests: 19 passed,
  - full pytest suite: 138 passed,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - whitespace check passed,
  - GKW serial/no-FFT build succeeded,
  - GKW `simple_example` completed successfully.

### 2026-05-31: Added Normalization-Equivalent Trace Norms

- Committed the Gyaradax trace comparison tranche as:
  - `f07f287 Add Gyaradax Cyclone trace comparison`.
- Extended `compare_cyclone_base_case_traces` with derived physical norm
  fields:
  - `physical_phi_norm = phi_norm * exp(log_normalization)`,
  - `physical_state_norm = state_norm * exp(log_normalization)`,
  - `physical_rhs_norm = rhs_norm * exp(log_normalization)`.
- Updated the Gyaradax exporter so the reduced trace comparison now includes
  time, physical amplitude, window growth, fitted growth, and the three
  physical norm fields.
- The reduced Gyaradax comparison remains PASS with max selected-field error
  `1.23687934e-02` at tolerance `2.0e-02`; physical norm field errors are below
  `2.0e-07`.
- Updated `TODO.md` and `STATUS.md`.
- Commands run:
  - `git commit -m "Add Gyaradax Cyclone trace comparison"`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check src/stellarator_gk/benchmarks.py tests/test_benchmark_references.py scripts/export_gyaradax_cyclone_trace.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check src tests scripts examples`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_benchmark_references.py`
  - `MPLCONFIGDIR=/tmp/stellarator_gk_matplotlib JAX_ENABLE_X64=1 .venv/bin/python scripts/export_gyaradax_cyclone_trace.py`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest`
- Verification results:
  - focused benchmark tests: 17 passed,
  - full pytest suite: 136 passed,
  - full ruff: all checks passed,
  - Gyaradax exporter PASS with the physical norm fields included,
  - `main.tex` built successfully with existing underfull-box warnings only.

### 2026-05-31: Enabled Gyaradax Trace Export and Comparison

- Committed the previous validation/trace tranche as:
  - `7ad50c6 Add CBC trace diagnostics and eik parity fixture`.
- Installed missing local Gyaradax runtime dependencies into `.venv`:
  - `omegaconf`,
  - `einops`,
  - transitive `antlr4-python3-runtime` and `pyyaml`.
  The first direct `python -m pip` attempt failed because this uv-managed
  virtual environment does not include `pip`; `uv pip install` succeeded after
  allowing uv to use its package cache.
- Added a project optional dependency extra `reference` for those Gyaradax
  runtime dependencies.
- Added reusable `CycloneTrace` CSV helpers:
  - `write_cyclone_trace_csv`,
  - `load_cyclone_trace_csv`.
- Extended `compare_cyclone_base_case_traces` with selectable fields so
  normalization-independent physical fields can be compared separately from raw
  normalized amplitudes and raw norm diagnostics.
- Added derived normalization-equivalent physical norm fields to trace
  comparison:
  - `physical_phi_norm`,
  - `physical_state_norm`,
  - `physical_rhs_norm`.
- Added `scripts/export_gyaradax_cyclone_trace.py`, which:
  - imports local `relevant-codes/gyaradax`,
  - builds a reduced s-alpha Cyclone selected-`ky` run,
  - exports `figures/gyaradax_cyclone_trace_reduced.csv`,
  - compares time, physical amplitude, window growth, and fitted growth against
    the solver's `CycloneTrace`,
  - writes `figures/gyaradax_cyclone_trace_comparison.csv`.
- The reduced Gyaradax physical trace comparison passes:
  - max selected-field error `1.23687934e-02`,
  - tolerance `2.0e-02`,
  - compared fields: `times`, `physical_amplitude`, `window_growth`,
    `fitted_growth`, `physical_phi_norm`, `physical_state_norm`,
    `physical_rhs_norm`.
- Raw amplitudes and log-normalization are not yet pass criteria because
  Gyaradax normalizes the state to exactly unit raw amplitude at window
  boundaries, while this solver records the raw amplitude after its own
  per-window scale convention.
- Updated `TODO.md`, `STATUS.md`, and `main.tex`.
- Commands run:
  - `git commit -m "Add CBC trace diagnostics and eik parity fixture"`
  - `.venv/bin/python -m pip install omegaconf einops`
  - `uv pip install --python .venv/bin/python omegaconf einops`
  - `MPLCONFIGDIR=/tmp/stellarator_gk_matplotlib PYTHONPATH=relevant-codes/gyaradax .venv/bin/python - <<'PY' ...`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/generate_validation_gate_figures.py scripts/export_gyaradax_cyclone_trace.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_benchmark_references.py`
  - `MPLCONFIGDIR=/tmp/stellarator_gk_matplotlib JAX_ENABLE_X64=1 .venv/bin/python scripts/export_gyaradax_cyclone_trace.py`
  - `MPLCONFIGDIR=/tmp/stellarator_gk_matplotlib JAX_ENABLE_X64=1 uv run --extra dev --extra reference python scripts/export_gyaradax_cyclone_trace.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python examples/generate_validation_gate_figures.py`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest`
- Verification results:
  - focused benchmark tests: 17 passed,
  - full pytest suite: 136 passed,
  - full ruff: all checks passed,
  - documented Gyaradax exporter command passed and wrote both Gyaradax trace
    CSVs,
  - validation figure generator passed with the public trace CSV writer,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - reduced Gyaradax/CycloneTrace physical comparison PASS:
    `max_abs_error=1.23687934e-02`, tolerance `2.0e-02`.

### 2026-05-31: Added CBC Trace-Level Diagnostics

- Added public `CycloneTrace` and `CycloneTraceComparisonReport` PyTree
  dataclasses.
- Added `run_cyclone_base_case_trace`, which records selected-`ky` CBC
  diagnostics after fixed RK4 windows:
  - raw selected-mode amplitude,
  - physical amplitude including window normalization,
  - physical per-window growth,
  - cumulative fitted growth,
  - phi norm,
  - state norm,
  - RHS norm,
  - selected-mode log-normalization.
- Added `compare_cyclone_base_case_traces` for field-by-field trace parity.
- Added CLI support via `examples/run_validation_gates.py --cyclone-trace`.
- Regenerated validation artifacts and added
  `figures/cyclone_trace_reduced.csv`.  The reduced trace currently starts
  from amplitude `3.084441e-03` and records fitted growth
  `-6.736663e-01` by `t=0.048` for the short diagnostic example; it is an
  implementation trace artifact, not yet an external-reference pass.
- Direct local Gyaradax import is currently blocked by a missing optional
  dependency: `omegaconf`.  The next trace-parity step is therefore either to
  install/enable Gyaradax's runtime dependencies or export the equivalent GKW
  time-history diagnostics.
- Updated `TODO.md`, `STATUS.md`, and `main.tex`.
- Commands run:
  - `PYTHONPATH=relevant-codes/gyaradax .venv/bin/python -c "import gyaradax"`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/run_validation_gates.py examples/generate_validation_gate_figures.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_benchmark_references.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python examples/run_validation_gates.py --cyclone-trace --cyclone-trace-windows 2 --rh-steps 1 --cyclone-steps 1`
  - `JAX_ENABLE_X64=1 .venv/bin/python examples/generate_validation_gate_figures.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check src tests scripts examples`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
- Verification results:
  - focused benchmark tests: 16 passed,
  - full ruff: all checks passed,
  - full pytest suite: 135 passed,
  - validation CLI `--cyclone-trace` smoke printed the reduced trace table with
    physical first-window growth `-3.636966e-01`,
  - validation figure generator wrote `figures/cyclone_trace_reduced.csv`,
    `figures/rh_plateau_demo.csv`, `figures/validation_gate_summary.csv`, and
    `figures/validation_gate_status.pdf`,
  - `main.tex` built successfully after replacing a fragile `\path` command in
    the figure caption; existing underfull-box warnings remain.

### 2026-05-31: Added CBC Term-Level Parity Audit

- Added a GKW/Gyaradax sign-dependent upwind parallel fallback for CBC
  finite-difference parity:
  - GKW fourth-order upwind \(D_z\) stencil tables for positive/negative
    characteristics,
  - open-boundary `s`/`kx` shift maps,
  - fused `disp_par` recurrence control for Term I,
  - GKW upwind Term VII for the parallel field drive.
- Wired the Cyclone selected-`ky` gates to use
  `parallel_derivative_model="gkw_upwind"` by default, while retaining the
  centered matrix backend as an explicit comparison mode.
- Added `run_cyclone_base_case_term_parity_audit`, a public term-level CBC
  audit for:
  - magnetic drift frequency,
  - equilibrium-gradient drive,
  - drift-field drive,
  - GKW open-boundary maps,
  - GKW cell-centered \(s\), \(v_\parallel\), and \(\mu\) normalization,
  - assembled RHS identity.
- The term audit passes with max stored error `0.0`; the diagnostic difference
  between centered-matrix and GKW-upwind parallel boundary operators on the
  audit state is `1.091092e-04`.
- Re-ran the medium production-control CBC comparison:
  - centered matrix fallback: observed `0.16471725401913284`, residual
    `-1.428274598086715`,
  - GKW upwind fallback: observed `0.16471456525100867`, residual
    `-1.4285434748991326`.
- Conclusion: the audited drift/drive/field-drive/boundary/normalization
  conventions are not the visible source of the remaining CBC growth-rate gap.
  The next round should compare state evolution, phi solve history, RK4/window
  normalization, initialization, and growth diagnostics directly against a
  Gyaradax/GKW trace.
- Regenerated:
  - `figures/rh_plateau_demo.csv`,
  - `figures/validation_gate_summary.csv`,
  - `figures/validation_gate_status.pdf`.
- Updated `TODO.md`, `STATUS.md`, and `main.tex`.
- Commands run:
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check src/stellarator_gk/physics/rhs_terms.py src/stellarator_gk/solver.py src/stellarator_gk/benchmarks.py src/stellarator_gk/physics/__init__.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/run_validation_gates.py examples/generate_validation_gate_figures.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_linear_rhs.py tests/test_benchmark_references.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python examples/generate_validation_gate_figures.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python examples/run_validation_gates.py --cyclone-term-audit --rh-steps 1 --cyclone-steps 1`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check src tests scripts examples`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
- Verification results:
  - focused RHS/benchmark tests: 26 passed,
  - focused ruff: all checks passed,
  - full ruff: all checks passed,
  - full pytest suite: 134 passed,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - validation CLI `--cyclone-term-audit` smoke: CBC term parity PASS,
    reduced RH/CBC growth rows expected OPEN,
  - validation summary now includes CBC terms PASS and Cyclone growth OPEN
    (`0.16471456525100867`).

### 2026-05-31: Added DESC/GX Eik Parity and Hardened the CBC Production Gate

- Added a GX DESC-block `eik.out` loader and a DESC/GX-convention geometry
  evaluator that mirrors the GX field-line normalization while using the
  current DESC coordinate API.
- Added `run_desc_gx_eik_external_geometry_gate`, public exports, CLI support,
  tests, and the matched DSHAPE fixture
  `fixtures/gx_desc_dshape_rho05_alpha0.eik.out`.
- The DESC/GX block-eik gate passes with observed maximum field error `0.0`
  and normalized residual `0.0` at tolerance `2e-6`.
- Hardened the Cyclone selected-`ky` production-control gate:
  - target metadata now records GKW finite-difference velocity fallback,
    zero-boundary finite-difference parallel fallback, and `disp_par=1`,
  - the runner jits each fixed-step amplitude window and jits the phi solve,
  - the medium validation-summary run observes `0.16471725401913284` against
    the GKW/Gyaradax target `0.179`.
- The Cyclone gate is narrowed but still OPEN with normalized residual
  `-1.428274598086715`; the next CBC work is a term-level
  drift/drive/field/normalization audit against GKW/Gyaradax.
- Regenerated:
  - `figures/rh_plateau_demo.csv`,
  - `figures/validation_gate_summary.csv`,
  - `figures/validation_gate_status.pdf`,
  - `main.pdf`.
- Updated `TODO.md`, `STATUS.md`, and the `main.tex` results/validation text to
  keep DESC-driven optimization examples labeled reduced until CBC passes.
- Commands run:
  - `JAX_ENABLE_X64=1 .venv/bin/python examples/generate_validation_gate_figures.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/run_validation_gates.py examples/generate_validation_gate_figures.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_benchmark_references.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check src tests scripts examples`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `JAX_ENABLE_X64=1 .venv/bin/python examples/run_validation_gates.py --desc-gx-eik --rh-steps 1 --cyclone-steps 1`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m ruff check examples/run_validation_gates.py`
- Verification results:
  - focused benchmark-reference tests: 14 passed,
  - full ruff: all checks passed,
  - full pytest suite: 133 passed,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - validation CLI `--desc-gx-eik` smoke: PASS for DESC/GX eik and expected
    OPEN reduced RH/CBC smoke rows,
  - validation summary: RH plateau PASS, Cyclone OPEN (`0.16471725401913284`),
    GX/eik PASS, DESC/eik PASS, DESC/GX eik PASS, GX/GIST PASS.

### 2026-05-30: Closed the Active RH Late-Time Plateau Gate

- Added a GKW finite-difference velocity backend:
  - cell-centered \(v_\parallel\) nodes,
  - GKW \(2\pi v_\perp\,\Delta v_\perp\) `mu` quadrature weights,
  - zero-fill finite-difference \(v_\parallel\) derivative fallback.
- Added in-residual GKW `disp_vp` velocity recurrence control alongside the
  existing `disp_par` path.
- Corrected finite-difference recurrence operators to use the direct GKW
  fourth-difference stencil `[-1, 4, -6, 4, -1] / (12 h)` instead of `D1^4`.
- Updated the RH setup to use:
  - finite-difference fallback stencils in both `s` and `v_parallel`,
  - exact GKW/Gyaradax `finit='zonal'` conjugate \(k_x=\pm1\) initialization,
  - `disp_par=0.01`, effective `disp_vp=0.08`,
  - the documented \(t>80\) residual metric,
  - a two-half late-window mean-convergence check.
- The default RH plateau gate now passes:
  - observed `0.07041301423095102`,
  - target `0.0711`,
  - normalized residual `-0.6869857690489783`,
  - late-window mean delta `7.498586e-03`.
- Regenerated:
  - `figures/rh_plateau_demo.csv`,
  - `figures/validation_gate_summary.csv`,
  - `figures/validation_gate_status.pdf`.
- Commands run:
  - `.venv/bin/python -m ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/physics/rhs_terms.py src/stellarator_gk/grids.py src/stellarator_gk/types.py examples/generate_validation_gate_figures.py examples/run_validation_gates.py tests/test_benchmark_references.py tests/test_finite_difference.py tests/test_linear_rhs.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_finite_difference.py tests/test_linear_rhs.py tests/test_benchmark_references.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python -m pytest`
  - `.venv/bin/python -m ruff check src tests scripts examples`
  - `JAX_ENABLE_X64=1 .venv/bin/python -u -c "from stellarator_gk import run_rosenbluth_hinton_plateau_gate; g=run_rosenbluth_hinton_plateau_gate(t_end=100,t_start=80,diagnostic_interval=1); print(float(g.observed_value), float(g.residual), bool(g.passed)); print(g.notes)"`
  - `JAX_ENABLE_X64=1 .venv/bin/python examples/generate_validation_gate_figures.py`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
- Verification results:
  - RH plateau default gate: PASS,
  - focused finite-difference/linear-RHS/benchmark tests: 29 passed,
  - full pytest suite: 131 passed,
  - full ruff: all checks passed,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - validation summary: RH plateau PASS, RH endpoint OPEN, Cyclone OPEN,
    GX/eik PASS, DESC/eik PASS, GX/GIST PASS.

### 2026-05-30: Corrected Cyclone Selected-`ky` Benchmark Setup

- Committed the previous recurrence/eik hardening checkpoint:
  - commit `4e2cf9d` (`Add GKW recurrence and external eik gates`).
- Corrected the Cyclone gate to use the normalized GKW cell-centered parallel
  coordinate instead of feeding a Boozer-angle grid into the \(s\)-alpha
  geometry.
- Updated the Cyclone setup to use the documented selected-mode convention:
  `nperiod=5`, single `ky=0.5`, `vpar_max=3.0`, default
  `mu_max=vpar_max^2/2`, cosine2 initial condition, and `disp_par=1.0`.
- Added `run_production_cyclone_base_case_gate`, a memory-light production
  control path that keeps only per-window selected-mode amplitudes and fits the
  late-window growth rate. Tests exercise it with reduced overrides.
- Regenerated the validation summary/figure. The corrected reduced Cyclone
  smoke gate remains OPEN, now with observed growth `-1.9691178816845982`
  against the GKW/Gyaradax target `0.179`.
- Commands run:
  - `.venv/bin/python -m pytest tests/test_benchmark_references.py`
  - `.venv/bin/python -m ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py`
  - `.venv/bin/python -m ruff check src tests scripts examples`
  - `JAX_ENABLE_X64=1 .venv/bin/python examples/generate_validation_gate_figures.py`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `.venv/bin/python -m pytest`
  - `git diff --check`
- Verification results:
  - focused benchmark-reference tests: 12 passed,
  - focused ruff: all checks passed,
  - full ruff: all checks passed,
  - full pytest suite: 128 passed,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - `git diff --check`: clean,
  - validation summary: RH endpoint OPEN, RH plateau OPEN, Cyclone OPEN
    (`-1.9691178816845982`), GX/eik PASS, DESC/eik PASS, GX/GIST PASS.

### 2026-05-29: Added In-Residual GKW `disp_par` Recurrence Control

- Added a GKW/Gyaradax-scaled parallel recurrence-control term inside the
  linear RHS:
  `-disp_par * |a_parallel,rms| * (dz^3/12) * d_z^4 f`.
- Wired `parallel_recurrence_rate` through the RHS and coupled-residual
  precomputes. RH defaults now use `disp_par=0.01`, and the reduced Cyclone
  gate defaults to `disp_par=1.0`.
- Kept post-step modal damping available only as an experimental hook and set
  the RH validation defaults to zero modal damping so the residual is not
  artificially filtered.
- Added unit tests for the negative-semidefinite recurrence-control operator
  and the RMS velocity scaling that matches GKW `idisp=2`.
- Regenerated the reduced validation CSV/PDF artifacts with the in-residual
  `disp_par` path. RH and Cyclone remain OPEN; GX/eik, DESC/eik, and GX/GIST
  pass.
- Updated `main.tex` with the `disp_par` equation and with the current reduced
  validation status.
- Commands run:
  - `.venv/bin/python -m pytest tests/test_linear_rhs.py tests/test_benchmark_references.py`
  - `.venv/bin/python -m ruff check src/stellarator_gk/physics/rhs_terms.py tests/test_linear_rhs.py`
  - `JAX_ENABLE_X64=1 .venv/bin/python examples/generate_validation_gate_figures.py`
  - `.venv/bin/python -m ruff check src tests scripts examples`
  - `.venv/bin/python -m pytest`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `git diff --check`
- Verification results:
  - focused tests: 20 passed,
  - full test suite: 127 passed,
  - ruff: all checks passed,
  - LaTeX: `main.pdf` built successfully,
  - diff check: clean,
  - validation summary: RH endpoint OPEN (`0.9995304769749543`), RH plateau
    OPEN (`0.9987612200398621`), Cyclone OPEN (`18.77815606907163`), GX/eik
    PASS, DESC/eik PASS, GX/GIST PASS.

### 2026-05-29: Added External GX/VMEC GIST eik Suite Gate

- Committed the previous RH plateau, validation-figure, and DESC/eik checkpoint:
  - commit `9d678ea` (`Add RH plateau and stellarator validation gates`).
- Corrected `load_gx_eik_geometry_reference` for the local GIST/GS2 text
  fixture drift-column order:
  `theta, bmag, gradpar, gds2, gds21, gds22, cvdrift, cvdrift0, gbdrift, gbdrift0`.
- Added `run_gx_gist_external_eik_suite_gate`, which maps independent GX/VMEC
  GIST eik fixtures into solver geometry and compares the fields and
  `k_perp^2` contract across multiple stellarator references.
- Added tests for the corrected drift-column order and for the three-fixture
  external eik-suite gate.
- Extended `examples/run_validation_gates.py` with `--gx-gist-suite`.
- Updated `examples/generate_validation_gate_figures.py` and regenerated the
  validation CSV/PDF so the current result figure includes `GX/GIST` PASS.
- Updated `main.tex` and `TODO.md` to record that GX/VMEC GIST external eik
  coverage is in place while matched DESC-vs-external-eik parity remains open.
- Commands run:
  - `.venv/bin/python -m pytest tests/test_benchmark_references.py`
  - `.venv/bin/python -m ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/run_validation_gates.py examples/generate_validation_gate_figures.py`
  - `.venv/bin/python examples/run_validation_gates.py --desc-eik --gx-gist-suite --rh-plateau --rh-t-end 0.05 --rh-t-start 0.02 --rh-diagnostic-interval 0.01 --rh-plateau-n-z 8 --rh-plateau-n-vpar 6 --rh-plateau-n-mu 4`
  - `.venv/bin/python examples/generate_validation_gate_figures.py`
- Verification results:
  - focused benchmark-reference tests: 11 passed,
  - focused ruff: all checks passed,
  - validation CLI: `gx_gist_external_eik_suite` PASS with observed residual `0.0`,
  - regenerated validation summary now includes `GX/GIST` PASS alongside
    `GX/eik` PASS, `DESC/eik` PASS, and the open RH/Cyclone gates.

### 2026-05-29: Added DESC Geometry to the GX/eik Validation Path

- Added `geometry_to_gx_eik_reference`, an exporter from internal solver
  geometry to GX/GS2 eik-compatible fields:
  `B`, `gradpar`, `gds2`, `gds21`, `gds22`, summed radial/binormal drifts, and
  `k_perp^2`.
- Added `run_geometry_to_gx_eik_export_gate`, which verifies solver-produced
  stellarator geometry against its exported eik-compatible contract while
  keeping the internal mirror coefficient `G` separate from the eik drift table.
- Extended `compare_geometry_to_gx_eik_reference` with an
  `include_mirror_proxy` option so imported GX/eik self-parity can keep the
  historical mirror proxy while DESC export checks compare only fields present
  in standard eik files.
- Added a DESC DSHAPE fixture test showing the solver-produced DESC geometry
  passes the eik export contract with zero residual.
- Extended `examples/run_validation_gates.py` with `--desc-eik`.
- Updated `examples/generate_validation_gate_figures.py` and regenerated:
  - `figures/validation_gate_status.pdf`,
  - `figures/validation_gate_summary.csv`,
  - `figures/rh_plateau_demo.csv`.
- Updated `main.tex` with the eik export mapping and the new DESC/eik result in
  the current validation figure.
- Updated `TODO.md` to mark the DESC eik-export contract gate complete while
  leaving independent external eik-output parity open.
- Commands run:
  - `.venv/bin/python -m pytest tests/test_benchmark_references.py tests/test_desc_adapter.py tests/test_flux_tube_geometry.py`
  - `.venv/bin/python -m ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/run_validation_gates.py examples/generate_validation_gate_figures.py`
  - `.venv/bin/python examples/run_validation_gates.py --desc-eik --rh-plateau --rh-t-end 0.05 --rh-t-start 0.02 --rh-diagnostic-interval 0.01 --rh-plateau-n-z 8 --rh-plateau-n-vpar 6 --rh-plateau-n-mu 4`
  - `.venv/bin/python examples/generate_validation_gate_figures.py`
  - `.venv/bin/python -m ruff check src tests scripts examples`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `git diff --check`
  - `.venv/bin/python -m pytest`
- Verification results:
  - focused geometry/benchmark tests: 24 passed,
  - focused and full ruff: all checks passed,
  - validation CLI: DESC/eik export gate PASS with observed residual `0.0`,
  - regenerated validation summary now includes DESC/eik PASS alongside the
    existing GX/eik PASS and open RH/Cyclone gates,
  - LaTeX: `main.pdf` built successfully with only existing underfull-box warnings,
  - diff check: no whitespace errors,
  - full suite: 123 passed.

### 2026-05-29: Added Reduced Validation-Gate Result Example

- Added `examples/generate_validation_gate_figures.py`, which runs the reduced
  RH endpoint, true RH plateau, Cyclone, and GX/eik gates and writes:
  - `figures/rh_plateau_demo.csv`,
  - `figures/validation_gate_summary.csv`,
  - `figures/validation_gate_status.pdf`.
- Added the validation-gate figure and numerical summary to the result section
  of `main.tex`.
- Updated `TODO.md` to record the reduced validation-gate plotting example as
  part of the Phase 10 validation tranche.
- The example reports the current short-window RH plateau metric decreasing
  from `0.9999248529` at `t_end=0.02` to `0.9987611432` at `t_end=0.10`, still
  far from the RH reference `0.0711`; the gate remains correctly marked OPEN.
- The generated gate summary reports:
  - RH endpoint: OPEN, normalized residual `9.2843043975e+02`,
  - RH plateau: OPEN, normalized residual `9.2766114315e+02`,
  - Cyclone: OPEN, normalized residual `1.8825077228e+03`,
  - GX/eik: PASS, normalized residual `0.0`.
- Commands run:
  - `.venv/bin/python examples/generate_validation_gate_figures.py`
  - `.venv/bin/python -m ruff check examples/generate_validation_gate_figures.py`
  - `.venv/bin/python -m ruff check src tests scripts examples`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `.venv/bin/python -m pytest`
  - `git diff --check`
- Verification results:
  - figure/CSV generation completed and printed all three output paths,
  - ruff: all checks passed,
  - LaTeX: `main.pdf` built successfully with only existing underfull-box warnings,
  - full suite: 122 passed,
  - diff check: no whitespace errors.

### 2026-05-29: Added True RH Late-Plateau Gate

- Committed the previous validation-hardening checkpoint:
  - commit `18ce711` (`Add validation hardening gates`).
- Added `build_modal_damping_filter`, a reusable spectral post-step filter for
  Chebyshev/Fourier modal damping in `v_parallel`, `mu`, and `z`.
- Replaced the active RH validation path with
  `run_rosenbluth_hinton_plateau_gate`, which computes the GKW/Gyaradax
  late-time metric `sqrt(mean(kxspec(t)/kxspec(0)))` over `t > t_start`.
- Switched the RH setup to the normalized GKW cell-centered `s` grid and tracks
  the nonzero zonal `kx rho_s = 0.025` mode.
- The plateau gate supports benchmark-controlled modal damping, with the
  default parallel damping motivated by the RH `disp_par=0.01` reference. Local
  probes showed that strong velocity filtering can stabilize recurrence but
  damps the RH residual itself, so it is not accepted as a production pass.
- Updated `examples/run_validation_gates.py` with `--rh-plateau` and RH plateau
  controls; removed the calibrated-crossing option from the active CLI path.
- Updated `TODO.md` and `main.tex` to record that the true plateau gate exists
  but remains OPEN until the GKW/Gyaradax dissipation/RHS parity is closed.
- Commands run:
  - `.venv/bin/python -m pytest tests/test_time_advance.py tests/test_benchmark_references.py`
  - `.venv/bin/python -m ruff check src/stellarator_gk/time_advance.py src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_time_advance.py tests/test_benchmark_references.py examples/run_validation_gates.py`
  - `.venv/bin/python examples/run_validation_gates.py --rh-plateau --rh-t-end 0.05 --rh-t-start 0.02 --rh-diagnostic-interval 0.01 --rh-plateau-n-z 8 --rh-plateau-n-vpar 6 --rh-plateau-n-mu 4`
  - `.venv/bin/python -m pytest`
  - `.venv/bin/python -m ruff check src tests scripts examples`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `git diff --check`
- Verification results:
  - focused time-advance and benchmark-reference tests: 17 passed,
  - full suite: 122 passed,
  - focused and full ruff: all checks passed,
  - short RH plateau CLI smoke: plateau gate reports OPEN with finite late-window metric,
  - LaTeX: `main.pdf` built successfully with only existing underfull-box warnings,
  - diff check: no whitespace errors.

### 2026-05-29: Added Validation-Hardening Tools for RH, CBC, and eik Parity

- Added `windowed_linear_growth_diagnostics`, which fits
  `log(A_ky(t))` over a selected late-time window and returns the same
  `LinearGrowthDiagnostics` contract used by endpoint diagnostics.
- Updated the reduced Cyclone gate to use late-window growth extraction from a
  stored potential history. The gate still reports OPEN against the production
  GKW/GX target.
- Added `run_calibrated_reduced_rosenbluth_hinton_gate`, a deterministic
  reduced-grid RH crossing at `N_z=16`, `N_vpar=16`, `N_mu=8`, `dt=0.01`,
  `N_t=620`. It passes the scalar RH target as a regression hook, while the
  production long-time plateau remains open.
- Added `GxEikGeometryParityReport`,
  `compare_geometry_to_gx_eik_reference`, and
  `run_solver_geometry_to_gx_eik_gate` for field-by-field solver geometry
  checks against GX/GS2 eik tables.
- Extended `examples/run_validation_gates.py` with `--calibrated-rh`.
- Updated `main.tex` and `TODO.md` to document the new validation tools and the
  remaining production gaps.
- Commands run:
  - `.venv/bin/python -m pytest tests/test_time_advance.py tests/test_benchmark_references.py`
  - `.venv/bin/python -m ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/time_advance.py src/stellarator_gk/__init__.py tests/test_time_advance.py tests/test_benchmark_references.py examples/run_validation_gates.py`
  - `.venv/bin/python examples/run_validation_gates.py --calibrated-rh`
  - `.venv/bin/python -m pytest`
  - `.venv/bin/python -m ruff check src tests scripts examples`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `git diff --check`
- Verification results:
  - focused time-advance and benchmark-reference tests: 16 passed,
  - full suite: 121 passed,
  - focused ruff: all checks passed,
  - full ruff: all checks passed,
  - validation report: default RH and CBC OPEN, imported eik metric PASS,
    calibrated reduced RH PASS,
  - LaTeX: `main.pdf` built successfully with only existing underfull-box warnings,
  - diff check: no whitespace errors.

### 2026-05-29: Added DESC-Style Geometry Array Coupling

- Implemented `build_desc_geometry_from_arrays`, a thin adapter that maps DESC-sampled Boozer/Clebsch flux-tube arrays into the internal `FluxTubeGeometry` contract without importing or refactoring DESC internals.
- Extended `single_surface_objective` and `scan_single_surface_objective` to accept a supplied imported geometry object; `geometry_model="desc"`/`"precomputed"` now requires such a geometry object instead of generating toy analytic geometry.
- Added tests for DESC-array shape validation, physical-to-internal drift/mirror mapping, differentiation through supplied geometry arrays, imported-geometry objective values, and objective gradients.
- Updated `TODO.md` and `main.tex` to record the array-based DESC coupling strategy and leave direct DESC object/output extraction as the next source-adapter step.
- Commands run:
  - `.venv/bin/python -m pytest tests/test_flux_tube_geometry.py tests/test_optimization_integration.py`
  - `.venv/bin/python -m ruff check src/stellarator_gk/geometry/flux_tube.py src/stellarator_gk/optimization.py tests/test_flux_tube_geometry.py tests/test_optimization_integration.py`
  - `.venv/bin/python -m pytest`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `git diff --check`

### 2026-05-29: Added Direct DESC Extraction Adapter and Script

- Added `src/stellarator_gk/geometry/desc_adapter.py` with:
  - `DESC_GEOMETRY_COMPUTE_KEYS`,
  - `desc_geometry_arrays_from_data`,
  - `desc_geometry_arrays_from_equilibrium`,
  - `build_desc_geometry_from_equilibrium`.
- The adapter uses DESC's field-line coordinates \((\rho,\alpha,\zeta)\), computes the required vector contractions from DESC output, and feeds the existing `build_desc_geometry_from_arrays` solver contract.
- Added `scripts/extract_desc_geometry_fixture.py` to load a DESC example equilibrium, sample it on `build_boozer_parallel_grid`, and write the physical flux-tube arrays to an `.npz` fixture file.
- Added unit tests with a fake DESC equilibrium/grid path so the package tests do not require DESC as a hard dependency.
- Local caveat: running the script against `relevant-codes/DESC` in the current `.venv` stops at DESC import because the local DESC checkout dependencies are not installed (`colorama` is the first missing package).
- Commands run:
  - `.venv/bin/python -m pytest tests/test_desc_adapter.py tests/test_flux_tube_geometry.py tests/test_optimization_integration.py`
  - `.venv/bin/python -m ruff check src/stellarator_gk/geometry/desc_adapter.py src/stellarator_gk/geometry/__init__.py src/stellarator_gk/__init__.py tests/test_desc_adapter.py scripts/extract_desc_geometry_fixture.py`
  - `.venv/bin/python scripts/extract_desc_geometry_fixture.py --help`
  - `.venv/bin/python scripts/extract_desc_geometry_fixture.py --desc-root relevant-codes/DESC --output /private/tmp/desc_geometry_probe.npz` (expected dependency failure until DESC requirements are installed)
  - `.venv/bin/python -m pytest`
  - `.venv/bin/python -m ruff check src tests scripts`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `git diff --check`

### 2026-05-29: Installed DESC Dependencies and Generated DSHAPE Fixture

- Installed DESC requirements into `.venv` with `uv pip install --python .venv/bin/python -r relevant-codes/DESC/requirements.txt`.
- The install downgraded JAX/JAXLIB from `0.10.1` to DESC-compatible `0.9.2`.
- Verified the local DESC checkout imports with `PYTHONPATH=relevant-codes/DESC`; DESC reports `0.17.1+27.gc119da0f8`.
- Ran the extraction script against the local DESC `DSHAPE` example and wrote `fixtures/desc_geometry_dshape_rho05_alpha0.npz`.
- Added a fixture-regression test that loads the `.npz`, checks grid consistency and finite positive geometry arrays, and maps it through `build_desc_geometry_from_arrays`.
- Commands run:
  - `uv pip install --python .venv/bin/python -r relevant-codes/DESC/requirements.txt`
  - `PYTHONPATH=relevant-codes/DESC .venv/bin/python -c "import desc, jax; import desc.examples; print('desc', desc.__version__); print('jax', jax.__version__); print(desc.examples.listall()[:5])"`
  - `.venv/bin/python scripts/extract_desc_geometry_fixture.py --desc-root relevant-codes/DESC --example DSHAPE --rho 0.5 --alpha 0.0 --output fixtures/desc_geometry_dshape_rho05_alpha0.npz`
  - `.venv/bin/python -m pytest tests/test_desc_adapter.py tests/test_flux_tube_geometry.py tests/test_optimization_integration.py`
  - `.venv/bin/python -m ruff check tests/test_desc_adapter.py`
  - `.venv/bin/python -m pytest`
  - `.venv/bin/python -m ruff check src tests scripts`
  - `git diff --check`

### 2026-05-29: Added DESC HDF5/Path Loading

- Added public DESC path-loading helpers:
  - `load_desc_equilibrium`,
  - `desc_geometry_arrays_from_path`,
  - `build_desc_geometry_from_path`.
- Extended `scripts/extract_desc_geometry_fixture.py` with `--desc-path`, `--file-format`, and `--family-index` so it can sample either a named DESC example or a direct HDF5/pickle path.
- Enabled JAX x64 inside the extraction script before grid construction so generated fixtures preserve float64 arrays.
- Regenerated `fixtures/desc_geometry_dshape_rho05_alpha0.npz` through the direct DESC HDF5 path `relevant-codes/DESC/desc/examples/DSHAPE_output.h5`.
- Added fake-loader tests for path loading and verified the real local HDF5 path extraction once outside the unit-test suite.
- Commands run:
  - `.venv/bin/python -m pytest tests/test_desc_adapter.py`
  - `.venv/bin/python -m ruff check src/stellarator_gk/geometry/desc_adapter.py src/stellarator_gk/geometry/__init__.py src/stellarator_gk/__init__.py tests/test_desc_adapter.py scripts/extract_desc_geometry_fixture.py`
  - `.venv/bin/python scripts/extract_desc_geometry_fixture.py --desc-root relevant-codes/DESC --desc-path relevant-codes/DESC/desc/examples/DSHAPE_output.h5 --rho 0.5 --alpha 0.0 --output /private/tmp/desc_geometry_path_probe.npz`
  - `.venv/bin/python scripts/extract_desc_geometry_fixture.py --desc-root relevant-codes/DESC --desc-path relevant-codes/DESC/desc/examples/DSHAPE_output.h5 --rho 0.5 --alpha 0.0 --output fixtures/desc_geometry_dshape_rho05_alpha0.npz`

### 2026-05-29: Added Benchmark-Informed DESC Objective Round

- Committed the DESC extraction and benchmark-target checkpoint:
  - commit `510970c` (`Add DESC extraction and benchmark targets`).
- Added `src/stellarator_gk/benchmarks.py` with:
  - named `BenchmarkTarget` objects for the documented Rosenbluth-Hinton residual and Cyclone Base Case growth target,
  - differentiable target residual/cost helpers,
  - a GX NetCDF `omega_kxkyt` growth/frequency loader,
  - a GX/GS2 eik-style geometry table loader.
- Added `single_surface_benchmark_objective`, which wraps the fixed-topology single-surface objective as a least-squares error to a named benchmark target.
- Added `tests/test_benchmark_references.py` covering named targets, GX Cyclone reference loading, GX/GS2 W7-X eik table loading, and a differentiable reduced objective using `fixtures/desc_geometry_dshape_rho05_alpha0.npz`.
- Added `examples/desc_fixture_optimization_loop.py`, which prints per-iteration cost, residual, observed growth, and profile knobs on the extracted DESC DSHAPE fixture.
- Commands run:
  - `.venv/bin/python -m pytest tests/test_benchmark_references.py tests/test_optimization_integration.py`
  - `.venv/bin/python -m ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/optimization.py tests/test_benchmark_references.py examples/desc_fixture_optimization_loop.py src/stellarator_gk/__init__.py`
  - `.venv/bin/python examples/desc_fixture_optimization_loop.py --iterations 3 --learning-rate 0.005`
  - `.venv/bin/python -m pytest`
  - `.venv/bin/python -m ruff check src tests scripts examples`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `git diff --check`
- Verification results:
  - focused benchmark/optimization tests: 11 passed,
  - full suite: 115 passed,
  - ruff: all checks passed,
  - LaTeX: `main.pdf` built successfully with only existing underfull-box warnings.

### 2026-05-29: Added Executable Benchmark Validation Gates

- Extended `src/stellarator_gk/benchmarks.py` with:
  - `BenchmarkGateResult`,
  - `evaluate_benchmark_gate`,
  - reduced executable RH and CBC gates,
  - GX/GS2 eik resampling and metric-to-solver geometry mapping,
  - a GX/eik `k_perp^2` contract gate.
- Added `examples/run_validation_gates.py`, which prints PASS/OPEN status, observed value, reference, residual, tolerance, and notes for RH, CBC, and GX/eik gates.
- Extended `tests/test_benchmark_references.py` so the GX/eik metric gate must pass and the current reduced RH/CBC gates must run, remain finite, and explicitly report OPEN against production targets.
- Current quick gate output:
  - RH reduced gate: OPEN, observed residual proxy `9.99519817e-01` vs target `7.11000000e-02`,
  - CBC reduced gate: OPEN, observed selected growth `6.62190126e+00` vs target `1.79000000e-01`,
  - GX/eik metric gate: PASS, max `k_perp^2` contract error `0.0`.
- Commands run:
  - `.venv/bin/python -m pytest tests/test_benchmark_references.py`
  - `.venv/bin/python -m ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py`
  - `.venv/bin/python examples/run_validation_gates.py`
  - `.venv/bin/python -m ruff check examples/run_validation_gates.py src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py`
  - `.venv/bin/python -m pytest`
  - `.venv/bin/python -m ruff check src tests scripts examples`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `git diff --check`
- Verification results:
  - focused benchmark-reference tests: 6 passed,
  - full suite: 117 passed,
  - ruff: all checks passed,
  - LaTeX: `main.pdf` built successfully with only existing underfull-box warnings.

### 2026-05-29: Added Reduced Optimization Results to `main.tex`

- Added a new `Current Reduced Optimization Results` section to `main.tex`.
- Added `graphicx` support and included three generated PDF figures:
  - `figures/optimization_objectives.pdf`,
  - `figures/optimization_growth_rates.pdf`,
  - `figures/optimization_geometry_knobs.pdf`.
- Added `examples/generate_optimization_figures.py`, which runs three reduced fixed-topology optimization cases and writes:
  - `figures/optimization_traces.csv`,
  - the three figure PDFs used by `main.tex`.
- Regenerated the figure PDFs with Matplotlib axes, numeric tick labels, gridlines, and legends, replacing the initial minimal custom PDF writer.
- Extended the result-generation run from 12 to 1000 optimization iterations after the absolute growth-rate and geometry-knob plots looked visually flat.
- Replaced the signed-growth objective trace with a zero-target least-squares cost, `J = 0.5 * r**2`, and now plot the absolute objective error `|J - 0|` so the objective curve has a meaningful zero target.
- Switched the absolute objective-error figure to a logarithmic y-axis; exact zero values remain exact in the CSV and are drawn at a small positive plotting floor.
- Changed the growth-rate and geometry-knob figures to plot increments relative to their initial values, and collapsed duplicated selected/max growth-rate curves where they coincide.
- Tested larger learning-rate multipliers; uniform increases above `1e-3` become unstable for Case A over 1000 iterations, so Cases B and C were raised from `8e-4` to `1e-3` and all documented cases now use `1e-3`.
- Added Matplotlib to the development dependencies because the result-figure generator now uses it directly.
- The documented examples optimize zero-target residual costs:
  - `0.5 * gamma(ky=0.35)**2`,
  - `0.5 * gamma(ky=0.50)**2`,
  - `0.5 * max(gamma(ky>0))**2` over `ky=0.25,0.50`.
- The `main.tex` results section records the reduced simulation setup:
  - one kinetic ion with adiabatic electrons,
  - circular analytic geometry,
  - `N_vparallel=3`, `N_mu=3`, `N_z=5`, `N_kx=3`,
  - endpoint-only RK4 with `dt=0.01` and two steps per objective evaluation,
  - 1000 gradient-descent iterations,
  - fixed topology with differentiable continuous knobs.
- Commands run:
  - `uv run --extra dev python examples/generate_optimization_figures.py`
  - `.venv/bin/python examples/generate_optimization_figures.py`
  - `.venv/bin/python -c <learning-rate multiplier sweep>`
  - `.venv/bin/python -m ruff check examples src tests`
  - `.venv/bin/python -m pytest`
  - `sips -s format png figures/<optimization figure>.pdf --out /tmp/<preview>.png`
  - `latexmk -pdf -interaction=nonstopmode main.tex`
  - `git diff --check`
- Verification:
  - figure generation completed and wrote all expected PDFs with axes, ticks, and legends,
  - the 1000-iteration CSV contains 3000 trace rows plus the header and remained finite for all three cases,
  - the plotted absolute objective error reaches `0.0` for Cases B and C and `3.78e-8` for Case A in the current 1000-iteration reduced examples,
  - rendered PNG previews of the three PDFs with `sips` to visually check axes, tick labels, and legends,
  - `.venv/bin/python -m ruff check examples src tests` passed,
  - `99 passed` in pytest,
  - `git diff --check` passed,
  - `main.tex` built successfully to `main.pdf`.

### 2026-05-29: Added Runnable Optimization Loop Example

- Committed the Phase 12 optimization integration checkpoint:
  - commit `e2a831f` (`Add Phase 12 optimization integration`).
- Added `examples/optimization_loop.py`.
- Extended `docs/optimization_integration.md` with the example run command.
- The example prints one row per optimization iteration:
  - scalar objective,
  - selected growth rate,
  - max growth rate,
  - `q`,
  - `shat`,
  - `R/L_T`,
  - `R/L_n`,
  - first two toy equilibrium coefficients.
- Commands run:
  - `uv run --extra dev python examples/optimization_loop.py --iterations 3`
  - `uv run --extra dev ruff check src tests examples`
  - `uv run --extra dev python examples/optimization_loop.py --iterations 2`
- Verification:
  - example ran successfully and printed three optimization iterations.
  - `ruff check src tests examples` passed.
  - example rerun printed two optimization iterations without x64 warnings.

### 2026-05-29: Implemented Phase 12 Optimization Integration Baseline

- Committed the Phase 10/11 validation and performance hardening checkpoint:
  - commit `b7a8f07` (`Add Phase 10 and 11 validation hardening`).
- Added `src/stellarator_gk/optimization.py` with:
  - `OptimizationKnobs`,
  - `SingleSurfaceOptimizationConfig`,
  - `SingleSurfaceOptimizationResult`,
  - `OptimizationScanResult`,
  - `ToyOptimizationStep`,
  - `build_optimization_species`,
  - `build_optimization_geometry`,
  - `single_surface_objective`,
  - `scan_single_surface_objective`,
  - `toy_gradient_descent_step`.
- Extended `initial_value_growth_objectives` with `store_history` so optimization paths can use endpoint-only Phase 11 integration.
- Added public exports for the Phase 12 optimization API in `src/stellarator_gk/__init__.py`.
- Added `docs/optimization_integration.md` with the fixed-topology AD contract and a toy `jax.value_and_grad`/gradient-step example.
- Added `tests/test_optimization_integration.py` covering:
  - mapping differentiable knobs to species and analytic geometry,
  - jitted `jax.value_and_grad` through the single-surface objective,
  - finite-difference agreement for a profile-gradient knob,
  - static scans over `rho`, `alpha`, and selected `ky`,
  - a toy gradient-descent update on the optimization knobs.
- Updated `TODO.md` to mark the Phase 12 baseline complete and record DESC/Boozer geometry objectives as the next extension.
- Commands run:
  - `uv run --extra dev python -m pytest tests/test_optimization_integration.py`
  - `uv run --extra dev ruff check src tests`
  - `uv run --extra dev python -m pytest`
- Verification:
  - `4 passed` for `tests/test_optimization_integration.py`.
  - `ruff check src tests` passed.
  - `99 passed` for the full pytest suite.

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
