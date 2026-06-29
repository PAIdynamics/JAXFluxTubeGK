# STATUS

Last updated: 2026-06-29

## Snapshot

The repository now contains a working JAX-first linear electrostatic
flux-tube gyrokinetic solver architecture.  The core path supports
perpendicular Fourier modes, spectral/collocation `z/v_parallel/mu` grids,
GKW finite-difference parity backends, self-consistent electrostatic
quasineutrality, matrix-free RHS evaluation, RK4 time advancement,
growth/frequency/mode-structure diagnostics, differentiable objectives, and
fixed-topology optimization examples.

The main implementation lives in `src/stellarator_gk/`.  The validation and
artifact-producing workflows live mostly in `examples/`, `scripts/`,
`fixtures/`, and `tests/`.

## Implemented

- Core types, grids, mode connectivity, finite-difference fallback operators,
  and static-vs-differentiable PyTree contracts.
- Analytic circular and s-alpha geometry plus Boozer/precomputed, DESC-array,
  DESC-path, eik, and stella geometry import paths.
- Physics primitives: Bessel/FLR factors, Maxwellian, thermodynamic drive,
  magnetic drift, mirror force, parallel streaming, adiabatic and kinetic field
  solves, velocity integrals, spectra, and flux/proxy diagnostics.
- Linear RHS assembly with term-level decomposition, `gkw_upwind`, and
  source-matched `gkw_igh` parity paths.
- RK4 integration, windowed growth/frequency diagnostics, mode-chain amplitude
  normalization, matrix-free operator helpers, objective wrappers, and reduced
  optimization loops.
- Hermite-Laguerre basis/moment utilities and GX-style hypercollision/moment
  discriminator hooks.
- Reduced RH, Cyclone, Gyaradax/GKW trace, DESC/eik, W7-X, stella, and GX
  handoff validation infrastructure.

## Passing Guardrails

- True Rosenbluth-Hinton late-plateau gate passes.
- Cyclone selected-ky scalar growth gate passes after matching the GKW
  `KTHRHO/kthnorm` convention.
- Cyclone CBC term algebra, GKW RHS/action trace, imported-state replay,
  initial/first-window contract, state normalization, and row-normalized
  `parallel_phi.dat` profile checks pass their current contracts.
- DESC/GX block-eik, DESC eik export, GX/GS2 eik import, and independent
  GX/VMEC GIST eik-source checks pass.
- Reduced W7-X scan, convergence/timing ledgers, and DESC optimization demos are
  executable but remain explicitly reduced/non-production.

## Active Blocker

The next scientific milestone is an externally validated linear W7-X run.  A
matched stella W7-X fixture exists at
`fixtures/w7x_itg_external_mode_structure_fixture.csv`, exported from the local
CPU stella run with `ky=(0.1,0.2,0.3)`, `kx=0`, `nzed=256`, `nmu=8`,
`nvgrid=16`, `tend=200`, and adiabatic electrons.

The solver has been matched to the stella geometry, field-line length,
selected `ky`, `kx=0`, and late-time window.  At `t=200`, growth is close
(`max_growth_error=8.00978267e-03`), but the W7-X gate remains open because the
`ky=0.3` frequency/profile errors remain large:

- `frequency_error=-1.65618027e-01` in the closest stella-scale
  finite-difference velocity case;
- `phi_phase_aligned_error=1.56517320e-01` for the same focus mode.

Velocity-grid refinement alone did not close the gap.  The solver-side
`ky=0.3` RHS/model balance is internally consistent, with zero RHS
reconstruction error and quasineutrality residual near machine precision.  The
balance is streaming dominated.

The standard stella `.out.nc` output confirms the available geometry/streaming
contract but lacks complex distribution and per-term RHS arrays.  A targeted
non-destructive stella trace patch now exists and produced a raw 263 MB
selected-mode trace in `/tmp`, summarized in
`fixtures/w7x_ky03_stella_rhs_trace_summary/rhs_trace_summary.json`.

The first stella-vs-solver scalar term comparison is narrowed but not decisive.
After fixing the stella geometry adapter to use the global-header `flux_fac`
for the equilibrium-gradient drive coefficient, the solver drive fraction moved
from `0.07783134` to `0.61459833` versus stella `0.77998161`.  The largest
remaining scale-free discrepancy is now the parallel-streaming bundle.

Current comparison status:
`fixtures/w7x_ky03_stella_rhs_trace_comparison/stella_solver_rhs_trace_comparison_status.json`
reports `blocked_array_contract_mismatch` because the current stella trace and
solver balance fixture use different z/velocity grids and the raw stella trace
is format v1 without velocity quadrature weights.

## Next Actions

1. Rerun the patched stella RHS trace in v2 format with `wgts_vpa` and
   z-dependent `wgts_mu`.
2. Add or generate a solver-side selected-mode full-array trace on a
   stella-compatible `z/vpa/mu` grid, or document an interpolation/weighting
   adapter.
3. Drop the duplicate stella z endpoint in the comparator and compare weighted
   complex arrays for distribution, streaming, mirror, magnetic drift,
   equilibrium drive, field-drive terms, total RHS, and field solve pieces.
4. If the array comparison confirms the current scalar mismatch, inspect the
   parallel-streaming derivative/linking convention before changing other
   physics.
5. Once W7-X parity passes, rerun production-shape convergence and guarded CPU
   timing, then let the production-readiness ledger decide whether DESC
   optimization can be labeled production-ready.

## Tests and Commands

Use focused tests while editing a subsystem, then broader checks before making a
production claim:

```bash
uv run ruff check src tests examples scripts
JAX_ENABLE_X64=1 uv run pytest tests/test_w7x_stella_rhs_trace_comparison.py -q
JAX_ENABLE_X64=1 uv run pytest tests/test_w7x_ky03_rhs_model_balance.py -q
JAX_ENABLE_X64=1 uv run pytest tests/test_stella_w7x_rhs_trace_prep.py \
  tests/test_stella_w7x_rhs_trace_summary.py \
  tests/test_w7x_stella_rhs_trace_comparison.py -q
```

The full test suite is:

```bash
JAX_ENABLE_X64=1 uv run pytest
```

Run it when the change affects shared solver behavior or validation contracts.

## Current Risks

- W7-X production readiness is blocked by solver/stella mode-structure parity,
  not by missing entry-point code.
- Current W7-X convergence/timing artifacts are reduced regression artifacts,
  not production benchmarks.
- Full selected-mode GKW state-history and multi-time velocity-slice parity
  remain open confidence gaps, even though scalar growth and RHS/action parity
  are strong.
- Multi-ky Cyclone/GX low-ky branch-shape parity remains open until a true
  external complex fixture or stronger moment-RHS comparison is available.
- The Hermite-Laguerre backend is a tested discriminator/future backend seed,
  not the current production kinetic RHS.
- Nonlinear turbulence, kinetic-electron TEM validation, collisions, and
  electromagnetic physics are deferred.

## Round Log

### 2026-06-29: Concise Planning Refresh

- Reorganized `TODO.md` into a current prioritized backlog.
- Reorganized `STATUS.md` into a compact status snapshot, active blocker, next
  actions, and test commands.
- Updated `main.tex` to keep the paper focused on the model, current validation
  status, and remaining W7-X validation tasks.
- No solver code was changed in this round.

### 2026-06-11: W7-X stella RHS Trace and `flux_fac` Correction

- Added the non-destructive stella `ky=0.3` RHS trace preparation and summary
  path.
- Added the stella-vs-solver RHS trace comparator and committed comparison
  fixtures.
- Fixed the stella geometry adapter's equilibrium-drive coefficient to use
  stella `flux_fac`.
- Current blocker moved from missing stella RHS data to a direct
  array-contract mismatch and remaining parallel-streaming-dominated scalar
  discrepancy.

### Earlier Milestones

- Implemented Phases 2-12: core types/grids, analytic and stellarator geometry,
  physics primitives, Hermite-Laguerre utilities, quasineutrality, diagnostics,
  linear RHS, time advancement, objectives, performance hardening, and reduced
  optimization integration.
- Closed major RH, selected Cyclone scalar, GKW RHS/action, imported-state
  replay, DESC/eik, and reduced validation-gate contracts.
