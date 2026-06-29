# TODO: Differentiable Flux-Tube Stellarator GK Solver

Last reviewed: 2026-06-29

## Goal

Build a differentiable, local flux-tube, linear electrostatic stellarator
gyrokinetic solver in JAX, following GKW/Gyaradax physics conventions and using
GX/stella/DESC as geometry, method, and benchmark references.  The current
milestone is not nonlinear turbulence yet; it is a trusted externally validated
linear W7-X stellarator run that can gate CPU timing and reduced DESC
optimization.

Keep the workflow simple:

1. Specify physics and numerics in `main.tex`.
2. Implement solver functionality in `src/`.
3. Add or update focused tests in `tests/` for every new public function.
4. Keep `STATUS.md` current with commands, results, and remaining blockers.

## Current State

Implemented and tested:

- core PyTree parameter/data types, Fourier grids, spectral collocation grids,
  mode connectivity, and GKW finite-difference fallback operators;
- circular, s-alpha, Boozer/precomputed, DESC-array, DESC-path, eik, and stella
  geometry adapters;
- Bessel/FLR, Maxwellian, drive, mirror, drift, streaming, quasineutrality, and
  diagnostic primitives;
- self-consistent matrix-free linear residual, `gkw_upwind` and `gkw_igh`
  parity backends, RK4 time stepping, growth/frequency diagnostics, objectives,
  and fixed-topology optimization helpers;
- Hermite-Laguerre basis, moment diagnostics, GX-style hypercollision hooks,
  and a reduced moment-RHS discriminator, but not a production replacement for
  the collocation kinetic RHS;
- reduced RH, Cyclone, Gyaradax/GKW, DESC/eik, W7-X, stella, and GX handoff
  validation infrastructure.

Current trusted guardrails:

- true Rosenbluth-Hinton late-plateau gate passes;
- Cyclone selected-ky scalar growth, term algebra, GKW RHS/action trace,
  imported-state replay, initial/first-window contract, and row-normalized
  `parallel_phi.dat` profile contracts pass their current tolerances;
- DESC/GX/eik geometry and independent GX/VMEC GIST eik-source contracts pass;
- reduced stellarator scan and optimization examples run and remain labeled
  reduced.

Current blockers:

- W7-X solver/stella parity remains open after matching geometry, field-line
  length, `ky`, `kx=0`, late-time window, and simple velocity resolution.
  Growth is close at `t=200`, but the `ky=0.3` real frequency/profile gate is
  still open.
- The W7-X `ky=0.3` scalar term comparison is narrowed but not yet an array
  parity proof.  After the stella `flux_fac` drive correction, the largest
  remaining scalar discrepancy is the parallel-streaming bundle.
- Production W7-X convergence, CPU timing, and DESC optimization readiness stay
  blocked until external W7-X parity passes.
- Multi-ky Cyclone/GX low-ky branch-shape parity is open pending a true
  external complex mode-structure fixture or a stronger moment-RHS comparison.
- Full nonlinear turbulence, kinetic-electron TEM production validation,
  collisions, electromagnetic effects, and full DESC shape optimization are
  deferred extensions.

## Immediate Next Round

### 1. Close W7-X stella `ky=0.3` term-array parity

- [ ] Rerun the patched stella RHS trace in format v2 so the raw trace includes
  `wgts_vpa` and z-dependent `wgts_mu`.
- [ ] Update the stella trace comparator to drop the duplicate periodic z
  endpoint before array comparison.
- [ ] Emit a solver-side selected-mode full-array trace on a stella-compatible
  `z/vpa/mu` grid, or add a documented interpolation/weighting adapter from the
  solver grid to the stella grid.
- [ ] Compare velocity-weighted complex arrays for distribution, parallel
  streaming, mirror force, magnetic drift, equilibrium drive, field-drive terms,
  total RHS, quasineutrality numerator/denominator, and field normalization.
- [ ] Inspect the parallel-streaming derivative/linking convention first if the
  weighted arrays confirm the current scalar mismatch.

Useful commands:

```bash
uv run python scripts/prepare_stella_w7x_rhs_trace_run.py --overwrite \
  --output-root /tmp/stellarator_gk_stella_w7x_rhs_trace
bash /tmp/stellarator_gk_stella_w7x_rhs_trace/build_stella_rhs_trace.sh
bash /tmp/stellarator_gk_stella_w7x_rhs_trace/run_stella_rhs_trace.sh
uv run python scripts/summarize_stella_w7x_rhs_trace.py \
  /tmp/stellarator_gk_stella_w7x_rhs_trace/run/stellarator_gk_w7x_ky03_rhs_trace.dat \
  --output fixtures/w7x_ky03_stella_rhs_trace_summary/rhs_trace_summary.json
uv run python scripts/compare_w7x_stella_rhs_trace_to_solver_balance.py --require-raw-trace
```

### 2. Promote W7-X from reduced regression to production claim

- [ ] After W7-X solver/stella term parity passes, rerun the W7-X
  mode-structure gate against the matched stella fixture.
- [ ] Replace the reduced W7-X convergence ladder with a production-control
  ladder in `n_z`, velocity resolution/backend, `kx/ky`, field-line length,
  timestep, and growth-window diagnostics.
- [ ] Rerun guarded production CPU timing only after the parity ledger passes.
- [ ] Keep `examples/desc_fixture_optimization_loop.py --require-production-ready`
  blocked until the readiness ledger passes.

Useful commands:

```bash
uv run python examples/run_w7x_mode_structure_gate.py \
  --observed-fixture fixtures/w7x_itg_stella_matched_time_ladder/runs/time_200/mode_structures.csv \
  --reference-fixture fixtures/w7x_itg_external_mode_structure_fixture.csv \
  --ky-values 0.1,0.2,0.3 --resample-reference-to-observed-z
uv run python scripts/run_w7x_production_readiness_gate.py
uv run python scripts/run_w7x_production_cpu_timing.py
```

### 3. Keep secondary external references available

- [ ] Optionally run the prepared GX W7-X workflow on a CUDA/GX-capable machine
  and ingest returned `.big.nc`/`.out.nc` files as a secondary moment-method
  cross-check.
- [ ] Re-run the exact GX Cyclone input-control scan after a true GX
  `.big.nc` complex mode-structure artifact exists for the low-ky branch.

Useful commands:

```bash
uv run python scripts/package_w7x_external_reference_bundle.py \
  --output fixtures/gx_w7x_mode_structure_run/w7x_external_reference_bundle.tar.gz
GX_EXECUTABLE=/path/to/gx bash fixtures/gx_w7x_mode_structure_run/run_external_reference.sh
bash fixtures/gx_w7x_mode_structure_run/ingest_returned_outputs.sh \
  --copy-outputs --resample-reference-to-observed-z
```

### 4. Maintain confidence gaps without blocking the W7-X path

- [ ] Keep the GKW full selected-mode state-history and multi-time velocity-slice
  discrepancies visible until either closed or explicitly demoted by an
  independent stellarator reference.
- [ ] Decide whether Chebyshev/GKW finite-difference velocity collocation is the
  production CPU backend or whether the GX-style Hermite-Laguerre backend must
  become production capable before optimization studies.
- [ ] Keep the multi-ky Cyclone low-ky branch-shape gap in the validation ledger.

### 5. Deferred physics after trusted linear W7-X

- [ ] Kinetic-electron TEM validation.
- [ ] Collisions and electromagnetic perturbations.
- [ ] Nonlinear ExB pseudo-spectral bracket, dealiasing, nonlinear timestep
  control, saturated heat-flux diagnostics, and nonlinear benchmark parity.
- [ ] Full DESC optimization over real equilibrium shape degrees of freedom,
  multiple surfaces/field lines, remeshing/topology handling, and production
  readiness gates.

## Rules

- Keep `main.tex` as the concise physics/numerics source of truth; keep history
  and raw details out of the paper unless they support a claim.
- Keep `STATUS.md` as a current snapshot plus a short round log, not an
  exhaustive transcript.
- Do not claim production stellarator optimization or nonlinear turbulence until
  the relevant parity, convergence, timing, and physics-extension gates pass.
- Prefer small focused tests and explicit fixture artifacts over hidden
  convention fixes.
