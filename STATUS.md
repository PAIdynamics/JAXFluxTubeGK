# STATUS

Last updated: 2026-06-02

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
runner. The matched 48/32/8 production-control gate now observes
`late_mean_window=0.17800063460817828` against the GKW/Gyaradax target
`0.179`, after applying the internal s-alpha `KTHRHO/kthnorm` convention. A
CBC term-level audit now passes with zero
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
concentration. A selected-mode operator audit at the same point now passes the
field/RHS consistency checks: field residual `4.742874840267547e-16`, phi
reconstruction error `3.7238012298709097e-16`, and RHS assembly error `0.0`.
The local matrix-versus-GKW-upwind streaming delta is
`5.228399497117055e-03`, while the local field-drive delta is
`2.7611004325485534e-19`. A source-reconstructed audit of
`linear_terms.F90::vpar_grad_df_4d_testnewbc` now matches the current Term I
implementation with maximum term error `5.137226812672801e-18`, maximum
coefficient error `4.440892098500626e-16`, and `idisp=2` recurrence-speed
error `5.551115123125783e-17`. This rules out the selected-mode Term I
stencil/sign/normalization implementation itself. A source-sequence audit of
`exp_integration.F90::rk4` plus `normalise.F90::normalize(2)` now matches the
solver's GKW-unweighted selected-mode window trace with RK4 step error
`3.476216610979977e-18`, growth-sequence error `1.2878587085651816e-14`,
post-normalization field-norm error `1.3322676295501878e-15`, and field
linearity error `1.668583973270389e-15`. This rules out the explicit
time-update/field-normalization cadence itself. A GKW diagnostic/packing audit
now reconstructs the `dist.F90::get_phi` field layout and
`diagnostic.F90::{parallel_phi,phi_ky_spec}` formulas with zero packing,
`parallel_phi`, `ky`-spectrum, and `kx`-spectrum error at the localized
production-control audit point. The remaining profile gap now points toward
GKW matrix/time-history construction or source-term/matrix-format conventions
rather than the quasineutrality solve, parallel field-drive assembly,
GKW-upwind streaming formula, RK4 staging, `normalise.F90` field norm, or GKW
field diagnostic packing.
A reduced `matdat.F90` sparse matrix/source convention audit now passes with
matrix-action error `2.6021205067653364e-18`, zero source term,
explicit-`dtim` delta error `8.47042252985074e-21`, duplicate-triplet
compression error `3.035783099198609e-18`, and `complex-real` split error
`1.7347484980098753e-18` on a 192-state reduced CBC matrix. This rules out the
homogeneous source convention, explicit `dtim*(source+mat*fdis_tmp)` action,
duplicate `(ii,jj)` compression, and `complex-real` matrix-format split as
remaining reduced matrix-format causes.
A production-control selected-`ky` coefficient/source audit now reconstructs
GKW `linear_terms.F90` Terms II, IV, V, VII, and VIII
(`vdgradf`, `trapdf_4d`, `ve_grad_fm`, `vpgrphi_3_newbc`, and
`vd_grad_phi_fm`) at the same localized profile point. It passes with maximum
term error `3.1031676915590914e-17`, zero coefficient error, and maximum
insertion error `3.1031676915590914e-17` at `t=3.72`, `z=0.09375`. This
rules out the separated source-level coefficient construction/insertion for
those terms as the remaining production selected-`ky` cause.
A production-control `ltrapping_arakawa` fused `igh` audit now reconstructs
GKW's combined Term I/IV Hamiltonian stencil plus `disp_par=1` and
`disp_vp=0.2` diffusion. It is intentionally OPEN against the separated solver
fallback: at `t=3.72`, `z=0.09375`, the local fused-vs-separated profile delta
is `3.5775663113790776e-04`, the maximum profile delta is
`1.5086176659840009e-03`, and the relative envelope is
`0.1257254487392138`. The GKW `igh` parallel and velocity diffusion components
peak at `7.044002628973732e-03` and `7.597330135842486e-04`. This makes the
fused production `igh` operator the next actionable CBC parity gap.
The optional matrix-free `gkw_igh` backend has now been rerun at the matched
production-control grid. It narrows but does not close the CBC gap. A
non-destructive GKW `cosine2` preparation path is now available as
`scripts/prepare_gkw_cosine2_run.py`; because GKW declares `finit` as a
six-character selector, the copied source uses `finit='cosin2'` for the
solver/Gyaradax \(1+\cos(2\pi s)\) profile. The patched serial/no-FFT GKW tree
builds and runs from `/tmp` without modifying `relevant-codes/gkw/`. The new
raw fixtures are stored as
`fixtures/gkw_cyclone_selected_ky_cosin2_linear_input.dat`,
`fixtures/gkw_cyclone_selected_ky_cosin2_time.dat`, and
`fixtures/gkw_cyclone_selected_ky_cosin2_parallel_phi.dat`. Against these
patched GKW fixtures, the pre-`krho`-fix solver `cosine2`/`gkw_igh` late-fit
growth was `0.16389932979797434` versus GKW `0.18741518752345235`, the
late-window mean was `0.15623725215738368` versus GKW `0.177999075`, and the
final-window growth was `0.13945213935812423` versus GKW `0.188741`. After
applying the GKW internal `KTHRHO/kthnorm` convention, the matched selected-`ky`
production-control growth gate passes: `late_fit=0.18560606277298422` and
`late_mean_window=0.17800063460817828` against target `0.179`. Initialization,
field normalization, and magnetic-drift wave-number conventions are therefore
no longer the dominant remaining explanations.
The selected-`ky` gap audit now combines the patched GKW `cosin2` time/profile
fixtures with the solver `cosine2`/`gkw_igh` traces in one aligned diagnostic.
It reports solver late-fit growth `0.1648622859363632` versus GKW
`0.18741518752345235`, late-fit delta `0.02255290158708914`, late-window mean
delta `0.021761822842616324`, final-window growth delta
`0.049288860641875765`, maximum row-normalized profile error
`0.029625447419939166`, and total-power ratio mean
`0.9999999827218208` with maximum deviation `1.953378815588991e-06`.
This rules out field total-power normalization as the remaining dominant
cause; the next narrowed target is the evolved velocity-space/distribution
state, using GKW `distr*.dat` slices or a final-state/restart dump.
The next velocity-space audit is now in place. GKW `distr1.dat`--`distr4.dat`
from the patched `cosin2` run are stored as fixtures and loaded as the peak-\(\phi\)
slice \((v_\parallel,\sqrt{2B\mu},\Im[f\,\Delta\mu\,\Delta v_\parallel/\phi],
\Re[f\,\Delta\mu\,\Delta v_\parallel/\phi])\). The matching solver diagnostic
uses the same final time, peak-\(\phi\) point, velocity weights, and
normalization. The production-control audit reports `vpar_error=5.0e-05`,
`vperp_error=4.664794806363837e-05`, `time_error=0.0`,
`complex_max_abs_error=0.035348895748192916`, and
`complex_relative_l2_error=0.378493904183015`. The follow-on convention audit
uses GKW's actual Fortran output contract: `global_vpar_mu(n_vpar,n_mu)` is
written by `output_slice_2d` as `mu` rows and \(v_\parallel\) columns. It finds
that a \(v_\parallel\)-column reversal is the best simple variant, reducing
the max error to `0.018543579974657488` and \(L^2\) error to
`0.005533740922357686`; direct ordering has max error
`0.035348895748192916` and \(L^2\) error `0.00845562568878207`. Transpose,
`mu` reversal, one-cell shifts, conjugation, sign flips, and \(\pm i\) phase
rotations do not beat the \(v_\parallel\)-reversed direct comparison. The
even/odd decomposition shows the same structure: the even-part error is
`0.01647851294172267` max / `0.005197749203708426` \(L^2\), the odd same-sign
error is `0.02430342389518037` max / `0.006669408444842744` \(L^2\), and the
odd opposite-sign error drops to `0.00606276656656781` max /
`0.0018988659276327384` \(L^2\). A controlled odd-\(v_\parallel\) RHS sign
audit now keeps GKW's 1-based `k=1,\ldots,nvpar` and
`vpgr=-vpmax+(k-0.5)*dvp` ordering explicit while flipping only selected odd
RHS blocks. It rules out a global fused `ltrapping_arakawa`/`igh` sign
reversal: flipping the fused Term I/IV block increases the direct final-slice
max error to `17.078737087702983`. Flipping only the parallel field-drive
Term VII path (`vpgrphi_3_newbc`) is diagnostic but informative: it makes the
direct Fortran layout the best layout, moves the peak point from `z=0.09375`
to `z=-0.09375`, and reduces the direct complex max error from
`0.035348895748192916` to `0.02010162421775191` with relative \(L^2\) error
`0.2691199000880014`. The next target is therefore the true Term VII
source/matrix/field-variable convention across `linear_terms.F90`,
`matdat.F90`, and `dist.F90::get_phi`, not a broad velocity-grid transpose or
global `igh` sign flip. A Term VII field-variable convention audit now varies
the potential supplied to Terms V, VII, and VIII. It rules out a global field
sign or conjugation convention: global sign, global conjugation, and global
negative-conjugation all worsen the direct final-slice error
(`0.12796120197506905`, `0.07481363070580742`, and
`0.11470706992462161`). The best diagnostic variant is Term VII-only
conjugation, with direct max error `0.01596045557377426`, \(L^2\) error
`0.005113660917074626`, and relative \(L^2\) error
`0.2288996174155926`; Term VII-only negative-conjugation is close but worse
at `0.016980699849443844`. This is still diagnostic and is not adopted as a
physics change. The remaining narrowed target is the Term VII complex phase
path through `vpgrphi_3_newbc`, `add_element`, `matdat.F90`, complex-real
matrix conventions, \(k_y\)/Fourier signs, and `dist.F90::get_phi`. A full
normalized state/restart or multi-time velocity-slice dump remains useful if
this final peak slice is not enough to localize the cumulative discrepancy.
The first pass through that complex phase path found no hidden source-side
conjugation: `vpgrphi_3_newbc` inserts real scalar coefficients into `iphi`,
`add_element` maps them directly to the connected `iphi` column, `put_element`
stores the coefficient unchanged in both `complex` and `complex-real` formats,
`dist.F90::get_phi` copies `fdis(iphi)` directly, and
`diagnostic.F90::velocity_space_output` normalizes by `phi`, not
`conjg(phi)`. A new velocity-slice phase-alignment audit also rules out a pure
eigenfunction/output phase. The best unit-phase variant remains
`reverse_vpar_columns:identity` with max error `0.018885582880959564` and
relative \(L^2\) error `0.24656223896253462`; the direct unit-phase error is
`0.033230903073356986`. Even allowing an unconstrained complex scale only
reduces the best max error to `0.017952545317588844`. The remaining issue is
therefore not a free final-slice phase or amplitude convention.
A non-destructive multi-time GKW velocity-slice path is now available through
`scripts/prepare_gkw_cosine2_run.py --multi-time-distr`. It patches only the
copied `diagnostic.F90`, imports `ntotstep`, calls
`velocity_space_output(ntotstep)` at every normal diagnostic output, and
writes suffixed snapshots while preserving the unsuffixed final `distr*.dat`
files. The patched serial/no-FFT selected-`ky` `cosin2` tree built and ran
from `/private/tmp/stellarator_gk_gkw_cosin2_multitime_v2`, producing 320
suffixed files: 80 diagnostic windows times four velocity-slice components,
from `distr1_00000020.dat` through `distr4_00001600.dat`. The public
`GkwVelocitySpaceSliceSeries`/`load_gkw_velocity_space_slice_series` API reads
that run as 80 snapshots with shape `(80, 8, 32)`, time range `0.06` to `4.8`,
and final snapshot index `1600`. The next useful discriminator is now a
multi-window solver/GKW velocity-slice comparison plus \(k_y\)/Fourier sign
tracing.
That discriminator has now been extended to controlled RHS and field-convention
variants over steps 20, 800, and 1600. The global `igh` sign flip remains ruled
out because it grows to direct complex max error `17.078737087702983` by step
1600, and global field conjugation worsens the middle/final snapshots. The best
direct diagnostic variant at step 800 is a Term VII-only sign flip, reducing the
baseline `0.036712468463562305` to `0.02604522658732149`; the best direct
variant at step 1600 is Term VII-only conjugation, reducing the baseline
`0.035348895748192916` to `0.01596045557377426`. This does not close the gap or
justify a solver RHS change. It localizes the remaining issue to the selected
mode Term VII \(k_y\)/Fourier sign and field-packing path rather than to a
global field convention or a fused `igh` sign convention.
The selected-mode Term VII source-path audit now closes that branch. It
reconstructs GKW `mode.F90` positive single-mode `krho`, `dist.F90::get_phi`
field packing, and `linear_terms.F90::vpgrphi_3_newbc` from the source-level
contracts. The production-control run at \(t=3.72\) has selected
`ky=gkw_krho=0.5`, Python boundary maps `ixplus=ixminus=-1` corresponding to
GKW's open `0` maps for the single nonzonal \(k_x\) chain, direct field
roundtrip error zero, and direct Term VII packed-field action error
`3.471140604459642e-18`. Conjugating or negating the packed field changes the
Term VII action by `0.028893793199993845` and `0.028925615674737656`,
respectively. Thus Term VII sign/conjugation variants remain diagnostic only;
the source-level mode/Fourier sign and field-packing path is direct.
The follow-on multi-window `ltrapping_arakawa`/`igh` series audit samples the
same fused-vs-separated profile metric at windows 1, 40, and 80. It remains
OPEN, but the worst maximum profile delta is at the first sampled window:
`0.014010106905660624` at \(t=0.06\), then `0.0028004685541289983` at
\(t=2.4\), and `0.0006152079285486119` at \(t=4.8\). The corresponding relative
deltas are `0.8985196896095599`, `0.15207297474358986`, and
`0.056947861475111494`. This means the fused-vs-separated `igh` mismatch is an
early operator-shape difference, not a monotone growth mechanism matching the
step-20-to-step-800 GKW/solver velocity-slice divergence. The next useful target
is a matched state-history, source-term time trace, or restart/final-state dump
under the promoted `gkw_igh` backend.
The solver-side half of that source-trace discriminator is now implemented.
`CycloneSourceTermTrace` records selected-mode raw phi/state/RHS norms, term
norms for fused `igh`, magnetic drift, equilibrium drive, GKW parallel
field-drive, drift-field drive, and dissipation, a raw reconstructed-RHS norm,
the direct reconstruction error, and the accumulated selected-`ky`
log-normalization. The production-control `cosine2`/`gkw_igh` artifact
`figures/cyclone_source_term_trace.csv` samples windows 0, 1, 40, and 80 with
maximum stored reconstruction error `0.0`. At the final sampled window
\(t=4.8\), the raw selected-mode RHS norm is `9.143666705809655e-02`, the
magnetic-drift norm is `9.042892647805491e-02`, the fused-`igh` norm is
`4.486003543507541e-03`, and the accumulated log-normalization is
`-4.398714455712693`. The remaining production task is now to generate or load
a matched GKW state/source/restart trace against this solver-side artifact.
The GKW-side compact state-history pathway is now prepared as a non-destructive
copy-tree patch. `scripts/prepare_gkw_cosine2_run.py --state-trace` adds a
`diagnostic.F90` hook that writes `stellarator_gk_state_trace.dat` at each
normal diagnostic output with columns `step`, `time`, `state_norm`, and
`phi_norm`; it can be combined with `--multi-time-distr`. The Python side now
loads this file as `GkwStateTrace` and compares it to `CycloneSourceTermTrace`
with `compare_gkw_state_trace_to_source_term_trace`. The matched
production-control serial/no-FFT GKW `cosin2` state trace is now stored in
`fixtures/gkw_cyclone_selected_ky_cosin2_state_trace.dat`, and the comparison
artifact is `figures/gkw_cosin2_cyclone_state_trace_comparison.csv`. Because
GKW writes diagnostics after `normalize(2,...)`, the solver source trace now
supports `snapshot_timing="post_normalization"` for this comparison. The
compact norm-level comparison passes at tolerance `5.0e-03`: maximum error
`3.3077755368903644e-03`, time error `7.016609515630989e-14`, state-norm error
`3.3077755368903644e-03`, and field-norm error
`6.106226635438361e-16`. Final GKW/solver state norms are
`0.016112898161769` and `0.016826748946865613`; final field norms agree at
`0.144337567297406`. Compact state/field norms therefore no longer explain the
remaining CBC growth/profile discrepancy by themselves.
The full selected-mode GKW state dump is now the next completed discriminator.
`scripts/prepare_gkw_cosine2_run.py --selected-state-dump` patches only the
copied GKW `diagnostic.F90` and writes
`stellarator_gk_selected_state_<step>.dat` files with the selected
distribution on the full `(z,mu,v_parallel)` grid plus complex `phi(z)`. The
sampled production-control fixtures at steps 20, 800, and 1600 are stored under
`fixtures/gkw_cyclone_selected_ky_cosin2_selected_state/` and load as
`SelectedModeStateTrace` with shape `(3, 32, 8, 48)`. The solver comparison
artifact is `figures/gkw_cosin2_cyclone_selected_state_comparison.csv`. Direct
layout is best, but the comparison remains OPEN after snapshot-wise complex
phase alignment: maximum `phi(z)` error `5.845543963934988e-02`, maximum
full-state error `4.780479264318512e-02`, and worst state relative \(L^2\)
error `0.3192978051120589` at step 800. This rules out scalar norms and simple
diagnostic layout as sufficient explanations and makes a matched GKW
term-resolved RHS/source trace at the same selected snapshots the next
actionable target.
The GKW side of that RHS/source trace is now available. The copied-source
`--rhs-trace` patch tags GKW matrix/source entries by term in `matdat.F90`,
sets term IDs through `linear_terms.F90`, and writes dtim-scaled selected-mode
actions from `exp_integration.F90` immediately after end-of-window
`normalize(2,...)`. The sampled fixtures are stored under
`fixtures/gkw_cyclone_selected_ky_cosin2_rhs_trace/` and load as
`GkwSelectedModeRhsTrace` with shape `(3, 9, 32, 8, 48)`. The internal
term-sum error is `9.550830415404176e-18`. At the final snapshot, the
dtim-scaled total action norm is `1.9404731497669497e-02`, dominated by
`vdgradf` (`1.912308487778263e-02`), followed by `ve_grad_fm`
(`2.6230193433310666e-03`) and `igh_or_term_i`
(`1.2506893514028245e-03`). The solver now records matching full selected-mode
term-action arrays through `run_cyclone_base_case_selected_rhs_trace()` and
compares them with `compare_selected_mode_rhs_traces()`. The production
GKW/solver elementwise RHS/action comparison improved after applying the GKW
s-alpha `mode.F90::kgrid` convention `krho = KTHRHO/kthnorm`, with
`kthnorm=q/(2*pi*eps)`. The best diagnostic layout is now direct, and the max
error is `2.2379489422925073e-05`; per-term errors are dominated by `vdgradf`
(`2.2379489422925073e-05`), then `igh_or_term_i`
(`1.1169352619622038e-06`), `ve_grad_fm`
(`4.5614093015065964e-08`), and `vd_grad_phi_fm`
(`3.9597207251785146e-08`). The matched selected-`ky` production-control
growth gate now passes at 48/32/8 resolution: `late_fit=0.18560606277298422`
and `late_mean_window=0.17800063460817828` against the `0.179` target.
The earlier multi-time velocity-slice discriminator has also been run on
early/mid/final fixtures. The
solver/GKW direct complex max errors at steps 20, 800, and 1600 are
`3.990529105190601e-03`, `3.6712468463562305e-02`, and
`3.5348895748192916e-02`; grid and time errors stay at GKW output precision.
The best simple layout is direct at step 20 but switches to
`reverse_vpar_columns:identity` by steps 800 and 1600. This shows that the
large final distribution mismatch is mostly cumulative between early and
mid-run time, not a pure final peak-\(\phi\) output phase or isolated final
slice-location bias. A same-state selected RHS/action replay now loads the GKW
selected-state snapshots and evaluates the solver term actions on those exact
states. This reduces the strict RHS/action residual from `2.2379489422925073e-05`
to `8.905204720648363e-07`. The remaining same-state residual is dominated by
`igh_or_term_i`; `vdgradf` drops to `3.1370451207010515e-09`, and replaying with
the dumped GKW `phi` instead of the solver-computed field gives the same
residual to the reported precision. The next narrowed consistency target is
therefore the fused GKW `ltrapping_arakawa`/`igh` action timing and trace
construction, not the magnetic-drift frequency or field solve. A source-level
same-state `igh` replay audit now compares the patched GKW `igh_or_term_i`
trace, solver fused `gkw_igh_streaming_mirror`, and reconstructed
`linear_terms.F90::igh` action on identical imported GKW states. The solver and
source reconstruction agree to `1.6446404919551064e-19`, while both remain
offset from the patched GKW trace by `8.905291461911714e-07`. This localizes
the remaining same-state `igh_or_term_i` residual to GKW RHS trace
tagging/timing or matrix/source-action extraction rather than the implemented
fused operator coefficients.
A reduced validation-gate example now writes CSV summaries and a paper figure
that show the current RH, Cyclone, CBC-term, GX/eik, DESC/eik, DESC/GX eik, and
GX/GIST gate status in `main.tex`, plus a reduced CBC trace CSV for the current
windowed selected-`ky` evolution and Gyaradax comparison CSVs. The
stellarator-geometry path now includes a
solver-produced DESC fixture export gate for GX/GS2 eik-compatible fields, so
DESC arrays can be audited through the same metric/drift and `k_perp^2`
contract before they are used in optimization. The external stellarator eik
path now also checks three independent GX/VMEC GIST fixtures, uses the correct
GIST drift-column order, exposes an independent external eik-producer
report/gate with per-producer errors and source names, and includes a matched
DESC/GX block-`eik.out` DSHAPE fixture with zero residual against the
solver-produced DESC/GX-convention geometry.

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
- `examples/audit_cyclone_profile_operator.py`: selected-mode Cyclone operator audit at the localized GKW profile mismatch.
- `examples/audit_cyclone_term_i_fortran.py`: selected-mode Cyclone Term I audit reconstructed directly from the GKW Fortran source formulas.
- `examples/audit_cyclone_time_normalization.py`: selected-mode Cyclone RK4/window-normalization audit against the GKW source sequence.
- `examples/audit_cyclone_diagnostic_packing.py`: selected-mode Cyclone GKW `get_phi` field-packing and diagnostic-output audit.
- `examples/audit_cyclone_matdat_matrix.py`: reduced Cyclone `matdat.F90` sparse matrix/source convention audit.
- `examples/audit_cyclone_coefficient_source.py`: production-control selected-`ky` GKW Term II/IV/V/VII/VIII coefficient/source audit.
- `examples/audit_cyclone_igh_arakawa.py`: production-control selected-`ky` GKW `ltrapping_arakawa` fused Term I/IV audit.
- `examples/audit_cyclone_igh_arakawa_series.py`: multi-window production-control GKW `ltrapping_arakawa` fused Term I/IV audit.
- `examples/audit_cyclone_term_vii_mode_packing.py`: selected-mode Term VII `mode.F90`/field-packing source-path audit.
- `examples/audit_cyclone_cosin2_gap.py`: combined patched-GKW `cosin2` selected-`ky` growth/profile gap audit.
- `examples/audit_cyclone_cosin2_velocity_slice.py`: patched-GKW `cosin2` selected-`ky` `distr*.dat` velocity-space slice audit.
- `examples/audit_cyclone_cosin2_velocity_series.py`: patched-GKW `cosin2` selected-`ky` multi-time `distr*_<ntotstep>.dat` velocity-space slice audit.
- `examples/audit_cyclone_cosin2_velocity_series_variants.py`: multi-time `gkw_igh`/Term VII field-variant audit against patched-GKW `cosin2` velocity slices.
- `examples/audit_cyclone_cosin2_vpar_odd_signs.py`: controlled odd-\(v_\parallel\) RHS sign audit against patched-GKW `cosin2` velocity slices.
- `examples/audit_cyclone_cosin2_term_vii_field_conventions.py`: controlled Term VII field-variable convention audit against patched-GKW `cosin2` velocity slices.
- `examples/audit_cyclone_cosin2_velocity_phase.py`: global phase/complex-scale alignment audit for patched-GKW `cosin2` velocity slices.
- `examples/compare_gkw_selected_state.py`: sampled full selected-mode GKW state dump versus solver state comparison with snapshot-wise phase alignment.
- `scripts/prepare_gkw_cosine2_run.py`: non-destructive helper that copies GKW to a scratch tree, adds the six-character `finit='cosin2'` initializer, writes a matched selected-`ky` input, and can optionally patch copied diagnostics/source to emit multi-time `distr*_<ntotstep>.dat` velocity-slice snapshots, compact state traces, full selected-mode state dumps, or selected-mode RHS/source action traces.
- `scripts/export_gyaradax_cyclone_trace.py`: optional Gyaradax trace exporter with reduced, production-control-smoke, full production-control, and explicit `finit` profiles.
- `figures/validation_gate_status.pdf`, `figures/rh_plateau_demo.csv`, `figures/validation_gate_summary.csv`, `figures/cyclone_trace_reduced.csv`, `figures/gyaradax_cyclone_trace_reduced.csv`, `figures/gyaradax_cyclone_trace_comparison.csv`, `figures/gyaradax_cyclone_trace_production_control_smoke.csv`, `figures/gyaradax_cyclone_trace_production_control_smoke_comparison.csv`, `figures/gyaradax_cyclone_trace_production_control.csv`, `figures/gyaradax_cyclone_trace_production_control_comparison.csv`, `figures/gyaradax_cyclone_trace_production_control_gkw_cosine.csv`, `figures/gyaradax_cyclone_trace_production_control_gkw_cosine_comparison.csv`, `figures/gkw_simple_example_time_trace.csv`, `figures/gkw_cyclone_selected_ky_time_trace.csv`, `figures/gkw_cyclone_selected_ky_time_comparison.csv`, `figures/gkw_cyclone_parallel_phi_profile_comparison.csv`, `figures/gkw_igh_cyclone_selected_ky_time_comparison.csv`, `figures/gkw_igh_cyclone_selected_ky_time_trace.csv`, `figures/gkw_igh_cyclone_parallel_phi_profile_comparison.csv`, `figures/gkw_cosin2_cyclone_selected_ky_time_comparison.csv`, `figures/gkw_cosin2_cyclone_selected_ky_time_trace.csv`, `figures/gkw_cosin2_cyclone_parallel_phi_profile_comparison.csv`, `figures/gkw_cosin2_cyclone_gap_audit.csv`, `figures/gkw_cosin2_cyclone_velocity_slice_audit.csv`, `figures/gkw_cosin2_cyclone_velocity_slice_conventions.csv`, `figures/gkw_cosin2_cyclone_velocity_series_audit.csv`, `figures/gkw_cosin2_cyclone_velocity_series_variant_audit.csv`, `figures/gkw_cosin2_cyclone_vpar_odd_sign_audit.csv`, `figures/gkw_cosin2_cyclone_term_vii_field_convention_audit.csv`, `figures/gkw_cosin2_cyclone_velocity_phase_audit.csv`, `figures/cyclone_profile_operator_audit.csv`, `figures/cyclone_term_i_fortran_audit.csv`, `figures/cyclone_time_normalization_audit.csv`, `figures/cyclone_diagnostic_packing_audit.csv`, `figures/cyclone_matdat_matrix_audit.csv`, `figures/cyclone_coefficient_source_audit.csv`, `figures/cyclone_igh_arakawa_audit.csv`, `figures/cyclone_igh_arakawa_series_audit.csv`, `figures/cyclone_term_vii_mode_packing_audit.csv`, and `figures/cyclone_growth_diagnostic_convention_comparison.csv`: current reduced validation-gate and CBC trace result artifacts.
- `fixtures/gkw_cyclone_selected_ky_linear_input.dat`, `fixtures/gkw_cyclone_selected_ky_time.dat`, and `fixtures/gkw_cyclone_selected_ky_parallel_phi.dat`: matched native-GKW selected-`ky` linear input, compact time diagnostic, and parallel `|phi|^2` diagnostic.
- `fixtures/gkw_cyclone_selected_ky_cosin2_selected_state/`: sampled patched-GKW `cosin2` full selected-mode state dumps at steps 20, 800, and 1600.
- `fixtures/gkw_cyclone_selected_ky_cosin2_rhs_trace/`: sampled patched-GKW `cosin2` selected-mode dtim-scaled RHS/source action traces at steps 20, 800, and 1600.
- `fixtures/gkw_cyclone_selected_ky_cosin2_linear_input.dat`, `fixtures/gkw_cyclone_selected_ky_cosin2_time.dat`, and `fixtures/gkw_cyclone_selected_ky_cosin2_parallel_phi.dat`: patched, non-destructive GKW `cosin2` selected-`ky` input and raw diagnostics for the solver/Gyaradax `cosine2` profile.
- `fixtures/gkw_cyclone_selected_ky_cosin2_distr1.dat` through `fixtures/gkw_cyclone_selected_ky_cosin2_distr4.dat`: patched GKW final-output velocity-space slices for the selected-`ky` `cosin2` run.
- `fixtures/gkw_cyclone_selected_ky_cosin2_multitime_distr/`: compact patched GKW multi-time velocity-space slices at steps 20, 800, and 1600, plus selected `time.dat` rows.
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

- use the new GKW `distr*.dat` loader, solver-side peak-\(\phi\) slice
  diagnostic, and convention audit to localize the remaining
  evolved-distribution error; the current best simple variant is a
  \(v_\parallel\)-column reversal, not a transpose or phase/sign rotation,
- audit \(v_\parallel\)-odd dynamics/sign conventions in parallel streaming,
  parallel field drive, and the fused `igh` backend with explicit Fortran
  1-based indexing checks,
- use the completed Term VII, multi-window fused-`igh`, solver-side
  source-term trace, and GKW compact state-trace patch as guardrails against
  promoting diagnostic sign variants; the full selected-mode state dump is now
  available and leaves an OPEN phase-aligned state gap; the GKW and solver
  full selected-mode RHS/action traces are now comparable elementwise, and the
  GKW `KTHRHO/kthnorm` fix reduces the selected RHS gap to `2.24e-05`; the
  same-state replay reduces this further to `8.91e-07` and moves the residual
  to `igh_or_term_i`; the new source-level `igh` replay audit shows the solver
  fused action and reconstructed GKW `igh` source action match to roundoff, so
  the next target is GKW trace tagging/timing rather than solver coefficients,
- retain both `late_fit` and `late_mean_window` production-gate diagnostics
  until the GKW/Gyaradax selected-mode history gap is isolated,
- keep the now-passing production-control Cyclone selected-`ky` regression in
  place while the remaining selected-state trace gap is narrowed,
- keep the independent GX/VMEC GIST external eik-producer report/gate separate
  from the matched DESC/GX block-eik convention fixture.

Expected file changes:

- GKW/Gyaradax/solver parallel profile, source-term, or amplitude-history
  comparison report updates,
- selected-mode initialization or state-history diagnostics needed to explain
  the remaining full selected-mode state gap,
- patched GKW `linear_terms.F90`/`matdat.F90`/`exp_integration.F90` trace
  diagnostics needed to close the remaining `8.91e-07` selected RHS/action
  residual now localized to traced `igh_or_term_i`,
- regenerated matched GKW selected-state/RHS fixtures after the corrected `n2`
  trace loop is used,
- any future independently generated DESC/GX-specific eik fixture if an
  external runner becomes available,
- `scripts/prepare_gkw_cosine2_run.py`,
- `TODO.md`,
- `STATUS.md`

Expected tests:

- Cyclone selected-`ky` growth-rate tolerance test promoted from OPEN to PASS
  when the remaining physics gap is closed,
- existing matched DESC solver-produced geometry parity tolerance test retained,
- continued reduced DESC objective and gradient checks.

## Round Log

### 2026-06-03: Corrected Future GKW RHS Trace Matrix-Section Contract

- Audited the patched GKW RHS trace path against
  `exp_integration.F90::calculate_rhs`.
- Found one concrete trace-contract defect in the generated
  `stellarator_gk_rhs_trace_output`: it summed compressed matrix entries
  through `n4`, while explicit complex GKW RHS application uses only section
  `n2` for the distribution RHS.
- Updated `scripts/prepare_gkw_cosine2_run.py` so future RHS trace patches
  import `n2` and loop `do elem = 1, n2`.
- Kept the term-tag compression safeguards in place:
  `stellarator_gk_mat_term` is moved during heap/sift swaps, copied during
  compression, and included in the duplicate-merge predicate.
- Updated the generated patched-run README text to state that the RHS trace
  mirrors the explicit complex `calculate_rhs` matrix range.
- Updated `TODO.md`: the next step is to regenerate the matched GKW
  selected-state/RHS fixtures with this corrected trace patch, then rerun the
  same-state RHS replay and source-level `igh` audit.
- Verification run this round:
  - `uv run ruff check scripts/prepare_gkw_cosine2_run.py tests/test_gkw_cosine2_patch.py`
  - `uv run pytest tests/test_gkw_cosine2_patch.py::test_prepare_gkw_cosine2_run_can_patch_rhs_trace tests/test_gkw_cosine2_patch.py::test_rhs_trace_patches_are_idempotent -q`
  - `uv run pytest tests/test_gkw_cosine2_patch.py -q`
- Verification result:
  - focused patch tests: `2 passed in 0.35s`,
  - full cosine2 patch/fixture suite: `24 passed in 94.82s`.

### 2026-06-02: Added Source-Level Same-State `igh` Replay Audit

- Added `CycloneSameStateIghReplayAudit` and
  `run_cyclone_base_case_same_state_igh_replay_audit()`.
- The audit inserts each patched GKW selected-state snapshot into the solver's
  single-mode CBC setup, compares the patched GKW `igh_or_term_i` RHS action
  against the solver fused `gkw_igh_streaming_mirror`, and independently
  reconstructs `linear_terms.F90::igh` from the source-level Hamiltonian,
  `disp_par`, and `disp_vp` contributions.
- x64 fixture audit result:
  - `gkw_vs_solver_fused_igh=8.905291461911702e-07`,
  - `gkw_vs_source_fused_igh=8.905291461911714e-07`,
  - `solver_vs_source_fused_igh=1.6446404919551064e-19`,
  - `source_fused_split_error=2.7939250479261703e-20`,
  - `gkw_vs_source_hamiltonian_plus_disp_par=6.116326659535234e-06`,
  - `source_parallel_diffusion_max_norm=7.993717918861919e-05`,
  - `source_velocity_diffusion_max_norm=6.510400977692796e-06`.
- Interpretation: the implemented fused `igh` operator matches the
  source-level GKW reconstruction to roundoff. The remaining same-state
  `igh_or_term_i` residual is therefore in the patched GKW RHS trace path:
  term tagging, matrix/source-action accumulation, compression ordering, or
  `calculate_rhs` timing.
- Updated `TODO.md` so the next Immediate Next Round item is the patched GKW
  trace path, not fused `igh` coefficient construction.
- Verification run this round:
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_gkw_cosine2_patch.py`
  - `uv run pytest tests/test_gkw_cosine2_patch.py -q`
  - `uv run pytest -q`
  - `JAX_ENABLE_X64=1 uv run python -c "... run_cyclone_base_case_same_state_igh_replay_audit(...) ..."`
- Verification result:
  - focused fixture suite: `24 passed in 62.13s`,
  - full suite: `200 passed in 535.16s`.

### 2026-06-02: Added Same-State GKW RHS Replay Gate

- Added `run_cyclone_base_case_same_state_rhs_replay()` and
  `run_cyclone_base_case_same_state_rhs_replay_gate()`.
- The replay loads patched GKW `SelectedModeStateTrace` snapshots, inserts each
  selected `(kx,ky)` distribution into the solver's production-control CBC grid,
  evaluates solver term actions immediately, and compares against the matching
  GKW `GkwSelectedModeRhsTrace`.
- The replay can use either solver-computed `phi` or the dumped GKW `phi` as a
  diagnostic field source.
- x64 fixture replay result:
  - solver-field replay maximum residual: `8.905204720648363e-07`,
  - GKW-field replay maximum residual: `8.90520472127105e-07`,
  - dominant term: `igh_or_term_i=8.905204720648363e-07`,
  - `vdgradf=3.1370451207010515e-09`,
  - `vpgrphi=2.8711265195012825e-11`.
- Interpretation: same-state replay removes the previous `vdgradf` dominance
  from independently evolved state comparison; the next consistency target is
  the fused GKW `ltrapping_arakawa`/`igh` action timing and RHS trace
  construction.
- Verification run this round:
  - `uv run pytest tests/test_gkw_cosine2_patch.py::test_patched_cosin2_rhs_trace_replays_on_gkw_selected_state -q`
  - `JAX_ENABLE_X64=1 uv run python -c "... run_cyclone_base_case_same_state_rhs_replay_gate(...) ..."`

### 2026-06-02: Added Independent External Eik Producer Gate

- Added `ExternalEikProducerReport` plus
  `run_independent_external_eik_producer_report()` and
  `run_independent_external_eik_producer_gate()`.
- The new report records one geometry-contract error per external producer,
  the maximum suite error, pass/fail status, producer names, source paths, and
  the comparison grid size.
- Refactored `run_gx_gist_external_eik_suite_gate()` to use this generic
  report while preserving the existing GX/GIST validation-gate name and target
  metadata.
- The independent producer suite uses the local GX/VMEC GIST fixtures and
  explicitly excludes the matched DESC/GX block-`eik.out` fixture, which remains
  a separate DESC convention-parity check.
- Updated `TODO.md` and `main.tex` to record that external eik-producer
  coverage is in place for the available GX/VMEC GIST path, while future
  DESC-specific external runners can still strengthen production parity.
- Verification run this round:
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py`
  - `uv run pytest tests/test_benchmark_references.py::test_external_gist_eik_suite_gate_runs_multiple_stellarator_fixtures -q`
  - `uv run pytest tests/test_benchmark_references.py -q`
  - `uv run python examples/generate_validation_gate_figures.py`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

### 2026-06-02: Fixed GKW Internal `krho` Convention for Cyclone CBC

- Implemented the GKW s-alpha wave-number convention in the Cyclone benchmark
  setup: `mode.F90::kgrid` divides input `KTHRHO` by
  `kthnorm=q/(2*pi*eps)` before the linear terms use `krho`.
- For the current Cyclone input, `KTHRHO=0.5`, `q=1.4`, and `eps=0.19`, so
  the solver now uses internal `krho=0.4263590029871862` in the GKW-matched
  Fourier grid.
- Before the fix, the GKW-implied `vdgradf` frequency was an exact scalar
  multiple of the solver frequency with scale `0.852718005974373`, i.e.
  `1/kthnorm`; after the fix, the selected RHS/action comparison no longer
  prefers reversed-`vpar` or phase-aligned layouts.
- Post-fix selected RHS/action comparison against
  `fixtures/gkw_cyclone_selected_ky_cosin2_rhs_trace/`:
  - `passed=False` at strict tolerance `1e-8`,
  - best total layout: `direct`,
  - best term layout: `direct`,
  - max error `2.2379489422925073e-05`,
  - dominant per-term errors: `vdgradf=2.2379489422925073e-05`,
    `igh_or_term_i=1.1169352619622038e-06`,
    `ve_grad_fm=4.5614093015065964e-08`,
    `vd_grad_phi_fm=3.9597207251785146e-08`,
    `vpgrphi=4.9491617703205574e-09`.
- Added a production-control selected-`ky` regression at the matched
  48/32/8 GKW trace resolution. The `late_mean_window` gate now passes with
  observed growth `0.17800063460817828` against target `0.179`; the companion
  diagnostic `late_fit` is `0.18560606277298422`.
- Interpretation: the large `vdgradf` parity gap was a GKW internal-wave-number
  convention mismatch, not a Term VII field-drive convention. The remaining CBC
  production work is now the selected-state history gap and independent eik
  coverage, not the selected-`ky` growth gate.
- Verification run this round:
  - `uv run ruff check src/stellarator_gk/benchmarks.py tests/test_benchmark_references.py tests/test_gkw_cosine2_patch.py`
  - `uv run pytest tests/test_gkw_cosine2_patch.py::test_patched_cosin2_rhs_trace_fixture_compares_to_solver_snapshot tests/test_benchmark_references.py::test_production_cyclone_selected_ky_gate_passes_matched_gkw_control_resolution tests/test_benchmark_references.py::test_cyclone_term_vii_mode_packing_audit_matches_source_path -q`
  - `JAX_ENABLE_X64=1 uv run python -c "... compare_selected_mode_rhs_traces(...) ..."`
  - `JAX_ENABLE_X64=1 uv run python -c "... run_production_cyclone_base_case_gate(...) ..."`

### 2026-06-02: Added Solver Selected-Mode RHS/Action Trace Comparison

- Added public `SolverSelectedModeRhsTrace`,
  `SelectedModeRhsTraceComparisonReport`,
  `run_cyclone_base_case_selected_rhs_trace()`, and
  `compare_selected_mode_rhs_traces()`.
- The solver trace records `dt`-scaled selected-mode actions with shape
  `(n_time, 9, n_vpar, n_mu, n_z)` in the same GKW term buckets as the patched
  `stellarator_gk_rhs_trace_*.dat` files.
- The term buckets intentionally keep GKW-empty groups explicit:
  `untagged`, `trapdf`, `hyper_collision`, and `field_eq` remain available as
  zero-action checks in the collisionless production selected-`ky` setup.
- Production x64 comparison against
  `fixtures/gkw_cyclone_selected_ky_cosin2_rhs_trace/`:
  - `passed=False` at tolerance `1e-8`,
  - best total layout: `reverse_vpar_phase_aligned`,
  - best term layout: `reverse_vpar_phase_aligned`,
  - max error `4.466017187736114e-03`,
  - dominant per-term errors: `vdgradf=4.466017187736114e-03`,
    `ve_grad_fm=6.045520645652958e-04`,
    `igh_or_term_i=2.0712355663949134e-04`,
    `vpgrphi=8.175992645595e-05`.
- Interpretation: selected-state and selected-RHS traces both prefer the
  reversed-`vpar` diagnostic lens, but the largest remaining elementwise RHS
  gap now sits in the magnetic-drift `vdgradf` bucket rather than Term VII.
- Verification run this round:
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_gkw_cosine2_patch.py`
  - `uv run pytest tests/test_gkw_cosine2_patch.py -q`
  - `JAX_ENABLE_X64=1 uv run python -c "... compare_selected_mode_rhs_traces(...) ..."`

### 2026-06-02: Retained Dual Cyclone Growth Diagnostics

- Verified the production Cyclone gate still carries both growth diagnostics:
  - `late_fit`, the selected-mode least-squares log-amplitude fit,
  - `late_mean_window`, the GKW `time.dat`-style late-window mean.
- The existing `growth_diagnostic` selector and tests keep both diagnostics
  available while the selected-mode state/RHS parity gates are OPEN.
- Interpretation: no diagnostic convention is promoted or removed in this
  round; the next actionable production-CBC target remains the `vdgradf`
  magnetic-drift RHS/action mismatch identified by the selected-mode trace
  comparison.

### 2026-06-02: Added GKW Selected-Mode RHS/Source Trace

- Prepared a copied GKW tree with:
  - `uv run python scripts/prepare_gkw_cosine2_run.py --overwrite --rhs-trace --output-root /tmp/stellarator_gk_gkw_cosin2_rhs_trace`
- Built and ran the copied serial/no-FFT GKW tree with:
  - `make FC=gfortran FFLAGS="-fdefault-real-8 -O2" FFTLIB=nofft PARALLEL=nompi LDFLAGS=""`
  - `./gkw.x`
- The copied-source patch now:
  - tags GKW matrix entries by term in `matdat.F90`,
  - sets trace term IDs around `linear_terms.F90` source-term builders,
  - writes `stellarator_gk_rhs_trace_<step>.dat` from `exp_integration.F90`
    immediately after end-of-window `normalize(2,...)`.
- Added sampled fixtures:
  - `fixtures/gkw_cyclone_selected_ky_cosin2_rhs_trace/stellarator_gk_rhs_trace_00000020.dat`,
  - `fixtures/gkw_cyclone_selected_ky_cosin2_rhs_trace/stellarator_gk_rhs_trace_00000800.dat`,
  - `fixtures/gkw_cyclone_selected_ky_cosin2_rhs_trace/stellarator_gk_rhs_trace_00001600.dat`.
- Added public `GkwSelectedModeRhsTrace` and
  `load_gkw_selected_mode_rhs_trace`.
- GKW RHS/source trace result:
  - shape `(3, 9, 32, 8, 48)`,
  - steps `[20, 800, 1600]`,
  - internal term-sum error `9.550830415404176e-18`,
  - final dtim-scaled total-action norm `1.9404731497669497e-02`,
  - final dominant term norms: `vdgradf=1.912308487778263e-02`,
    `ve_grad_fm=2.6230193433310666e-03`,
    `igh_or_term_i=1.2506893514028245e-03`.
- Interpretation: the GKW-side term-action artifact is now ready; the next
  implementation target is solver full selected-mode term-action snapshots for
  elementwise comparison, since existing solver term norms are not enough.
- Verification run this round:
  - `uv run pytest tests/test_gkw_cosine2_patch.py -q`

### 2026-06-02: Captured Full Selected-Mode GKW State Dumps

- Prepared a copied GKW tree with:
  - `uv run python scripts/prepare_gkw_cosine2_run.py --overwrite --selected-state-dump --output-root /tmp/stellarator_gk_gkw_cosin2_selected_state`
- Built and ran the copied serial/no-FFT GKW tree with:
  - `make FC=gfortran FFLAGS="-fdefault-real-8 -O2" FFTLIB=nofft PARALLEL=nompi LDFLAGS=""`
  - `./gkw.x`
- The copied diagnostic patch produced 80
  `stellarator_gk_selected_state_<step>.dat` snapshots. Three production-control
  samples are now stored as fixtures:
  - `fixtures/gkw_cyclone_selected_ky_cosin2_selected_state/stellarator_gk_selected_state_00000020.dat`,
  - `fixtures/gkw_cyclone_selected_ky_cosin2_selected_state/stellarator_gk_selected_state_00000800.dat`,
  - `fixtures/gkw_cyclone_selected_ky_cosin2_selected_state/stellarator_gk_selected_state_00001600.dat`.
- Added public `SelectedModeStateTrace`,
  `load_gkw_selected_mode_state_trace`,
  `run_cyclone_base_case_selected_state_trace`, and
  `compare_selected_mode_state_traces`.
- Added `examples/compare_gkw_selected_state.py`, which writes:
  - `figures/gkw_cosin2_cyclone_selected_state_comparison.csv`.
- Selected-state comparison result:
  - OPEN at tolerance `2.0e-02`,
  - maximum snapshot-wise phase-aligned error: `5.845543963934988e-02`,
  - phase-aligned `phi(z)` error: `5.845543963934988e-02`,
  - phase-aligned full-state error: `4.780479264318512e-02`,
  - worst full-state relative \(L^2\) error: `0.3192978051120589` at step 800.
- Interpretation: direct layout remains best, scalar norms are already aligned,
  and a simple output phase/layout convention does not close the selected-mode
  state gap. The next target is a term-resolved GKW RHS/source trace at the same
  selected snapshots.
- Verification run this round:
  - `uv run python examples/compare_gkw_selected_state.py`

### 2026-06-02: Captured Matched GKW Compact State Trace

- Prepared a copied GKW tree with:
  - `uv run python scripts/prepare_gkw_cosine2_run.py --overwrite --state-trace --output-root /tmp/stellarator_gk_gkw_cosin2_state_trace`
- Built and ran the copied serial/no-FFT GKW tree with:
  - `make FC=gfortran FFLAGS="-fdefault-real-8 -O2" FFTLIB=nofft PARALLEL=nompi LDFLAGS=""`
  - `./gkw.x`
- Added the matched production-control fixture:
  - `fixtures/gkw_cyclone_selected_ky_cosin2_state_trace.dat`.
- Added `examples/compare_gkw_state_trace.py`, which writes:
  - `figures/gkw_cosin2_cyclone_state_trace_comparison.csv`.
- Added `snapshot_timing="post_normalization"` to
  `run_cyclone_base_case_source_term_trace` to match GKW's diagnostic order
  after `normalize(2,...)`.
- Compact state-trace comparison result:
  - PASS at tolerance `5.0e-03`,
  - maximum error: `3.3077755368903644e-03`,
  - time error: `7.016609515630989e-14`,
  - state-norm error: `3.3077755368903644e-03`,
  - field-norm error: `6.106226635438361e-16`.
- Interpretation: compact state and field norms are now aligned well enough
  that the remaining CBC gap needs a full selected-mode state dump or
  term-resolved GKW RHS/source trace.
- Verification run this round:
  - `uv run --extra dev python examples/compare_gkw_state_trace.py`

### 2026-06-02: Added GKW Compact State-Trace Patch Path

- Extended `scripts/prepare_gkw_cosine2_run.py` with `--state-trace`.
- The copied-source patch adds `stellarator_gk_state_trace_output` to
  `diagnostic.F90` and writes `stellarator_gk_state_trace.dat` with:
  - `step`,
  - `time`,
  - `state_norm`,
  - `phi_norm`.
- Added public `GkwStateTrace`, `load_gkw_state_trace`, and
  `compare_gkw_state_trace_to_source_term_trace`.
- The patch can be combined with the existing `--multi-time-distr` option and
  still leaves `relevant-codes/gkw/` untouched.
- The actual production-control GKW state-trace fixture is still open; this
  round added the generation and comparison contract.
- Verification run this round:
  - `uv run pytest tests/test_gkw_cosine2_patch.py -q`
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py scripts/prepare_gkw_cosine2_run.py tests/test_gkw_cosine2_patch.py`

### 2026-06-02: Added Solver-Side Cyclone Source-Term Trace

- No tracked changes were waiting to commit at the start of this round after:
  - `7f35779 Add multi-window Cyclone igh audit`.
- Added `CycloneSourceTermTrace`,
  `run_cyclone_base_case_source_term_trace`, and
  `write_cyclone_source_term_trace_csv`.
- Added `examples/audit_cyclone_source_term_trace.py`, which writes:
  - `figures/cyclone_source_term_trace.csv`.
- Production-control `cosine2`/`gkw_igh` source-term trace results for output
  windows 0, 1, 40, and 80:
  - maximum stored RHS reconstruction error: `0.0`,
  - final raw RHS norm: `9.143666705809655e-02`,
  - final raw magnetic-drift norm: `9.042892647805491e-02`,
  - final raw fused-`igh` norm: `4.486003543507541e-03`,
  - final selected-`ky` log-normalization: `-4.398714455712693`.
- Interpretation: the solver-side source-term decomposition now has a
  machine-checkable trace target. The remaining CBC action item is the matched
  GKW source/state/restart trace, not another unanchored sign variant.
- Verification run this round:
  - `uv run pytest tests/test_benchmark_references.py -k "source_term_trace"`
  - `uv run --extra dev python examples/audit_cyclone_source_term_trace.py`

### 2026-06-02: Added Multi-Window Igh Series Audit

- No tracked changes were waiting to commit at the start of this round after:
  - `7427c25 Add Term VII mode packing audit`.
- Added `CycloneIghArakawaSeriesAudit` and
  `run_cyclone_base_case_igh_arakawa_series_audit`.
- Added `examples/audit_cyclone_igh_arakawa_series.py`, which writes:
  - `figures/cyclone_igh_arakawa_series_audit.csv`.
- Production-control audit results for output windows 1, 40, and 80:
  - max fused-vs-separated profile deltas:
    `0.014010106905660624`, `0.0028004685541289983`,
    `0.0006152079285486119`,
  - relative deltas:
    `0.8985196896095599`, `0.15207297474358986`,
    `0.056947861475111494`,
  - worst sampled window: `1`,
  - target \(z\): `0.09375`, target index: `24`.
- Interpretation: the `igh` fused-vs-separated mismatch is strongest
  immediately and decreases over the sampled time history. The remaining
  GKW/solver distribution mismatch should now be debugged with matched
  state-history, source-term time traces, or restart/final-state dumps.
- Verification run this round:
  - `python -m py_compile src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_igh_arakawa_series.py`
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_igh_arakawa_series.py`
  - `uv run pytest tests/test_benchmark_references.py::test_cyclone_igh_arakawa_series_audit_samples_multiple_windows -q`
  - `uv run python examples/audit_cyclone_igh_arakawa_series.py`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

### 2026-06-02: Added Term VII Mode-Packing Source Audit

- Committed the previous multi-time velocity-series variant audit as:
  - `d55b250 Add Cyclone velocity series variant audit`.
- Added `CycloneTermVIIModePackingAudit` and
  `run_cyclone_base_case_term_vii_mode_packing_audit`.
- Added `examples/audit_cyclone_term_vii_mode_packing.py`, which writes:
  - `figures/cyclone_term_vii_mode_packing_audit.csv`.
- Production-control audit results at \(t=3.72\):
  - selected `ky` / GKW `krho`: `0.5` / `0.5`,
  - selected nonzonal single-chain maps: `ixplus=-1`, `ixminus=-1` in Python
    for GKW's open `0` maps,
  - direct field roundtrip error: `0.0`,
  - direct packed-field Term VII action error:
    `3.471140604459642e-18`,
  - conjugated packed-field Term VII delta:
    `0.028893793199993845`,
  - negated packed-field Term VII delta:
    `0.028925615674737656`.
- Interpretation: the selected positive-\(k_y\) Term VII source path is direct,
  not conjugated or sign-flipped. The remaining production gap should be pursued
  through cumulative state evolution and the fused GKW `ltrapping_arakawa`/`igh`
  path, not by promoting Term VII diagnostic sign variants.
- Verification run this round:
  - `python -m py_compile src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_term_vii_mode_packing.py`
  - `uv run pytest tests/test_benchmark_references.py::test_cyclone_term_vii_mode_packing_audit_matches_source_path -q`
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_term_vii_mode_packing.py`
  - `uv run python examples/audit_cyclone_term_vii_mode_packing.py`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

### 2026-06-02: Added Multi-Time Term VII/Field Variant Audit

- No tracked changes were waiting to commit at the start of this round after
  `d0079fb Add Cyclone velocity series audit`; pre-existing untracked external
  references and LaTeX artifacts were left untouched.
- Added `CycloneVelocitySpaceSliceSeriesVariantAudit`,
  `audit_cyclone_velocity_space_slice_series_variants`, and
  `run_cyclone_base_case_cosin2_velocity_series_variant_audit`.
- Added `examples/audit_cyclone_cosin2_velocity_series_variants.py`, which
  writes:
  - `figures/gkw_cosin2_cyclone_velocity_series_variant_audit.csv`.
- Production-control diagnostic results:
  - baseline direct max errors at steps 20, 800, and 1600:
    `3.990529105190601e-03`, `3.6712468463562305e-02`,
    `3.5348895748192916e-02`,
  - `flip_igh` final direct max:
    `17.078737087702983`,
  - best step-800 direct variant: `flip_term_vii_only`, with error
    `2.604522658732149e-02`,
  - best step-1600 direct variant: `conjugate_term_vii_only`, with error
    `1.596045557377426e-02`,
  - global field conjugation worsens the middle and final snapshots.
- Interpretation: the diagnostic variants improve but do not close the
  velocity-slice gap. The next narrowed source-level target is the selected-mode
  Term VII \(k_y\)/Fourier sign and field-packing path through `mode.F90`,
  `dist.F90::get_phi`, `linear_terms.F90::vpgrphi_3_newbc`, and the real/complex
  storage convention.
- Verification run this round:
  - `python -m py_compile src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_cosin2_velocity_series_variants.py`
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_cosin2_velocity_series_variants.py`
  - `uv run pytest tests/test_benchmark_references.py::test_cyclone_velocity_space_slice_series_variant_audit_accepts_matched_series tests/test_benchmark_references.py::test_cosin2_velocity_series_variant_audit_runner_accepts_matched_reduced_reference -q`
  - `uv run python examples/audit_cyclone_cosin2_velocity_series_variants.py`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  - `uv run ruff check src tests examples scripts`
  - `uv run pytest -q` (`178 passed in 178.11s`)

### 2026-06-02: Added Multi-Time Velocity-Slice Series Audit

- Committed the previous multi-time GKW diagnostic-output tranche as:
  - `11bc27a Add multi-time GKW velocity slices`.
- Added compact patched-GKW multi-time fixtures under:
  - `fixtures/gkw_cyclone_selected_ky_cosin2_multitime_distr/`.
- The fixture stores steps 20, 800, and 1600 from the real patched serial/no-FFT
  `cosin2` GKW run, plus matching selected `time.dat` rows.
- Added `CycloneVelocitySpaceSliceSeriesAudit`,
  `audit_cyclone_velocity_space_slice_series`,
  `run_cyclone_base_case_velocity_space_slice_series`, and
  `run_cyclone_base_case_cosin2_velocity_series_audit`.
- Added `examples/audit_cyclone_cosin2_velocity_series.py`, which writes:
  - `figures/gkw_cosin2_cyclone_velocity_series_audit.csv`.
- Production-control audit results:
  - step 20 direct max / \(L^2\) / relative \(L^2\):
    `3.990529105190601e-03` / `2.3429816924998466e-03` /
    `2.4960628711028654e-01`,
  - step 800 direct max / \(L^2\) / relative \(L^2\):
    `3.6712468463562305e-02` / `1.3059709271108258e-02` /
    `5.287496799802696e-01`,
  - step 1600 direct max / \(L^2\) / relative \(L^2\):
    `3.5348895748192916e-02` / `8.45562568878207e-03` /
    `3.78493904183015e-01`,
  - best simple layout: direct at step 20, then
    `reverse_vpar_columns:identity` at steps 800 and 1600,
  - maximum best-layout error over the sampled windows:
    `2.9423527841286302e-02`.
- Interpretation: the evolved-distribution mismatch is not just a final
  peak-\(\phi\) output artifact. It is small but already present at the first
  diagnostic window and grows strongly by mid-run, so the next narrowed target
  is the cumulative step-20-to-step-800 evolution through `gkw_igh`, Term VII,
  field packing, and \(k_y\)/Fourier sign conventions.
- Verification run this round:
  - `python -m py_compile src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_cosin2_velocity_series.py`
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_cosin2_velocity_series.py`
  - `uv run pytest tests/test_benchmark_references.py::test_gkw_velocity_space_slice_series_loader_reads_matched_cosin2_fixture tests/test_benchmark_references.py::test_cyclone_velocity_space_slice_series_audit_accepts_matched_series tests/test_benchmark_references.py::test_cosin2_velocity_series_audit_runner_accepts_matched_reduced_reference tests/test_benchmark_references.py::test_cosin2_velocity_phase_audit_runner_accepts_matched_reduced_reference -q`
  - `uv run python examples/audit_cyclone_cosin2_velocity_series.py`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  - `uv run ruff check src tests examples scripts`
  - `uv run pytest -q`
  - `git diff --check`
- Verification results:
  - focused series/phase tests: 4 passed,
  - full pytest suite: 176 passed,
  - focused and full ruff: all checks passed,
  - production example completed and wrote the CSV above,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - whitespace check passed.

### 2026-06-02: Added Multi-Time GKW Velocity-Slice Output

- Committed the previous selected-`ky` velocity/phase audit tranche as:
  - `5ee9501 Add Cyclone velocity and Term VII audits`,
  - `9bd534d Add Cyclone velocity phase audit`.
- Extended `scripts/prepare_gkw_cosine2_run.py` with an optional
  `--multi-time-distr` mode. The patch is still non-destructive: it edits only
  the copied GKW tree and leaves `relevant-codes/gkw/` unchanged.
- The copied `diagnostic.F90` patch:
  - imports `ntotstep` in `write_output`,
  - calls `velocity_space_output(ntotstep)` at every normal diagnostic output,
  - changes `velocity_space_output` to accept an optional snapshot index,
  - writes suffixed `distr*_<ntotstep>.dat` files while preserving the original
    unsuffixed final `distr*.dat` output path.
- Added `GkwVelocitySpaceSliceSeries` and
  `load_gkw_velocity_space_slice_series`, exported publicly from
  `stellarator_gk`.
- Added unit coverage in `tests/test_gkw_cosine2_patch.py` for:
  - default non-multi-time behavior,
  - multi-time patch insertion,
  - patch idempotence,
  - loading complete suffixed `distr1`--`distr4` snapshot groups.
- Built and ran a real patched serial/no-FFT selected-`ky` `cosin2` GKW tree:
  - scratch tree:
    `/private/tmp/stellarator_gk_gkw_cosin2_multitime_v2`,
  - emitted suffixed snapshot count: `320`,
  - snapshot range: `00000020` through `00001600`,
  - loader shape: `(80, 8, 32)`,
  - loaded time range: `0.06` through `4.8`,
  - final snapshot index: `1600`,
  - final GKW-reported growth remained the known patched-`cosin2` value
    `0.188741`.
- Interpretation: the final peak-\(\phi\) `distr*.dat` slice is no longer the
  only available distribution diagnostic. The next narrowed target is a
  multi-window solver/GKW velocity-slice comparison to decide whether the
  mismatch appears immediately, accumulates during time evolution, or is tied
  to the final peak-\(\phi\) slice location.
- Verification run this round:
  - `python -m py_compile scripts/prepare_gkw_cosine2_run.py src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_gkw_cosine2_patch.py`
  - `uv run ruff check scripts/prepare_gkw_cosine2_run.py src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_gkw_cosine2_patch.py`
  - `uv run pytest tests/test_gkw_cosine2_patch.py -q`
  - `python scripts/prepare_gkw_cosine2_run.py --output-root /tmp/stellarator_gk_gkw_cosin2_multitime_v2 --multi-time-distr`
  - `make FC=gfortran FFLAGS="-fdefault-real-8 -O2" FFTLIB=nofft PARALLEL=nompi LDFLAGS=""`
  - `./gkw.x`
  - `uv run python -c "from stellarator_gk import load_gkw_velocity_space_slice_series; ..."`
  - `uv run pytest tests/test_benchmark_references.py::test_gkw_velocity_space_slice_loader_reads_distr_files tests/test_benchmark_references.py::test_gkw_velocity_space_slice_loader_reads_matched_cosin2_fixture -q`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  - `uv run pytest -q`
  - `git diff --check`
- Verification results:
  - focused patch/loader tests: 7 passed,
  - focused existing final-slice loader tests: 2 passed,
  - full pytest suite: 173 passed,
  - focused ruff: all checks passed,
  - patched GKW build succeeded,
  - patched GKW run completed successfully,
  - loader read the full 80-window suffixed snapshot series,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - whitespace check passed.

### 2026-06-01: Added Velocity-Slice Phase Audit

- Committed the previous selected-`ky` velocity and Term VII audit tranche as:
  - `5ee9501 Add Cyclone velocity and Term VII audits`.
- Added `VelocitySlicePhaseAudit`,
  `audit_velocity_space_slice_phase_alignment`, and
  `run_cyclone_base_case_cosin2_velocity_phase_audit`.
- Added `examples/audit_cyclone_cosin2_velocity_phase.py`, which writes:
  - `figures/gkw_cosin2_cyclone_velocity_phase_audit.csv`.
- Source inspection result:
  - `linear_terms.F90::vpgrphi_3_newbc` inserts real scalar coefficients into
    the `iphi` column,
  - `linear_terms.F90::add_element` maps those coefficients directly to the
    connected `iphi` index,
  - `matdat.F90::put_element` stores `mat_elem` unchanged in both `complex`
    and `complex-real` modes,
  - `dist.F90::get_phi` copies `fdis(indx(...,iphi))` directly,
  - `diagnostic.F90::velocity_space_output` writes
    `fdisi(...)*intmu*intvp/phi`, not division by `conjg(phi)`.
- Production-control x64 audit result:
  - best unit-phase variant: `reverse_vpar_columns:identity`,
  - best unit-phase max / \(L^2\) / relative \(L^2\):
    `0.018885582880959564` / `0.005508247236254402` /
    `0.24656223896253462`,
  - direct unit-phase max / \(L^2\) / relative \(L^2\):
    `0.033230903073356986` / `0.008423993122181344` /
    `0.37707795531385896`,
  - best unconstrained complex-scale variant: `reverse_vpar_columns:identity`,
  - best scaled max / \(L^2\) / relative \(L^2\):
    `0.017952545317588844` / `0.005189532875988546` /
    `0.23229582663822185`.
- Interpretation: a pure eigenfunction/output phase or amplitude-phase scale
  does not explain the final `distr*.dat` velocity-slice mismatch. The
  remaining discriminator is \(k_y\)/Fourier sign tracing and/or a multi-time
  or full normalized GKW state dump.
- Verification run this round:
  - `python -m py_compile src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_cosin2_velocity_phase.py`
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_cosin2_velocity_phase.py`
  - `uv run pytest tests/test_benchmark_references.py::test_velocity_space_slice_phase_audit_detects_global_phase tests/test_benchmark_references.py::test_cosin2_velocity_phase_audit_runner_accepts_matched_reduced_reference -q`
  - `JAX_ENABLE_X64=1 uv run python examples/audit_cyclone_cosin2_velocity_phase.py`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  - `uv run pytest -q` (`170 passed in 171.18s`)
  - `git diff --check`
- Next action:
  add either a GKW multi-time `distr*.dat`/state diagnostic or a Fourier-sign
  audit that compares \(k_y\), complex conjugation, and mode-label conventions
  before promoting any Term VII conjugation experiment to a code change.

### 2026-06-01: Added Term VII Field-Convention Audit

- Added `CycloneTermVIIFieldConventionAudit` and
  `run_cyclone_base_case_cosin2_term_vii_field_convention_audit`.
- Added `examples/audit_cyclone_cosin2_term_vii_field_conventions.py`, which
  writes:
  - `figures/gkw_cosin2_cyclone_term_vii_field_convention_audit.csv`.
- The audit keeps the same patched-GKW `cosin2`, selected-`ky`,
  production-control setup as the velocity-slice and odd-sign audits, but
  varies the `phi` convention supplied to Term V, Term VII, and Term VIII.
- Production-control x64 audit result:
  - baseline direct max / \(L^2\) / relative \(L^2\):
    `0.035348895748192916` / `0.008455625688782069` /
    `0.378493904183015`,
  - global sign direct max: `0.12796120197506905`,
  - global conjugation direct max: `0.07481363070580742`,
  - global negative-conjugation direct max: `0.11470706992462161`,
  - Term VII-only sign direct max: `0.02010162421775191`,
  - Term VII-only conjugation direct max / \(L^2\) / relative \(L^2\):
    `0.01596045557377426` / `0.005113660917074626` /
    `0.2288996174155926`,
  - Term VII-only negative-conjugation direct max:
    `0.016980699849443844`.
- Interpretation: a global field-variable sign/conjugation convention is ruled
  out. The strongest diagnostic signal is local to Term VII and points to a
  complex phase or conjugation convention in `vpgrphi_3_newbc` insertion or
  output normalization, not to a broad `phi` sign change. No physics sign or
  conjugation change has been adopted yet.
- Verification run this round:
  - `python -m py_compile src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_cosin2_term_vii_field_conventions.py`
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_cosin2_term_vii_field_conventions.py`
  - `uv run pytest tests/test_benchmark_references.py::test_cosin2_term_vii_field_convention_audit_runs_reduced_against_matched_reference -q`
  - `JAX_ENABLE_X64=1 uv run python examples/audit_cyclone_cosin2_term_vii_field_conventions.py`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  - `uv run pytest -q` (`168 passed in 193.54s`)
  - `git diff --check`
- Next action:
  inspect the Term VII complex phase path through `linear_terms.F90::add_element`,
  `matdat.F90`, complex-real matrix storage, \(k_y\)/Fourier signs, and
  `dist.F90::get_phi`; add multi-time or restart-state velocity diagnostics if
  the final peak-\(\phi\) slice remains ambiguous.

### 2026-06-01: Added Odd-\(v_\parallel\) RHS Sign Audit

- Added `CycloneVparOddSignAudit` and
  `run_cyclone_base_case_cosin2_vpar_odd_sign_audit`.
- Added `examples/audit_cyclone_cosin2_vpar_odd_signs.py`, which writes:
  - `figures/gkw_cosin2_cyclone_vpar_odd_sign_audit.csv`.
- The audit keeps GKW's Fortran velocity ordering explicit:
  `k=1,\ldots,nvpar` and `vpgr=-vpmax+(k-0.5)*dvp`, while testing controlled
  sign flips of the fused `linear_terms.F90::igh` Term I/IV block and the
  `linear_terms.F90::vpgrphi_3_newbc` Term VII parallel field-drive block.
- Production-control x64 audit result:
  - baseline direct max / \(L^2\) / relative \(L^2\):
    `0.035348895748192916` / `0.00845562568878207` /
    `0.37849390418301498`,
  - baseline best simple layout: `reverse_vpar_columns:identity`, with max
    `0.018543579974657488`,
  - `flip_igh` direct max: `17.078737087702983`,
  - `flip_parallel_field_drive` direct max / \(L^2\) / relative \(L^2\):
    `0.02010162421775191` / `0.006012189669100319` /
    `0.2691199000880014`,
  - `flip_parallel_field_drive` best layout:
    `direct_mu_rows_vpar_columns:identity`,
  - `flip_igh_and_parallel_field_drive` direct max: `17.078063622294973`.
- Interpretation: a global fused `igh` sign reversal is ruled out. A diagnostic
  flip of only Term VII improves the direct velocity slice and removes the
  post-processing preference for \(v_\parallel\)-column reversal, so the next
  narrowed target is the true Term VII source/matrix/field-variable convention
  through `linear_terms.F90`, `matdat.F90`, and `dist.F90::get_phi`.
- Verification run this round:
  - `python -m py_compile src/stellarator_gk/benchmarks.py tests/test_benchmark_references.py`
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py`
  - `uv run pytest tests/test_benchmark_references.py::test_cosin2_vpar_odd_sign_audit_runs_reduced_against_matched_reference -q`
  - `python -m py_compile examples/audit_cyclone_cosin2_vpar_odd_signs.py`
  - `uv run ruff check examples/audit_cyclone_cosin2_vpar_odd_signs.py`
  - `JAX_ENABLE_X64=1 uv run python examples/audit_cyclone_cosin2_vpar_odd_signs.py`
  - `uv run pytest tests/test_benchmark_references.py::test_cosin2_vpar_odd_sign_audit_runs_reduced_against_matched_reference tests/test_benchmark_references.py::test_cosin2_velocity_convention_audit_runner_accepts_matched_reduced_fixtures tests/test_benchmark_references.py::test_cosin2_velocity_slice_audit_runner_accepts_matched_reduced_fixtures -q`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  - `uv run pytest -q`
  - `git diff --check`
- Verification results:
  - focused odd-sign audit test: 1 passed,
  - focused velocity/convention/odd-sign tests: 3 passed,
  - full pytest suite: 167 passed,
  - focused ruff: all checks passed,
  - production x64 audit example completed and wrote the CSV above,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - whitespace check passed.

### 2026-06-01: Added GKW `distr*.dat` Velocity-Slice Audit

- Committed the previous selected-`ky` gap-audit tranche as:
  - `d9fa9f9 Add Cyclone cosin2 gap audit`.
- Copied the patched GKW `cosin2` final-output velocity diagnostics into:
  - `fixtures/gkw_cyclone_selected_ky_cosin2_distr1.dat`,
  - `fixtures/gkw_cyclone_selected_ky_cosin2_distr2.dat`,
  - `fixtures/gkw_cyclone_selected_ky_cosin2_distr3.dat`,
  - `fixtures/gkw_cyclone_selected_ky_cosin2_distr4.dat`.
- Added `GkwVelocitySpaceSlice`, `CycloneVelocitySpaceSliceAudit`,
  `load_gkw_velocity_space_slice`, `run_cyclone_base_case_velocity_space_slice`,
  `audit_cyclone_velocity_space_slice`, and
  `run_cyclone_base_case_cosin2_velocity_slice_audit`.
- Added `examples/audit_cyclone_cosin2_velocity_slice.py`, which writes:
  - `figures/gkw_cosin2_cyclone_velocity_slice_audit.csv`.
- Production-control audit results:
  - `vpar_error`: `5.0000000000105516e-05`,
  - `vperp_error`: `4.664794806363837e-05`,
  - `time_error`: `0.0`,
  - `real_max_abs_error`: `0.032260602362433624`,
  - `imag_max_abs_error`: `0.021464898760242965`,
  - `complex_max_abs_error`: `0.035348895748192916`,
  - `complex_l2_error`: `0.00845562568878207`,
  - `complex_relative_l2_error`: `0.378493904183015`,
  - `observed_l2_norm`: `0.023575570972117522`,
  - `reference_l2_norm`: `0.022340189882406877`.
- Added `VelocitySliceConventionAudit`,
  `audit_velocity_space_slice_conventions`, and
  `run_cyclone_base_case_cosin2_velocity_convention_audit`.
- The convention audit explicitly respects GKW's Fortran output:
  `global_vpar_mu(n_vpar,n_mu)` is written by `output_slice_2d` with rows over
  the second index and columns over the first. It then tests
  \(v_\parallel\)-column reversal, `mu`-row reversal, one-cell axis rolls,
  C/F-order flatten/reshape variants, conjugation, sign flips, and \(\pm i\)
  phase rotations.
- Production-control convention audit result:
  - best variant: `reverse_vpar_columns:identity`,
  - best max error: `0.018543579974657488`,
  - best \(L^2\) error: `0.005533740922357686`,
  - direct max error: `0.035348895748192916`,
  - direct \(L^2\) error: `0.00845562568878207`.
- Even/odd \(v_\parallel\) decomposition:
  - even max / \(L^2\): `0.01647851294172267` / `0.005197749203708426`,
  - odd same-sign max / \(L^2\): `0.02430342389518037` / `0.006669408444842744`,
  - odd opposite-sign max / \(L^2\): `0.00606276656656781` / `0.0018988659276327384`.
- Interpretation: the remaining final-slice mismatch is dominated by a
  \(v_\parallel\)-order/sign issue, not a Fortran row/column transpose,
  one-based off-by-one shift, `mu` ordering, or simple complex phase/sign
  convention. The next narrowed target is the \(v_\parallel\)-odd dynamics in
  parallel streaming, parallel field drive, and the fused
  `ltrapping_arakawa`/`igh` backend.
- Verification run this round:
  - `uv run pytest tests/test_benchmark_references.py::test_gkw_velocity_space_slice_loader_reads_distr_files tests/test_benchmark_references.py::test_gkw_velocity_space_slice_loader_reads_matched_cosin2_fixture tests/test_benchmark_references.py::test_cyclone_velocity_space_slice_audit_accepts_matched_slice tests/test_benchmark_references.py::test_cosin2_velocity_slice_audit_runner_accepts_matched_reduced_fixtures -q`
  - `uv run pytest tests/test_benchmark_references.py::test_velocity_space_slice_convention_audit_keeps_direct_baseline tests/test_benchmark_references.py::test_velocity_space_slice_convention_audit_detects_one_based_axis_shift tests/test_benchmark_references.py::test_cosin2_velocity_convention_audit_runner_accepts_matched_reduced_fixtures -q`
  - `uv run python examples/audit_cyclone_cosin2_velocity_slice.py`
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_cosin2_velocity_slice.py`
  - `python -m py_compile src/stellarator_gk/benchmarks.py examples/audit_cyclone_cosin2_velocity_slice.py`
  - `uv run ruff check src tests examples scripts`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  - `uv run pytest -q`
  - `git diff --check`
- Verification results:
  - focused velocity-slice tests: 4 passed,
  - focused convention tests: 3 passed,
  - full pytest suite: 163 passed,
  - focused ruff: all checks passed,
  - full ruff: all checks passed,
  - production audit example completed and wrote the CSV above,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - whitespace check passed.

### 2026-06-01: Added Patched-`cosin2` Selected-`ky` Gap Audit

- Committed the previous `gkw_igh`/patched-GKW tranche as:
  - `69821dc Add GKW cosin2 Cyclone reference path`.
- Added `CycloneSelectedKyGapAudit`,
  `audit_cyclone_selected_ky_gap`, and
  `run_cyclone_base_case_cosin2_gap_audit`.
- Added `examples/audit_cyclone_cosin2_gap.py`, which writes:
  - `figures/gkw_cosin2_cyclone_gap_audit.csv`.
- The audit aligns solver post-window samples with patched GKW `cosin2`
  `time.dat`/`parallel_phi.dat` samples, then reports late-fit growth,
  late-window mean growth, final-window growth, profile-shape error,
  worst `(t,z)`, and total-power normalization.
- Production-control audit results:
  - solver late-fit growth: `0.1648622859363632`,
  - patched GKW late-fit growth: `0.18741518752345235`,
  - late-fit delta: `0.02255290158708914`,
  - late-window mean delta: `0.021761822842616324`,
  - final-window growth delta: `0.049288860641875765`,
  - maximum row-normalized profile error: `0.029625447419939166`,
  - worst profile point: \(t=3.78\), \(z=-0.09375\),
  - total-power ratio mean: `0.9999999827218208`,
  - maximum total-power deviation: `1.953378815588991e-06`.
- Interpretation: the combined audit confirms that the remaining selected-`ky`
  gap is not caused by the native-vs-`cosine2` initialization convention, field
  output ordering, or total-power normalization. The next narrowed target is
  the evolved velocity-space/distribution state, using GKW `distr*.dat` slices
  if sufficient or a non-destructive final-state/restart dump otherwise.
- Verification run this round:
  - `uv run pytest tests/test_benchmark_references.py::test_cyclone_selected_ky_gap_audit_aligns_post_window_samples tests/test_benchmark_references.py::test_cosin2_gap_audit_runner_accepts_matched_reduced_fixtures -q`
  - `uv run ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_cosin2_gap.py`
  - `python -m py_compile src/stellarator_gk/benchmarks.py examples/audit_cyclone_cosin2_gap.py`
  - `uv run python examples/audit_cyclone_cosin2_gap.py`
  - `uv run ruff check src tests examples scripts`
  - `uv run pytest -q`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  - `git diff --check`
- Verification results:
  - focused gap-audit tests: 2 passed,
  - full pytest suite: 159 passed,
  - focused and full ruff: all checks passed,
  - production audit example completed and wrote the CSV above.
  - `main.tex` built successfully with existing underfull-box warnings only,
  - whitespace check passed.

### 2026-06-01: Added Non-Destructive GKW `cosin2` Reference Path

- Added `scripts/prepare_gkw_cosine2_run.py`. The script copies
  `relevant-codes/gkw/` to a scratch output directory, patches only the copied
  `src/init.f90`, adds `case('cosin2')`, and writes a matched
  production-control `input.dat`.
- The patched branch implements the solver/Gyaradax `cosine2` profile
  \(1+\cos(2\pi s)\). The branch name is `cosin2` because GKW's
  `components.F90` declares `finit` as `character(len = 6)`.
- Added `tests/test_gkw_cosine2_patch.py` covering:
  - non-destructive source copying,
  - idempotent insertion of the `cosin2` branch,
  - rejection of inputs without a native `finit='cosine'` selector,
  - loading the stored patched GKW `cosin2` time/profile fixtures.
- Prepared, built, and ran the patched local serial/no-FFT GKW tree in:
  - `/tmp/stellarator_gk_gkw_cosin2_smoke_20260601_1`.
- Stored the raw patched GKW artifacts as:
  - `fixtures/gkw_cyclone_selected_ky_cosin2_linear_input.dat`,
  - `fixtures/gkw_cyclone_selected_ky_cosin2_time.dat`,
  - `fixtures/gkw_cyclone_selected_ky_cosin2_parallel_phi.dat`.
- Wrote the solver/GKW `cosin2` comparison artifacts:
  - `figures/gkw_cosin2_cyclone_selected_ky_time_comparison.csv`,
  - `figures/gkw_cosin2_cyclone_selected_ky_time_trace.csv`,
  - `figures/gkw_cosin2_cyclone_parallel_phi_profile_comparison.csv`.
- Results against the patched GKW `cosin2` fixtures:
  - late-fit growth: solver `0.16389932979797434`, GKW
    `0.18741518752345235`,
  - late-window mean growth: solver `0.15623725215738368`, GKW
    `0.177999075`,
  - final-window growth: solver `0.13945213935812423`, GKW `0.188741`,
  - row-normalized `parallel_phi.dat` maximum profile error:
    `2.9625447419939166e-02`,
  - worst signed profile row: \(t=3.78\), \(z=-0.09375\), solver
    `0.35698553011960743` versus GKW `0.32736008269966826`.
- Interpretation: the explicit `cosine2`/`cosin2` initialization comparison
  improves the profile mismatch relative to native `finit='cosine'`, but the
  growth and central profile-width gap remain open. The next narrowed target
  is a remaining production GKW time-history convention rather than
  initialization.
- Verification run this round:
  - `uv run pytest tests/test_gkw_cosine2_patch.py -q`
  - `uv run ruff check scripts/prepare_gkw_cosine2_run.py tests/test_gkw_cosine2_patch.py`
  - `python -m py_compile scripts/prepare_gkw_cosine2_run.py`
  - `uv run python scripts/prepare_gkw_cosine2_run.py --output-root /tmp/stellarator_gk_gkw_cosin2_smoke_20260601_1`
  - `make -C /tmp/stellarator_gk_gkw_cosin2_smoke_20260601_1 FC=gfortran FFLAGS="-fdefault-real-8 -O2" FFTLIB=nofft PARALLEL=nompi LDFLAGS=`
  - `./gkw.x` from `/tmp/stellarator_gk_gkw_cosin2_smoke_20260601_1`
  - `uv run python examples/compare_gkw_igh_cyclone_growth.py --gkw-time fixtures/gkw_cyclone_selected_ky_cosin2_time.dat --initial-profile cosine2 --output figures/gkw_cosin2_cyclone_selected_ky_time_comparison.csv --trace-output figures/gkw_cosin2_cyclone_selected_ky_time_trace.csv`
  - `uv run python examples/compare_gkw_parallel_phi_profile.py --gkw-parallel-phi fixtures/gkw_cyclone_selected_ky_cosin2_parallel_phi.dat --gkw-time fixtures/gkw_cyclone_selected_ky_cosin2_time.dat --initial-profile cosine2 --parallel-derivative-model gkw_igh --output figures/gkw_cosin2_cyclone_parallel_phi_profile_comparison.csv`
  - `uv run ruff check src tests examples scripts`
  - `uv run pytest -q`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  - `git diff --check`
- Verification results:
  - focused patch tests: 4 passed,
  - full pytest suite: 157 passed,
  - focused and full ruff: all checks passed,
  - patched GKW build and run completed successfully,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - whitespace check passed.

### 2026-06-01: Reran Production Cyclone With Fused `gkw_igh`

- Added `velocity_recurrence_rate` controls to the production Cyclone growth,
  trace, and parallel-profile entry points. The default remains zero for
  separated `gkw_upwind` runs and becomes GKW's `disp_vp=0.2` for
  `parallel_derivative_model="gkw_igh"`.
- Added `examples/compare_gkw_igh_cyclone_growth.py`, which writes:
  - `figures/gkw_igh_cyclone_selected_ky_time_comparison.csv`,
  - `figures/gkw_igh_cyclone_selected_ky_time_trace.csv`.
- Extended `examples/compare_gkw_parallel_phi_profile.py` with
  `--parallel-derivative-model` and `--velocity-recurrence-rate`, then wrote:
  - `figures/gkw_igh_cyclone_parallel_phi_profile_comparison.csv`.
- Production-control `gkw_igh` rerun settings:
  - \(N_z=48\), \(N_{v_\parallel}=32\), \(N_\mu=8\),
  - 20 RK4 steps per diagnostic window,
  - 80 windows,
  - GKW-native `finit='cosine'`,
  - GKW-unweighted normalization,
  - `disp_par=1`, `disp_vp=0.2`.
- Results:
  - direct production gate late-fit growth: `0.1644648840181204`
    versus matched GKW reconstructed-amplitude late fit
    `0.18853144053590817`,
  - direct production gate late-window mean growth:
    `0.15473860600968004` versus matched GKW `0.180407525`,
  - final-window growth: `0.147643262940426` versus matched GKW `0.210113`,
  - row-normalized `parallel_phi.dat` profile maximum error:
    `3.284446487881976e-02`,
  - best circular-shift error: `3.284446487881976e-02` with best shift zero,
  - worst signed profile row in the new comparison:
    \(t=3.66\), \(z=-0.09375\), solver `0.35625058334280435`
    versus GKW `0.3234061184639846`.
- Interpretation: promoting the source-reconstructed `igh` operator into the
  residual did not close the production-control Cyclone selected-`ky`
  growth/profile gap. The next narrowing target is now a non-destructive GKW
  `cosine2` patch or restart/state-injection path, so the matched GKW run can
  start from the same profile family used by the solver/Gyaradax default.
- Verification run this round:
  - `uv run pytest tests/test_benchmark_references.py::test_production_cyclone_gate_supports_gkw_igh_backend tests/test_benchmark_references.py::test_cyclone_parallel_phi_trace_records_gkw_style_profiles -q`
  - `uv run python examples/compare_gkw_igh_cyclone_growth.py`
  - `uv run python examples/compare_gkw_parallel_phi_profile.py --parallel-derivative-model gkw_igh --output figures/gkw_igh_cyclone_parallel_phi_profile_comparison.csv`
  - direct production-gate command for `late_fit` and `late_mean_window` with
    `parallel_derivative_model="gkw_igh"`
  - `uv run ruff check src/stellarator_gk/benchmarks.py tests/test_benchmark_references.py examples/compare_gkw_parallel_phi_profile.py examples/compare_gkw_igh_cyclone_growth.py`
  - `uv run ruff check src tests examples scripts`
  - `uv run pytest -q`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Verification results:
  - focused tests: 2 passed,
  - full pytest suite: 153 passed,
  - focused and full ruff: all checks passed,
  - `main.tex` built successfully with existing underfull-box warnings only,
  - production-control comparison scripts completed and wrote the CSV artifacts
    above.

### 2026-06-01: Implemented Optional GKW Fused `igh` RHS Backend

- Committed the previous `igh` audit tranche as:
  - `4e6a6d8 Add Cyclone igh Arakawa audit`.
- Added `GKWArakawaIghStencil`, `build_gkw_igh_stencil`, and
  `gkw_igh_streaming_mirror`.
- Added `parallel_derivative_model="gkw_igh"` to the linear RHS precompute and
  residual assembly. In this backend, GKW's fused `linear_terms.F90::igh`
  action replaces the separated parallel-streaming, mirror-force,
  `disp_par`, and `disp_vp` pieces; Term VII still uses the GKW upwind field
  derivative.
- The fused backend precomputes source-shift weights for
  \(\Delta v_\parallel,\Delta z\in[-2,2]\), including the combined Term I/IV
  Hamiltonian stencil and in-operator `disp_par`/`disp_vp` fourth-difference
  diffusion on the GKW finite-difference fallback grid.
- Added a direct regression test comparing the matrix-free backend against the
  source-reconstructed Fortran-style `igh` action, including a JIT call.
- Fixed the Cyclone setup helper so `velocity_recurrence_rate` can be passed to
  the `igh` audit/backend path without changing existing GKW-upwind call sites.
- Updated `TODO.md` and `main.tex` to record that the fused backend exists and
  that the next step is a production selected-`ky` growth/profile rerun with
  `parallel_derivative_model="gkw_igh"`.
- Verification run this round:
  - `python -m py_compile src/stellarator_gk/physics/rhs_terms.py src/stellarator_gk/benchmarks.py`
  - `uv run ruff format src/stellarator_gk/physics/rhs_terms.py src/stellarator_gk/physics/__init__.py src/stellarator_gk/__init__.py src/stellarator_gk/benchmarks.py tests/test_linear_rhs.py`
  - `uv run pytest tests/test_linear_rhs.py::test_gkw_igh_backend_matches_fortran_style_reference_operator -q`
  - `uv run pytest tests/test_linear_rhs.py -q`
  - `uv run ruff check src/stellarator_gk/physics/rhs_terms.py src/stellarator_gk/physics/__init__.py src/stellarator_gk/__init__.py src/stellarator_gk/benchmarks.py tests/test_linear_rhs.py`
  - `uv run pytest tests/test_linear_rhs.py tests/test_benchmark_references.py::test_cyclone_igh_arakawa_audit_quantifies_fused_path_gap -q`
  - `uv run ruff check src tests examples scripts`
  - `uv run pytest -q`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Verification results:
  - focused fused-`igh` backend test: 1 passed,
  - linear RHS suite: 12 passed,
  - focused RHS plus `igh` benchmark audit suite: 13 passed,
  - full pytest suite: 152 passed,
  - focused and full ruff: all checks passed,
  - `main.tex` built successfully with existing underfull-box warnings only.

### 2026-06-01: Added GKW `ltrapping_arakawa` `igh` Audit

- Committed the previous coefficient/source tranche as:
  - `ddec9e7 Add Cyclone coefficient source audit`.
- Added `CycloneIghArakawaAudit` and
  `run_cyclone_base_case_igh_arakawa_audit`.
- Added `examples/audit_cyclone_igh_arakawa.py`, which reconstructs GKW
  `linear_terms.F90::igh` plus `jhg_interior`, `igh_zero_two`, `igh_two`, and
  `diffus`, then writes:
  - `figures/cyclone_igh_arakawa_audit.csv`.
- Main production-control findings at output window 62 (`t=3.72`,
  `z=0.09375`):
  - local fused-vs-separated profile delta:
    `3.5775663113790776e-04`,
  - maximum profile delta: `1.5086176659840009e-03`,
  - relative delta envelope: `0.1257254487392138`,
  - maximum GKW `igh` parallel diffusion profile:
    `7.044002628973732e-03`,
  - maximum GKW `igh` velocity diffusion profile:
    `7.597330135842486e-04`,
  - worst profile row: `z_index=25`.
- Interpretation: unlike the separated Term II/IV/V/VII/VIII coefficient
  audit, the production `ltrapping_arakawa` fused `igh` audit exposes an
  actionable operator mismatch between GKW's actual selected-`ky` path and the
  current separated streaming/mirror fallback. The next implementation target
  is an optional matrix-free GKW `igh` RHS backend.
- Verification run this round:
  - `uv run --extra dev ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_igh_arakawa.py`
  - `uv run --extra dev pytest tests/test_benchmark_references.py::test_cyclone_igh_arakawa_audit_quantifies_fused_path_gap -q`
  - `uv run --extra dev python examples/audit_cyclone_igh_arakawa.py`
  - `uv run --extra dev ruff check src tests examples scripts`
  - `uv run --extra dev pytest tests/test_benchmark_references.py -q`
  - `uv run --extra dev pytest -q`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Verification results:
  - focused `igh` audit test: 1 passed,
  - focused benchmark suite: 32 passed,
  - full pytest suite: 151 passed,
  - focused ruff: all checks passed,
  - full ruff: all checks passed,
  - production-control `igh` example: OPEN as expected,
  - `main.tex` built successfully with existing underfull-box warnings only.

### 2026-06-01: Added GKW Coefficient Source Audit

- Committed the previous matrix-convention tranche as:
  - `a4dbd29 Add Cyclone matdat matrix audit`.
- Added `CycloneCoefficientSourceAudit` and
  `run_cyclone_base_case_coefficient_source_audit`.
- Added `examples/audit_cyclone_coefficient_source.py`, which reconstructs
  production-control selected-`ky` GKW source formulas for Terms II, IV, V,
  VII, and VIII and writes:
  - `figures/cyclone_coefficient_source_audit.csv`.
- Main production-control findings at output window 62 (`t=3.72`,
  `z=0.09375`):
  - Term II `vdgradf` action/coefficient/insertion errors: `0.0`,
  - Term IV `trapdf_4d` action/coefficient/insertion errors: `0.0`,
  - Term V `ve_grad_fm` action/insertion error:
    `3.1031676915590914e-17`, coefficient error `0.0`,
  - Term VII `vpgrphi_3_newbc` action/insertion error:
    `2.4532694666933987e-18`, coefficient error `0.0`,
  - Term VIII `vd_grad_phi_fm` action/insertion error:
    `3.878959614448864e-18`, coefficient error `0.0`,
  - maximum term/coefficient/insertion envelope:
    `3.1031676915590914e-17`.
- Interpretation: the separated source-level coefficient construction and
  insertion for Terms II, IV, V, VII, and VIII is no longer a plausible cause
  of the remaining production selected-`ky` profile/growth gap. The next
  narrowed target is GKW's production `ltrapping_arakawa` fused `igh` path,
  including the combined Term I/IV Hamiltonian stencil and `disp_vp=0.2`.
- Verification run this round:
  - `uv run --extra dev ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_coefficient_source.py`
  - `uv run --extra dev pytest tests/test_benchmark_references.py::test_cyclone_coefficient_source_audit_matches_gkw_formulas -q`
  - `uv run --extra dev python examples/audit_cyclone_coefficient_source.py`
  - `uv run --extra dev ruff check src tests examples scripts`
  - `uv run --extra dev pytest tests/test_benchmark_references.py -q`
  - `uv run --extra dev pytest -q`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Verification results:
  - focused coefficient/source audit test: 1 passed,
  - focused benchmark suite: 31 passed,
  - full pytest suite: 150 passed,
  - focused ruff: all checks passed,
  - full ruff: all checks passed,
  - production-control coefficient/source example: PASS,
  - `main.tex` built successfully with existing underfull-box warnings only.

### 2026-06-01: Added GKW Matdat Matrix Convention Audit

- Committed the previous diagnostic-packing tranche as:
  - `0cb5007 Add Cyclone diagnostic packing audit`.
- Added `CycloneMatdatMatrixAudit` and
  `run_cyclone_base_case_matdat_matrix_audit`.
- Added `examples/audit_cyclone_matdat_matrix.py`, which builds a reduced CBC
  dense residual matrix, reconstructs GKW `matdat.F90` sparse/source
  conventions, and writes:
  - `figures/cyclone_matdat_matrix_audit.csv`.
- Main reduced-grid findings for \(N_z=8\), \(N_{v_\parallel}=6\),
  \(N_\mu=4\), and 192 state entries:
  - nonzero matrix entries: `19032`,
  - duplicate triplets after synthetic split: `38064`,
  - real/complex-real split counts: `14424` real and `4608` complex,
  - matrix-action error: `2.6021205067653364e-18`,
  - source maximum absolute value: `0.0`,
  - explicit `dtim*(source+mat*fdis_tmp)` delta error:
    `8.47042252985074e-21`,
  - duplicate-triplet compression error: `3.035783099198609e-18`,
  - `complex-real` split action error: `1.7347484980098753e-18`,
  - linearity error: `7.105427357601002e-15`.
- Interpretation: the reduced matrix-free residual obeys the GKW sparse
  matrix/source conventions, so the remaining CBC gap should be chased through
  source-level coefficient construction for Terms II, IV, V, VII, and VIII in
  the production selected-`ky` setup.
- Verification run this round:
  - `uv run --extra dev ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_matdat_matrix.py`
  - `uv run --extra dev pytest tests/test_benchmark_references.py::test_cyclone_matdat_matrix_audit_matches_sparse_conventions -q`
  - `uv run --extra dev python examples/audit_cyclone_matdat_matrix.py`
  - `uv run --extra dev ruff check src tests examples scripts`
  - `uv run --extra dev pytest tests/test_benchmark_references.py -q`
  - `uv run --extra dev pytest -q`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Verification results:
  - focused matdat audit test: 1 passed,
  - focused benchmark suite: 30 passed,
  - full pytest suite: 149 passed,
  - full ruff: all checks passed,
  - `main.tex` built successfully with existing underfull-box warnings only.

### 2026-06-01: Added GKW Diagnostic Packing Audit

- Added `CycloneDiagnosticPackingAudit` and
  `run_cyclone_base_case_diagnostic_packing_audit`.
- Added `examples/audit_cyclone_diagnostic_packing.py`, which packs the
  solver field into the GKW `dist.F90`/`index_function.F90` field layout and
  reconstructs the `diagnostic.F90::parallel_phi` and `phi_ky_spec` formulas:
  - `figures/cyclone_diagnostic_packing_audit.csv`.
- Main production-control findings at output window 62 (`t=3.72`):
  - field-packing roundtrip error: `0.0`,
  - `parallel_phi.dat` source-formula error: `0.0`,
  - selected single-mode profile error: `0.0`,
  - `ky` field-spectrum error: `0.0`,
  - `kx` field-spectrum error: `0.0`.
- Interpretation: GKW field packing, `get_phi`, `parallel_phi.dat`, and
  field-spectrum diagnostic formulas are not the source of the matched
  Cyclone profile gap. The remaining immediate target is the GKW matrix and
  source-term construction path in `matdat.F90`.
- Verification run this round:
  - `uv run --extra dev ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_diagnostic_packing.py`
  - `uv run --extra dev pytest tests/test_benchmark_references.py::test_cyclone_diagnostic_packing_audit_matches_gkw_source_layout -q`
  - `uv run --extra dev python examples/audit_cyclone_diagnostic_packing.py`
  - `uv run --extra dev ruff check src tests examples scripts`
  - `uv run --extra dev pytest tests/test_benchmark_references.py -q`
  - `uv run --extra dev pytest -q`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Verification results:
  - focused diagnostic-packing audit test: 1 passed,
  - focused benchmark suite: 29 passed,
  - full pytest suite: 148 passed,
  - full ruff: all checks passed,
  - `main.tex` built successfully with existing underfull-box warnings only.

### 2026-06-01: Added GKW RK4 and Normalization Sequence Audit

- Added `CycloneTimeNormalizationAudit` and
  `run_cyclone_base_case_time_normalization_audit`.
- Added `normalization_model='gkw_unweighted'` support to
  `run_cyclone_base_case_trace` and `run_production_cyclone_base_case_gate`
  so the windowed trace can use the same unweighted field norm computed by
  `normalise.F90::calc_factor`.
- Added `examples/audit_cyclone_time_normalization.py`, which reconstructs
  the `exp_integration.F90::rk4` stage sequence and
  `normalise.F90::normalize(2)` window-normalization cadence and writes:
  - `figures/cyclone_time_normalization_audit.csv`.
- Main production-control findings for `finit='cosine'`, 20 steps per window,
  and 80 windows:
  - RK4 source-sequence error: `3.476216610979977e-18`,
  - solver/source window-growth sequence error: `1.2878587085651816e-14`,
  - post-normalization field-norm error: `1.3322676295501878e-15`,
  - quasineutrality field-linearity error after normalization:
    `1.668583973270389e-15`.
- Interpretation: the selected-mode RK4 staging, diagnostic-window cadence,
  and GKW unweighted field normalization now match the source sequence to
  roundoff. The remaining CBC profile/growth gap should be chased through
  GKW matrix construction, diagnostic output, and field/index packing
  conventions before adding a `cosine2` GKW patch.
- Verification run this round:
  - `uv run --extra dev ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_time_normalization.py`
  - `uv run --extra dev pytest tests/test_benchmark_references.py::test_cyclone_time_normalization_audit_matches_gkw_sequence -q`
  - `uv run --extra dev python examples/audit_cyclone_time_normalization.py`
  - `uv run --extra dev ruff check src tests examples scripts`
  - `uv run --extra dev pytest tests/test_benchmark_references.py -q`
  - `uv run --extra dev pytest -q`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Verification results:
  - focused time-normalization audit test: 1 passed,
  - focused benchmark suite: 28 passed,
  - full pytest suite: 147 passed,
  - full ruff: all checks passed,
  - `main.tex` built successfully with existing underfull-box warnings only.

### 2026-06-01: Added GKW Fortran Term I Source Audit

- Committed the previous profile-operator audit tranche as:
  - `4080d5b Add Cyclone profile operator audit`.
- Added `CycloneTermIFortranAudit` and
  `run_cyclone_base_case_term_i_fortran_audit`.
- Added `examples/audit_cyclone_term_i_fortran.py`, which reconstructs the
  GKW Term I coefficients directly from
  `linear_terms.F90::vpar_grad_df_4d_testnewbc` and writes:
  - `figures/cyclone_term_i_fortran_audit.csv`.
- Main findings at output window 62 (`t=3.72`, `z=0.09375`):
  - maximum Term I operator error: `5.137226812672801e-18`,
  - maximum coefficient-table error: `4.440892098500626e-16`,
  - `idisp=2` recurrence-speed error: `5.551115123125783e-17`,
  - local target-row Term I error is at roundoff.
- Interpretation: the selected-mode GKW-upwind Term I implementation,
  sign choice, boundary stencils, and `vpgr_rms` recurrence normalization
  match the GKW Fortran source formulas. The remaining CBC selected-`ky`
  discrepancy should now be chased through the time-update and normalization
  sequence (`exp_integration.F90`/`normalise.F90`) before patching GKW for a
  `cosine2` initialization.
- Verification run this round:
  - `uv run --extra dev ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_term_i_fortran.py`
  - `uv run --extra dev pytest tests/test_benchmark_references.py::test_cyclone_term_i_fortran_audit_matches_source_reconstruction -q`
  - `uv run --extra dev python examples/audit_cyclone_term_i_fortran.py`

### 2026-06-01: Added Cyclone Profile Operator Audit

- Committed the previous alignment/localization tranche as:
  - `e11e3ed Add GKW parallel phi alignment audit`.
- Added `CycloneProfileOperatorAudit` and
  `run_cyclone_base_case_profile_operator_audit`.
- Added `examples/audit_cyclone_profile_operator.py`, which advances the
  production-control selected-`ky` Cyclone setup to the localized GKW
  `parallel_phi.dat` mismatch row and writes:
  - `figures/cyclone_profile_operator_audit.csv`.
- Main findings at output window 62 (`t=3.72`, `z=0.09375`):
  - field residual: `4.742874840267547e-16`,
  - phi reconstruction error: `3.7238012298709097e-16`,
  - RHS assembly error: `0.0`,
  - local matrix-versus-GKW-upwind streaming delta:
    `5.228399497117055e-03`,
  - maximum streaming delta: `7.044002628973733e-03`,
  - boundary streaming delta: `1.9373242808842016e-03`,
  - local matrix-versus-GKW-upwind field-drive delta:
    `2.7611004325485534e-19`,
  - maximum field-drive delta: `2.1023285024078316e-06`.
- Interpretation: the field solve, field reconstruction, field-drive assembly,
  and RHS assembly are not the visible source of the matched GKW
  `parallel_phi.dat` gap. The remaining mismatch now points most strongly to
  the selected-mode parallel streaming/upwind path and should be checked
  against the GKW Fortran Term I stencil/sign/normalization logic.
- Verification run this round:
  - `uv run --extra dev ruff check src/stellarator_gk/benchmarks.py src/stellarator_gk/__init__.py tests/test_benchmark_references.py examples/audit_cyclone_profile_operator.py`
  - `uv run --extra dev pytest tests/test_benchmark_references.py::test_cyclone_profile_operator_audit_checks_selected_mode_assembly -q`
  - `uv run --extra dev pytest tests/test_benchmark_references.py::test_cyclone_profile_operator_audit_checks_selected_mode_assembly tests/test_benchmark_references.py::test_cyclone_parallel_phi_trace_records_gkw_style_profiles -q`
  - `uv run --extra dev ruff check src tests examples scripts`
  - `uv run --extra dev pytest tests/test_benchmark_references.py -q`
  - `uv run --extra dev pytest -q`
  - `uv run --extra dev python examples/audit_cyclone_profile_operator.py`
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

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
