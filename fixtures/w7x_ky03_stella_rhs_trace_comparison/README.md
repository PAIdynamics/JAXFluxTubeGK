# W7-X ky=0.3 stella RHS Trace Comparison

This fixture compares the patched stella selected-mode RHS trace
with the current solver-side RHS/model balance fixture.

The comparison converts stella `rhs*dt` records to continuous-time
RHS norms and compares scale-free term/total norm ratios. It does
not claim direct array parity yet, because the committed solver
fixture stores scalar term summaries and uses a different velocity
grid from the stella trace. The array adapter uses separable linear
complex interpolation, forbids extrapolation, and evaluates weighted
errors with the chosen target grid's `w_z*w_vpar*w_mu` quadrature.

When `--solver-array` is supplied, `weighted_array_comparison.csv`
retains compact metrics for every labeled stella RHS call. Raw
stella and solver arrays remain external. The v3 contract includes
quasineutrality and normalization; numerical parity currently fails
the declared relative-L2 tolerance.
