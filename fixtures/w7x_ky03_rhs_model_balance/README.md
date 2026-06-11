# W7-X ky=0.3 RHS/model balance

This fixture freezes the stella-imported W7-X geometry, `kx=0`, 
`n_kx=1`, species gradients, and late-time normalization controls 
used by the W7-X/stella comparison, then decomposes the solver RHS 
for the discrepant `ky=0.3` branch.

Files:

- `rhs_term_balance.csv`: scalar selected-mode RHS norms and projections.
- `rhs_density_balance.csv`: z profiles of quasineutrality numerator rates.
- `geometry_model_balance.csv`: z-local geometry, FLR, drift, and field inputs.
- `rhs_model_balance_status.json`: diagnostic status and next action.

This is a solver-side diagnostic.  A production parity claim still 
requires a matched stella source-term or distribution/RHS trace.
