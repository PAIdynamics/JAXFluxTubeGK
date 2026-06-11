# W7-X ky=0.3 stella trace contract

This fixture audits what the standard matched stella `.out.nc`
output can compare against the solver-side RHS balance.

The direct geometry, streaming multiplier, and `kperp2` contract
is compared in `stella_solver_geometry_comparison.csv`.
The status JSON records that standard stella diagnostics do not
contain the complex distribution or per-term RHS/source arrays
needed for true streaming/mirror parity.

Use `stella_rhs_trace_patch_plan.md` as the next stella-side
diagnostic target.
