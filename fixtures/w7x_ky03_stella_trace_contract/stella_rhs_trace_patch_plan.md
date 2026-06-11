# stella W7-X ky=0.3 RHS Trace Patch Plan

The standard matched stella `.out.nc` file was audited against
`fixtures/w7x_ky03_rhs_model_balance/`.  The solver fixture was built from
stella's rounded ASCII `.geometry` file, while this audit reads full-precision
arrays from `.out.nc`.  The practical geometry-file precision contract passes,
while strict `.out.nc` precision does not:

- max `z` error: `5.551115123125783e-17`
- max `B` error: `0.0004946551081135286`
- max `F=b_dot_gradz/(2*pi)` error: `6.74434460409476e-07`
- max `kperp2` error: `0.00039841172216437126`
- stella `.geometry` rounding tolerance:
  `0.0007`
- strict `.out.nc` tolerance:
  `5e-10`

The standard file does not contain the complex arrays required for true term
parity:

- `complex_g(vpa,mu,z,kx,ky,species)`
- `rhs_total(vpa,mu,z,kx,ky,species,ri)`
- `rhs_parallel_streaming(vpa,mu,z,kx,ky,species,ri)`
- `rhs_mirror_force(vpa,mu,z,kx,ky,species,ri)`
- `rhs_magnetic_drift(vpa,mu,z,kx,ky,species,ri)`
- `rhs_field_drive(vpa,mu,z,kx,ky,species,ri)`

Minimal stella-side insertion points:

1. In `STELLA_CODE/gyrokinetic_equation/gyrokinetic_equation_explicit.f90`,
   inside `add_explicit_gyrokinetic_terms`, snapshot `rhs` immediately before
   and after these calls:
   - `advance_mirror_explicit(pdf, rhs)`
   - `advance_wdrifty_explicit(pdf, phi, bpar, rhs)`
   - `advance_wdriftx_explicit(pdf, phi, bpar, rhs)`
   - `advance_wstar_explicit(phi, rhs)`
   - `advance_parallel_streaming_explicit(pdf, phi, bpar, rhs)`
2. Write the selected serial-run arrays for Fortran indices `iky=4`, `ikx=1`,
   all `z`, all `vpa`, all `mu`, species 1, at the final or requested trace
   step.  Store each delta as real/imag columns with term names.
3. Also write the input `pdf` state used by the RHS call and the solved `phi`.
4. Keep the trace units as stella's native `rhs*dt`; the Python comparator
   should divide by `delt` only if comparing against a continuous-time RHS.

This patch plan intentionally does not reinterpret stella's `|g|^2` diagnostics
as a complex distribution.  The solver-side balance is streaming dominated, so
the first true comparison should prioritize the streaming and mirror deltas.
