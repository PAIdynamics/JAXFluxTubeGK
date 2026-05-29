# CPU Performance and Differentiability Notes

Phase 11 keeps the first production target CPU-oriented and differentiable. GX
remains a useful scaling and benchmark reference, but its CUDA/GPU execution
model is not a near-term requirement for this package.

## CPU Execution

- Build grids, geometry, FLR factors, field-solve denominators, and RHS
  coefficients once with `build_linear_residual_precompute`.
- Reuse the precompute object inside `linear_residual` or
  `jitted_linear_residual`; the traced residual should see array operations, not
  geometry/topology construction.
- Use `integrate_fixed_step(..., store_history=False)` for optimization loops
  that only need endpoint fields or endpoint state norms. The default
  `store_history=True` is retained for diagnostics and validation.
- Use `benchmark_linear_residual` only as a reduced-grid smoke profiler. It is
  not a substitute for a full benchmark campaign against GKW/Gyaradax/GX cases.

## Memory

`estimate_linear_memory_from_dimensions` gives a static target-grid estimate
without allocating the phase-space state. `estimate_linear_memory_from_precompute`
checks assembled reduced problems and includes the actual precompute PyTree byte
count.

The dominant arrays scale like

```text
state      ~ N_species N_vparallel N_mu N_z N_kx N_ky
history    ~ (N_steps + 1) state          when store_history=True
endpoints  ~ 2 state                      when store_history=False
drift/FLR  ~ N_species N_vparallel N_mu N_z N_kx N_ky
field      ~ N_z N_kx N_ky
```

This is the main CPU difference from GX: GX uses GPU-resident moment arrays and
CUDA kernels, while this code first prioritizes small-to-medium differentiable
linear solves, reliable gradients, and clear CPU memory behavior.

## Differentiable Quantities

Differentiable:

- continuous species/profile parameters such as density, temperature, and
  gradients,
- analytic geometry scalars such as `q`, `shat`, and `eps`,
- Boozer/precomputed geometry arrays passed as JAX arrays,
- field solves, RHS terms, RK4 steps, diagnostics, and objective helpers.

Static under JAX transforms:

- grid sizes and derivative backends,
- Fourier topology, mode labels, and twist-and-shift connectivity,
- file I/O and parsers for external benchmark inputs,
- benchmark/profiling timers and memory-estimate dataclasses.

Integer topology should be rebuilt outside traced objectives whenever grid
resolution, `ikxspace`, or mode connectivity changes. Continuous geometry and
profile parameters may be differentiated through on a fixed topology.
