# Velocity Backend Policy

The production CPU representation is velocity-space collocation. The
Hermite–Laguerre implementation remains a reduced GX-style discriminator and is
not a prerequisite for continued production use of the collocation solver.

The decision is claim-specific:

| Backend | Role | Accepted scope |
| --- | --- | --- |
| `chebyshev` | General finite-box collocation | Reduced and differentiable linear CPU solves |
| `finite_difference` | GKW-compatible specialist collocation | Reduced CPU solves and GKW term/operator parity |
| `midpoint_gauss_laguerre` | Source-matched W7-X collocation | Reduced/differentiable linear CPU solves and the validated W7-X linear recipe |
| `native` | Arbitrary supplied nodes | Custom-grid plumbing only until a named gate passes |
| `hermite_laguerre` | GX-style moments | Reduced GX branch discriminator only |

The W7-X claim is not granted by the backend name alone. It also requires the
validated 32-by-8 resolution, phase-space measure, stella-compatible split
ordering, timestep, initial state, growth window, geometry, and convergence
controls.

Call `require_velocity_backend_for_claim(backend, claim)` before attaching a
named scientific claim to a run. Unsupported combinations raise
`VelocityBackendNotReadyError` with the recorded limitation. This policy keeps
backend implementation, numerical validation, and scientific scope separate.

Hermite–Laguerre can be promoted later only after it is connected to the full
stellarator geometry/residual and kinetic field solve and passes convergence,
external parity, optimization-gradient, and CPU timing gates comparable to the
collocation path.
