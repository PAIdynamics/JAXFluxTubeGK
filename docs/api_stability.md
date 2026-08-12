# API stability for 0.1.x

`jax-fluxtube-gk` is pre-1.0 software, but the `0.1.x` series maintains a small
documented compatibility boundary. Patch releases preserve the names,
dataclass fields, argument meanings, and serialized schema versions listed
below. Additive keyword arguments and new exports may still be introduced.

## Stable solver surface

- Public immutable specifications in `jax_fluxtube_gk.types`, including grid,
  species, topology, geometry-scalar, and solver-control records.
- Grid builders in `jax_fluxtube_gk.grids` and the provider-neutral geometry
  contracts in `jax_fluxtube_gk.geometry`.
- Linear residual precompute/build/evaluation entry points in
  `jax_fluxtube_gk.solver`, fixed-step/adaptive integration results in
  `jax_fluxtube_gk.time_advance`, and public diagnostics.
- `BenchmarkTarget` and its residual/cost helpers in `jax_fluxtube_gk.targets`.
- Fixed-topology optimization contracts and versioned geometry/design
  checkpoint schemas. Schema changes require a new schema version and a
  fail-closed loader error rather than silent reinterpretation.

The canonical imports for this surface are from `jax_fluxtube_gk` or the
focused subsystem module. Symbols present in `jax_fluxtube_gk.__all__` define
the compact top-level API for this series.

### Provider-neutral naming

Core APIs are named for solver concepts and file contracts, not external
implementations. The production refactor intentionally removed the former
GX-prefixed core aliases:

| Current API | Purpose |
| --- | --- |
| `EikData`, `EikGeometryProvider` | Generic numeric or labelled eik tables |
| `load_eik_data`, `resample_eik_data` | Eik parsing and interpolation |
| `MomentRHSParams`, `moment_linear_rhs` | Experimental spectral-moment model |
| `linked_kz_wavenumbers`, `apply_kz_hypercollision` | Linked parallel spectral operators |

External-code-specific parsers and comparisons belong under
`jax_fluxtube_gk.validation` and are not re-exported by the core package.

## Validation surface

External-reference workflows live under `jax_fluxtube_gk.validation`:

- `validation.fixture_io`
- `validation.cyclone_gkw`
- `validation.geometry_parity`
- `validation.w7x`

These contracts are versioned with their fixture/report schemas, but their
workflow functions may evolve when an external solver changes. Historical
benchmark names remain available lazily from `jax_fluxtube_gk` for `0.1.x`
compatibility; they are deliberately absent from `jax_fluxtube_gk.__all__` and
new code should use the validation namespace.

## Experimental/internal surface

Underscore-prefixed helpers, the legacy `jax_fluxtube_gk.benchmarks` module,
Hermite--Laguerre production experiments, unrestricted topology-changing
optimization, and unvalidated nonlinear/electromagnetic combinations are not
stable APIs. Their behavior may change without a deprecation cycle, and they
must not be used to infer a completed scientific validation gate.
