# Geometry Provider Contract

`stellarator_gk.geometry` defines the boundary between equilibrium/geometry
codes and the gyrokinetic solver. Providers implement one method:

```python
class GeometryProvider(Protocol):
    def get_geometry(self, request: GeometryRequest) -> GeometryResult: ...
```

`GeometryRequest` contains only field-line selection and static grid controls:
the named configuration, radial coordinate/value, `alpha`, parallel interval,
topology, periodic-endpoint policy, field-period span, resolution, and dtype.
`GeometryResult` contains a `ParallelGrid`, a
`PhysicalFluxTubeGeometry`, and immutable `GeometryMetadata`. The solver calls
`internal_geometry_from_result` after the provider boundary; providers do not
construct solver coefficients such as `F`, `G`, `E_y`, `D_x`, or `D_y`.

## Schema version 1

`GEOMETRY_SCHEMA_VERSION` is currently `1`. Serialized provider results must
record this version and all `GeometryMetadata` fields:

- provider name/version, source revision, configuration, source, and command;
- radial coordinate (`rho`, `psi`, or `x`), `alpha` convention, and signs;
- `iota`, `nfp`, field-period span, topology, endpoint policy, and linking;
- quadrature weights and the declared coordinate period;
- normalization name and the unit of every physical array;
- whether continuous provider outputs are differentiable.

The version-1 normalization is `stellarator_gk_physical_v1`. Angles are in
radians. Magnetic field, length, and flux use provider-declared references
`B_ref`, `L_ref`, and `psi_ref`; the complete per-field unit table is exported
as `GEOMETRY_FIELD_UNITS`. The sign contract uses
`alpha = theta - iota * phi`,
`b dot grad = b_dot_grad_z * d/dz`, right-handed cross products, and an outward
direction given by the increasing declared radial coordinate.

Periodic grids exclude the upper endpoint. `validate_geometry_result` rejects
duplicate endpoints, non-finite or incorrectly shaped arrays, non-positive
magnetic field or quadrature weights, inconsistent topology/grid sizes,
incompatible normalization, and invalid perpendicular metrics with errors that
name the failed field or convention.

## Differentiability boundary

Continuous geometry arrays, `iota`, and shear are JAX PyTree leaves. Provider
selection, file I/O, schema metadata, topology, endpoint policy, `nfp`, field
periods, and linking maps are static. A provider may set
`metadata.differentiable = True` only when its in-memory continuous arrays retain
a JAX trace back to equilibrium/design parameters. File readers normally set it
to `False`.

Call validation outside `jax.jit`/`jax.grad`, then pass the returned result into
the differentiable physical-to-internal map. Validation intentionally converts
arrays to NumPy so malformed external data fails before compilation.

## Caching

The live provider call is canonical. Expensive callers may explicitly
serialize a result in a user cache, but the cache must be outside the source
repository. `cache_path_is_external` enforces that boundary. A cached result
must retain the complete schema/provenance record and pass the same validation
as a live result; it is never required for a solver run.
