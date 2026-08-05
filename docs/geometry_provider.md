# Geometry Provider Contract

`stellarator_gk.geometry` defines the boundary between equilibrium/geometry
codes and the gyrokinetic solver. Providers implement one method:

```python
class GeometryProvider(Protocol):
    def get_geometry(self, request: GeometryRequest) -> GeometryResult: ...
```

`GeometryRequest` contains only field-line selection and static grid controls:
the named configuration, radial coordinate/value, `alpha`, named parallel
coordinate/unit and interval, topology, periodic-endpoint policy, field-period
span, resolution, and dtype.
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

The version-1 normalization is `stellarator_gk_physical_v1`. `theta`, `phi`,
and `alpha` are in radians; `z` uses the explicitly declared parallel
coordinate and unit (for example `zeta` in radians or `zed_over_2pi` in turns).
Magnetic field, length, and flux use provider-declared references
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

## DESC adapter

`DescGeometryProvider` accepts either an in-memory DESC equilibrium or a DESC
path. Exactly one must be supplied. File loading and DESC grid construction are
contained inside `get_geometry`; after `resolve_geometry` returns, solver code
uses the same provider-neutral result as every other backend:

Install the supported optional backend with `uv sync --extra desc` (or the
complete MHD environment with `uv sync --extra mhd`). Corresponding `vmecpp`
and `gvec` extras are also declared for selective installations. Exact fork
revisions remain pinned by `dependencies.toml` and the bootstrap workflow. The
provider imports DESC lazily and never modifies `PYTHONPATH` or assumes a source
checkout.

```python
provider = DescGeometryProvider(path="/path/to/equilibrium.h5")
request = GeometryRequest(
    configuration="my-equilibrium",
    radial_value=0.5,
    alpha=0.0,
    n_z=64,
)
result = resolve_geometry(provider, request)
geometry = internal_geometry_from_result(result)
```

An in-memory provider may explicitly declare `differentiable=True` when its
DESC evaluation preserves JAX traces. A path-backed provider cannot make that
claim. Both forms record provider version, revision, source, and configuration
in the result metadata.

When neither an object nor a path is supplied, the provider resolves
`request.configuration` through DESC's installed examples API. This gives a
named `W7-X` path without source-tree assumptions. The fixed-topology gradient
contract is covered with a JAX-compatible equilibrium object; installed DESC
W7-X evaluation is an opt-in integration check.

## VMEC++ and GVEC adapters

`VmecppGeometryProvider` accepts an in-memory `VmecOutput`/`wout`, a
`VmecInput`, or an optional user `wout` path. With no explicit source it loads
VMEC++'s installed named configuration (including `w7x-standard`), calls
`vmecpp.run(...)`, and converts the returned Fourier data directly in memory.
The transformation evaluates the PEST field line, VMEC full/half radial meshes,
metrics, parallel derivative, grad-B drifts, and curvature drifts in the common
normalization. It does not invoke GX/GIST or stella and does not write NetCDF.

`GvecGeometryProvider` accepts an in-memory GVEC state or an explicit parameter
file and requests the minimum PEST quantities needed by the same physical
contract. GVEC currently has no stable named W7-X constructor, so named W7-X
selection is not claimed for that backend. Both native adapters import their
packages lazily and are non-differentiable at their current native boundaries.

## GX/GIST and stella readers

`GxEikGeometryProvider` and `StellaGeometryProvider` contain the file parsing
that previously lived in `benchmarks.py` and the linear-scan example. Both are
file-backed and therefore non-differentiable. They convert imported normalized
grad-B and curvature contributions into the common physical split; the
canonical mapper then constructs the summed solver drift coefficients.

GX/GIST eik requests use a radian parallel coordinate. The adapter supports
both the numeric ten-column layout and the older GX multi-block layout,
resampling onto the requested endpoint-excluded grid. stella requests declare
`parallel_coordinate="zed_over_2pi"` with unit `turn`; the adapter drops and
rejects duplicate periodic endpoints, retains the file's field-line span, and
converts `b.Gz` consistently for the normalized coordinate.

## Synthetic and in-memory arrays

`SyntheticGeometryProvider` is the standalone differentiable provider used for
interface and solver smoke tests. `PhysicalArrayGeometryProvider` adapts an
in-memory mapping or callable that returns physical fields. It is also the
tested low-level handoff used to isolate provider transformations: adapters
supply physical arrays, `iota`, shear, and provenance, while solver construction
remains unchanged. The array provider rejects missing physical fields and never
accepts precomputed `F`, `G`, `E_y`, `D_x`, or `D_y`.

## Caching

The live provider call is canonical. Expensive callers may explicitly
serialize a result in a user cache, but the cache must be outside the source
repository. `cache_path_is_external` enforces that boundary. A cached result
must retain the complete schema/provenance record and pass the same validation
as a live result; it is never required for a solver run.

`write_geometry_result_cache(result, path, repository_root=...)` writes
`metadata.json` plus compressed `arrays.npz`. It refuses an existing cache
unless `overwrite=True`. `load_geometry_result_cache(path, request=...)`
reconstructs the grid and physical model, checks the schema and optional
request, and runs full validation. A loaded result is always marked
non-differentiable because serialization severs the original JAX trace.
