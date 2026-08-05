"""Versioned public boundary between MHD geometry providers and the GK solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from ..types import ParallelGrid, _PyTreeDataclass
from .flux_tube import FluxTubeGeometry, PhysicalFluxTubeGeometry, map_physical_to_internal_geometry


GEOMETRY_SCHEMA_VERSION = 2
GEOMETRY_NORMALIZATION = "stellarator_gk_physical_v2"
GEOMETRY_FIELD_UNITS = (
    ("z", "declared_parallel_coordinate"),
    ("theta", "radian"),
    ("phi", "radian"),
    ("alpha", "radian"),
    ("rho", "declared_radial_coordinate"),
    ("B", "B_ref"),
    ("b_dot_grad_z", "1/L_ref"),
    ("grad_psi_sq", "psi_ref^2/L_ref^2"),
    ("grad_alpha_sq", "1/L_ref^2"),
    ("grad_psi_dot_grad_alpha", "psi_ref/L_ref^2"),
    ("B_cross_gradB_dot_grad_psi", "B_ref^2*psi_ref/L_ref^2"),
    ("B_cross_gradB_dot_grad_alpha", "B_ref^2/L_ref^2"),
    ("b_cross_kappa_dot_grad_psi", "psi_ref/L_ref^2"),
    ("b_cross_kappa_dot_grad_alpha", "1/L_ref^2"),
    ("equilibrium_drive_scale", "normalized_diamagnetic_frequency/ky"),
    ("w_z", "radian"),
)

_RADIAL_COORDINATES = ("rho", "psi", "x")
_TOPOLOGIES = ("periodic", "open")
_ENDPOINT_POLICIES = ("exclude", "include")
_PHYSICAL_ARRAY_FIELDS = (
    "z",
    "w_z",
    "theta",
    "phi",
    "alpha",
    "rho",
    "B",
    "b_dot_grad_z",
    "grad_psi_sq",
    "grad_alpha_sq",
    "grad_psi_dot_grad_alpha",
    "B_cross_gradB_dot_grad_psi",
    "B_cross_gradB_dot_grad_alpha",
    "b_cross_kappa_dot_grad_psi",
    "b_cross_kappa_dot_grad_alpha",
    "equilibrium_drive_scale",
)


@dataclass(frozen=True)
class GeometryRequest:
    """Static field-line and grid request passed to every geometry provider."""

    configuration: str = "unspecified"
    radial_coordinate: str = "rho"
    radial_value: float = 0.5
    alpha: float = 0.0
    parallel_coordinate: str = "zeta"
    parallel_coordinate_unit: str = "radian"
    n_z: int = 32
    z_min: float = -float(np.pi)
    z_max: float = float(np.pi)
    topology: str = "periodic"
    endpoint_policy: str = "exclude"
    field_periods: float = 1.0
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if not self.configuration:
            raise ValueError("geometry configuration must be non-empty")
        if self.radial_coordinate not in _RADIAL_COORDINATES:
            raise ValueError(
                f"radial_coordinate must be one of {_RADIAL_COORDINATES}; "
                f"got {self.radial_coordinate!r}"
            )
        if self.radial_coordinate in ("rho", "x") and not 0.0 <= self.radial_value <= 1.0:
            raise ValueError("rho/x radial_value must lie in [0, 1]")
        if self.n_z < 2:
            raise ValueError("geometry request n_z must be at least 2")
        if self.z_max <= self.z_min:
            raise ValueError("geometry request z_max must be greater than z_min")
        if not self.parallel_coordinate or not self.parallel_coordinate_unit:
            raise ValueError("parallel coordinate name and unit must be non-empty")
        if self.topology not in _TOPOLOGIES:
            raise ValueError(f"topology must be one of {_TOPOLOGIES}; got {self.topology!r}")
        if self.endpoint_policy not in _ENDPOINT_POLICIES:
            raise ValueError(
                f"endpoint_policy must be one of {_ENDPOINT_POLICIES}; "
                f"got {self.endpoint_policy!r}"
            )
        if self.topology == "periodic" and self.endpoint_policy != "exclude":
            raise ValueError("periodic provider grids must exclude the duplicate upper endpoint")
        if self.field_periods <= 0.0:
            raise ValueError("field_periods must be positive")


@dataclass(frozen=True)
class GeometryNormalization:
    """Named normalization references for all physical geometry arrays."""

    name: str = GEOMETRY_NORMALIZATION
    length_reference: str = "provider_declared_L_ref"
    magnetic_field_reference: str = "provider_declared_B_ref"
    flux_reference: str = "provider_declared_psi_ref"
    field_units: tuple[tuple[str, str], ...] = GEOMETRY_FIELD_UNITS


@dataclass(frozen=True)
class GeometrySignConvention:
    """Explicit orientation and field-line coordinate sign contract."""

    alpha_definition: str = "alpha = theta - iota * phi"
    parallel_derivative: str = "b_dot_grad = b_dot_grad_z * d/dz"
    magnetic_drift: str = "B cross grad(B) and b cross kappa use right-handed cross products"
    radial_direction: str = "increasing declared radial coordinate"


@dataclass(frozen=True)
class GeometryLinking:
    """Static parallel-boundary linking information."""

    boundary_condition: str = "periodic"
    twist_and_shift: bool = False
    kx_shift: tuple[int, ...] = ()
    phase_shift: float = 0.0


@dataclass(frozen=True)
class GeometryProvenance:
    """Reproducibility record for a provider result."""

    provider: str
    provider_version: str = "unknown"
    revision: str = "unknown"
    source: str = "in-memory"
    configuration: str = "unspecified"
    command: str = "programmatic provider call"


@dataclass(frozen=True)
class GeometryMetadata:
    """Static, serializable metadata attached to a physical geometry result."""

    provenance: GeometryProvenance
    schema_version: int = GEOMETRY_SCHEMA_VERSION
    radial_coordinate: str = "rho"
    parallel_coordinate: str = "zeta"
    parallel_coordinate_unit: str = "radian"
    coordinate_period: float = 2.0 * float(np.pi)
    nfp: int = 1
    field_periods: float = 1.0
    topology: str = "periodic"
    endpoint_policy: str = "exclude"
    normalization: GeometryNormalization = GeometryNormalization()
    signs: GeometrySignConvention = GeometrySignConvention()
    linking: GeometryLinking = GeometryLinking()
    differentiable: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != GEOMETRY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported geometry schema version {self.schema_version}; "
                f"expected {GEOMETRY_SCHEMA_VERSION}"
            )
        if self.radial_coordinate not in _RADIAL_COORDINATES:
            raise ValueError(f"unsupported metadata radial coordinate {self.radial_coordinate!r}")
        if self.coordinate_period <= 0.0:
            raise ValueError("geometry coordinate_period must be positive")
        if not self.parallel_coordinate or not self.parallel_coordinate_unit:
            raise ValueError("metadata parallel coordinate name and unit must be non-empty")
        if self.nfp < 1:
            raise ValueError("geometry nfp must be at least 1")
        if self.field_periods <= 0.0:
            raise ValueError("geometry field_periods must be positive")
        if self.topology not in _TOPOLOGIES:
            raise ValueError(f"unsupported metadata topology {self.topology!r}")
        if self.endpoint_policy not in _ENDPOINT_POLICIES:
            raise ValueError(f"unsupported endpoint policy {self.endpoint_policy!r}")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GeometryResult(_PyTreeDataclass):
    """Provider output with differentiable arrays and immutable static metadata."""

    parallel_grid: ParallelGrid
    physical: PhysicalFluxTubeGeometry
    metadata: GeometryMetadata

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("parallel_grid", "physical")
    _static_fields: ClassVar[tuple[str, ...]] = ("metadata",)


@runtime_checkable
class GeometryProvider(Protocol):
    """Small common API implemented by synthetic, MHD, and file adapters."""

    def get_geometry(self, request: GeometryRequest) -> GeometryResult:
        """Construct physical geometry for ``request`` without solver-specific coefficients."""


def build_geometry_metadata(
    request: GeometryRequest,
    *,
    provenance: GeometryProvenance,
    nfp: int = 1,
    linking: GeometryLinking | None = None,
    normalization: GeometryNormalization | None = None,
    signs: GeometrySignConvention | None = None,
    differentiable: bool = False,
) -> GeometryMetadata:
    """Build metadata whose coordinate and topology fields mirror a request."""

    return GeometryMetadata(
        provenance=provenance,
        radial_coordinate=request.radial_coordinate,
        parallel_coordinate=request.parallel_coordinate,
        parallel_coordinate_unit=request.parallel_coordinate_unit,
        coordinate_period=request.z_max - request.z_min,
        nfp=nfp,
        field_periods=request.field_periods,
        topology=request.topology,
        endpoint_policy=request.endpoint_policy,
        normalization=normalization or GeometryNormalization(),
        signs=signs or GeometrySignConvention(),
        linking=linking
        or GeometryLinking(
            boundary_condition=request.topology,
            twist_and_shift=False,
        ),
        differentiable=differentiable,
    )


def resolve_geometry(
    provider: GeometryProvider,
    request: GeometryRequest,
    *,
    validate: bool = True,
) -> GeometryResult:
    """Call a provider and enforce the public result contract at the I/O boundary."""

    result = provider.get_geometry(request)
    if not isinstance(result, GeometryResult):
        raise TypeError(
            f"geometry provider returned {type(result).__name__}; expected GeometryResult"
        )
    return validate_geometry_result(result, request=request) if validate else result


def internal_geometry_from_result(result: GeometryResult) -> FluxTubeGeometry:
    """Derive all solver-specific coefficients in the single canonical mapper."""

    return map_physical_to_internal_geometry(result.physical, result.parallel_grid)


def validate_geometry_result(
    result: GeometryResult,
    *,
    request: GeometryRequest | None = None,
) -> GeometryResult:
    """Validate a provider result with actionable schema and array errors."""

    grid = result.parallel_grid
    physical = result.physical
    metadata = result.metadata
    n_z = int(np.asarray(grid.z).shape[0])
    expected_shape = (n_z,)

    if request is not None:
        if n_z != request.n_z:
            raise ValueError(f"provider returned n_z={n_z}; request requires n_z={request.n_z}")
        if metadata.radial_coordinate != request.radial_coordinate:
            raise ValueError(
                "provider radial coordinate mismatch: "
                f"{metadata.radial_coordinate!r} != {request.radial_coordinate!r}"
            )
        if metadata.topology != request.topology:
            raise ValueError(
                f"provider topology mismatch: {metadata.topology!r} != {request.topology!r}"
            )
        if metadata.parallel_coordinate != request.parallel_coordinate:
            raise ValueError("provider parallel coordinate does not match request")
        if metadata.parallel_coordinate_unit != request.parallel_coordinate_unit:
            raise ValueError("provider parallel coordinate unit does not match request")
        if metadata.endpoint_policy != request.endpoint_policy:
            raise ValueError(
                "provider endpoint policy mismatch: "
                f"{metadata.endpoint_policy!r} != {request.endpoint_policy!r}"
            )
        if not np.allclose(np.asarray(physical.alpha), request.alpha, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"provider alpha does not match requested alpha={request.alpha}")
        if not np.allclose(
            np.asarray(physical.rho), request.radial_value, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"provider radial values do not match requested value={request.radial_value}"
            )

    if grid.topology != metadata.topology or physical.topology != metadata.topology:
        raise ValueError(
            "parallel topology is inconsistent across grid, physical geometry, and metadata"
        )
    if physical.radial_coordinate != metadata.radial_coordinate:
        raise ValueError("physical geometry radial_coordinate does not match metadata")
    if physical.normalization != metadata.normalization.name:
        raise ValueError(
            f"incompatible physical normalization {physical.normalization!r}; "
            f"metadata declares {metadata.normalization.name!r}"
        )
    if physical.nfp != metadata.nfp:
        raise ValueError(f"physical nfp={physical.nfp} does not match metadata nfp={metadata.nfp}")
    if not np.isclose(physical.field_periods, metadata.field_periods):
        raise ValueError("physical field_periods does not match metadata")
    if physical.endpoint_policy != metadata.endpoint_policy:
        raise ValueError("physical endpoint_policy does not match metadata")
    if physical.twist_and_shift != metadata.linking.twist_and_shift:
        raise ValueError("physical twist_and_shift does not match metadata linking")
    if physical.provider != metadata.provenance.provider:
        raise ValueError("physical provider does not match metadata provenance")

    for name in ("iota", "shear"):
        array = np.asarray(getattr(physical, name))
        if array.shape not in ((), expected_shape):
            raise ValueError(
                f"physical geometry field {name!r} must be scalar or shape "
                f"{expected_shape}; got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"physical geometry field {name!r} contains non-finite values")

    for name in _PHYSICAL_ARRAY_FIELDS:
        array = np.asarray(getattr(physical, name))
        if array.shape != expected_shape:
            raise ValueError(f"physical geometry field {name!r} must have shape {expected_shape}; got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"physical geometry field {name!r} contains non-finite values")

    for name in ("z", "w_z"):
        grid_array = np.asarray(getattr(grid, name))
        physical_array = np.asarray(getattr(physical, name))
        if grid_array.shape != expected_shape:
            raise ValueError(f"parallel grid {name!r} must have shape {expected_shape}; got {grid_array.shape}")
        if not np.allclose(grid_array, physical_array, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"physical geometry {name!r} does not match the returned parallel grid")

    if np.asarray(grid.D_z).shape != (n_z, n_z):
        raise ValueError(f"parallel derivative matrix must have shape {(n_z, n_z)}")
    if not np.all(np.asarray(grid.w_z) > 0.0):
        raise ValueError("parallel quadrature weights must be positive")
    if not np.all(np.asarray(physical.B) > 0.0):
        raise ValueError("physical magnetic-field magnitude B must be positive")

    determinant = (
        np.asarray(physical.grad_psi_sq) * np.asarray(physical.grad_alpha_sq)
        - np.asarray(physical.grad_psi_dot_grad_alpha) ** 2
    )
    if np.any(np.asarray(physical.grad_psi_sq) <= 0.0):
        raise ValueError("grad_psi_sq must be positive")
    if np.any(np.asarray(physical.grad_alpha_sq) <= 0.0):
        raise ValueError("grad_alpha_sq must be positive")
    if np.any(determinant < -1.0e-12):
        raise ValueError("perpendicular metric is not positive semidefinite")

    if metadata.topology == "periodic" and metadata.endpoint_policy == "exclude":
        z = np.asarray(grid.z)
        covered_span = float(z[-1] - z[0])
        if covered_span >= metadata.coordinate_period * (1.0 - 1.0e-12):
            raise ValueError(
                "periodic grid includes a duplicate upper endpoint; endpoint_policy='exclude'"
            )

    return result


def cache_path_is_external(path: str | Path, *, repository_root: str | Path) -> Path:
    """Validate that an optional provider cache is outside the source repository."""

    resolved = Path(path).expanduser().resolve()
    root = Path(repository_root).expanduser().resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError(f"geometry cache must be outside the source repository: {resolved}")
    return resolved


def write_geometry_result_cache(
    result: GeometryResult,
    path: str | Path,
    *,
    repository_root: str | Path,
    overwrite: bool = False,
) -> Path:
    """Serialize a validated result into an explicit external cache directory."""

    validate_geometry_result(result)
    cache = cache_path_is_external(path, repository_root=repository_root)
    metadata_path = cache / "metadata.json"
    arrays_path = cache / "arrays.npz"
    if not overwrite and (metadata_path.exists() or arrays_path.exists()):
        raise FileExistsError(f"geometry cache already exists: {cache}")
    cache.mkdir(parents=True, exist_ok=True)

    physical = result.physical
    payload = {
        "format": "stellarator_gk_geometry_cache_v1",
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "metadata": asdict(result.metadata),
        "parallel_static": {
            "backend": result.parallel_grid.backend,
            "topology": result.parallel_grid.topology,
        },
        "physical_static": {
            "nfp": physical.nfp,
            "field_periods": physical.field_periods,
            "topology": physical.topology,
            "endpoint_policy": physical.endpoint_policy,
            "twist_and_shift": physical.twist_and_shift,
            "normalization": physical.normalization,
            "provider": physical.provider,
            "radial_coordinate": physical.radial_coordinate,
            "source": physical.source,
        },
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        arrays_path,
        grid_z=np.asarray(result.parallel_grid.z),
        grid_w_z=np.asarray(result.parallel_grid.w_z),
        grid_D_z=np.asarray(result.parallel_grid.D_z),
        grid_modal_transform=np.asarray(result.parallel_grid.modal_transform),
        grid_inverse_modal_transform=np.asarray(result.parallel_grid.inverse_modal_transform),
        **{
            f"physical_{name}": np.asarray(getattr(physical, name))
            for name in (*_PHYSICAL_ARRAY_FIELDS, "iota", "shear")
        },
    )
    return cache


def load_geometry_result_cache(
    path: str | Path,
    *,
    request: GeometryRequest | None = None,
) -> GeometryResult:
    """Load an explicit cache as a non-differentiable provider result."""

    cache = Path(path).expanduser().resolve()
    metadata_path = cache / "metadata.json"
    arrays_path = cache / "arrays.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise ValueError(f"geometry cache requires metadata.json and arrays.npz: {cache}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("format") != "stellarator_gk_geometry_cache_v1":
        raise ValueError(f"unsupported geometry cache format in {metadata_path}")
    if payload.get("schema_version") != GEOMETRY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported cached geometry schema {payload.get('schema_version')}; "
            f"expected {GEOMETRY_SCHEMA_VERSION}"
        )
    metadata = _metadata_from_dict(payload["metadata"])
    metadata = replace(metadata, differentiable=False)
    parallel_static = payload["parallel_static"]
    physical_static = payload["physical_static"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        required = {
            "grid_z",
            "grid_w_z",
            "grid_D_z",
            "grid_modal_transform",
            "grid_inverse_modal_transform",
            *(f"physical_{name}" for name in (*_PHYSICAL_ARRAY_FIELDS, "iota", "shear")),
        }
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise ValueError(f"geometry cache is missing arrays: {missing}")
        parallel = ParallelGrid(
            z=jnp.asarray(arrays["grid_z"]),
            w_z=jnp.asarray(arrays["grid_w_z"]),
            D_z=jnp.asarray(arrays["grid_D_z"]),
            modal_transform=jnp.asarray(arrays["grid_modal_transform"]),
            inverse_modal_transform=jnp.asarray(arrays["grid_inverse_modal_transform"]),
            backend=parallel_static["backend"],
            topology=parallel_static["topology"],
        )
        dynamic = {
            name: jnp.asarray(arrays[f"physical_{name}"])
            for name in (*_PHYSICAL_ARRAY_FIELDS, "iota", "shear")
        }
    physical = PhysicalFluxTubeGeometry(
        **dynamic,
        **physical_static,
    )
    return validate_geometry_result(
        GeometryResult(parallel_grid=parallel, physical=physical, metadata=metadata),
        request=request,
    )


def _metadata_from_dict(payload: dict) -> GeometryMetadata:
    normalization_data = dict(payload["normalization"])
    normalization_data["field_units"] = tuple(
        tuple(item) for item in normalization_data["field_units"]
    )
    linking_data = dict(payload["linking"])
    linking_data["kx_shift"] = tuple(linking_data["kx_shift"])
    return GeometryMetadata(
        provenance=GeometryProvenance(**payload["provenance"]),
        schema_version=payload["schema_version"],
        radial_coordinate=payload["radial_coordinate"],
        parallel_coordinate=payload["parallel_coordinate"],
        parallel_coordinate_unit=payload["parallel_coordinate_unit"],
        coordinate_period=payload["coordinate_period"],
        nfp=payload["nfp"],
        field_periods=payload["field_periods"],
        topology=payload["topology"],
        endpoint_policy=payload["endpoint_policy"],
        normalization=GeometryNormalization(**normalization_data),
        signs=GeometrySignConvention(**payload["signs"]),
        linking=GeometryLinking(**linking_data),
        differentiable=payload["differentiable"],
    )
