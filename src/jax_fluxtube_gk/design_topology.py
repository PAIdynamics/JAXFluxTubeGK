"""Fixed-topology contracts for differentiable design evaluations."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib

import numpy as np

from .types import FourierGrid, ModeConnectivity, ParallelGrid, VelocityGrid


DESIGN_TOPOLOGY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OptimizationTopologyContract:
    """Serializable signature of every discrete choice held fixed by AD."""

    schema_version: int
    velocity_shape: tuple[int, int]
    velocity_backend: str
    velocity_nodes_digest: str
    parallel_size: int
    parallel_backend: str
    parallel_topology: str
    parallel_nodes_digest: str
    fourier_shape: tuple[int, int]
    ikxspace: int
    fourier_modes_digest: str
    connectivity_digest: str
    provider_topology: tuple[object, ...]


class TopologyChangeError(ValueError):
    """Raised when an optimizer tries to reuse AD state after remeshing."""

    def __init__(self, changed_fields: tuple[str, ...]):
        """Record which topology contract fields changed and build a descriptive error message."""
        self.changed_fields = changed_fields
        super().__init__(
            "optimization topology changed in "
            + ", ".join(changed_fields)
            + "; rebuild grids, connectivity, precomputes, and compiled objectives"
        )


def build_optimization_topology_contract(
    velocity_grid: VelocityGrid,
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    *,
    connectivity: ModeConnectivity | None = None,
    geometry_metadata=None,
) -> OptimizationTopologyContract:
    """Capture the discrete mesh, linking, and provider topology outside JAX."""

    return OptimizationTopologyContract(
        schema_version=DESIGN_TOPOLOGY_SCHEMA_VERSION,
        velocity_shape=(int(velocity_grid.vpar.shape[0]), int(velocity_grid.mu.shape[0])),
        velocity_backend=str(velocity_grid.backend),
        velocity_nodes_digest=_arrays_digest(velocity_grid.vpar, velocity_grid.mu),
        parallel_size=int(parallel_grid.z.shape[0]),
        parallel_backend=str(parallel_grid.backend),
        parallel_topology=str(parallel_grid.topology),
        parallel_nodes_digest=_arrays_digest(parallel_grid.z),
        fourier_shape=(int(fourier_grid.kx.shape[0]), int(fourier_grid.ky.shape[0])),
        ikxspace=int(fourier_grid.ikxspace),
        fourier_modes_digest=_arrays_digest(fourier_grid.kx, fourier_grid.ky),
        connectivity_digest=_connectivity_digest(connectivity),
        provider_topology=_provider_topology(geometry_metadata),
    )


def optimization_topology_changes(
    reference: OptimizationTopologyContract,
    candidate: OptimizationTopologyContract,
) -> tuple[str, ...]:
    """Return contract fields that require rebuilding the differentiated solve."""

    return tuple(
        field.name
        for field in fields(OptimizationTopologyContract)
        if getattr(reference, field.name) != getattr(candidate, field.name)
    )


def assert_fixed_optimization_topology(
    reference: OptimizationTopologyContract,
    candidate: OptimizationTopologyContract,
) -> None:
    """Reject reuse of compiled objectives or gradients after a topology change."""

    changed = optimization_topology_changes(reference, candidate)
    if changed:
        raise TopologyChangeError(changed)


def _arrays_digest(*arrays) -> str:
    """Return a SHA-256 hex digest of the given arrays' shapes, dtypes, and raw bytes."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _connectivity_digest(connectivity: ModeConnectivity | None) -> str:
    """Return a digest of the connectivity's arrays and scalars, or "none" if unset."""
    if connectivity is None:
        return "none"
    return _arrays_digest(
        connectivity.mode_label,
        connectivity.ixplus,
        connectivity.ixminus,
        connectivity.kx_shift,
        connectivity.valid_shift,
        np.asarray(
            [connectivity.ixzero, connectivity.iyzero, connectivity.max_shift],
            dtype=np.int64,
        ),
    )


def _provider_topology(metadata) -> tuple[object, ...]:
    """Flatten geometry provider metadata's schema, topology, and linking fields into a hashable tuple."""
    if metadata is None:
        return ()
    linking = metadata.linking
    return (
        int(metadata.schema_version),
        int(metadata.nfp),
        float(metadata.field_periods),
        str(metadata.topology),
        str(metadata.endpoint_policy),
        str(linking.boundary_condition),
        bool(linking.twist_and_shift),
        tuple(int(value) for value in linking.kx_shift),
    )
