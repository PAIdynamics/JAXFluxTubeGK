"""Synthetic and in-memory physical-array geometry providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp

from ..grids import build_parallel_grid
from ..types import ParallelGrid, ParallelGridSpec
from .flux_tube import build_physical_flux_tube_geometry_from_coordinate_arrays
from .provider import (
    GeometryProvenance,
    GeometryRequest,
    GeometryResult,
    build_geometry_metadata,
)


PhysicalArrayFactory = Callable[[GeometryRequest, ParallelGrid], Mapping[str, object]]


@dataclass(frozen=True)
class PhysicalArrayGeometryProvider:
    """Adapt in-memory physical arrays without exposing solver coefficients."""

    arrays: Mapping[str, object] | PhysicalArrayFactory
    provider_name: str = "physical-arrays"
    provider_version: str = "unknown"
    revision: str = "unknown"
    source: str = "in-memory physical arrays"
    iota: object = 1.0
    shear: object = 0.0
    nfp: int = 1
    differentiable: bool = False

    def __post_init__(self) -> None:
        if not self.provider_name:
            raise ValueError("physical-array provider_name must be non-empty")
        if self.nfp < 1:
            raise ValueError("physical-array provider nfp must be at least 1")

    def get_geometry(self, request: GeometryRequest) -> GeometryResult:
        parallel = _parallel_grid(request)
        values = self.arrays(request, parallel) if callable(self.arrays) else self.arrays
        physical = build_physical_flux_tube_geometry_from_coordinate_arrays(
            parallel,
            theta=values.get("theta", parallel.z),
            phi=values.get("phi", (parallel.z - request.alpha) / self.iota),
            alpha=values.get("alpha", request.alpha),
            rho=values.get("rho", request.radial_value),
            iota=values.get("iota", self.iota),
            shear=values.get("shear", self.shear),
            B=_required(values, "B"),
            b_dot_grad_z=_required(values, "b_dot_grad_z"),
            grad_psi_sq=_required(values, "grad_psi_sq"),
            grad_alpha_sq=_required(values, "grad_alpha_sq"),
            grad_psi_dot_grad_alpha=_required(values, "grad_psi_dot_grad_alpha"),
            B_cross_gradB_dot_grad_psi=_required(
                values, "B_cross_gradB_dot_grad_psi"
            ),
            B_cross_gradB_dot_grad_alpha=_required(
                values, "B_cross_gradB_dot_grad_alpha"
            ),
            b_cross_kappa_dot_grad_psi=_required(values, "b_cross_kappa_dot_grad_psi"),
            b_cross_kappa_dot_grad_alpha=_required(
                values, "b_cross_kappa_dot_grad_alpha"
            ),
            nfp=self.nfp,
            field_periods=request.field_periods,
            endpoint_policy=request.endpoint_policy,
            radial_coordinate=request.radial_coordinate,
            source=self.source,
            provider=self.provider_name,
        )
        metadata = build_geometry_metadata(
            request,
            provenance=GeometryProvenance(
                provider=self.provider_name,
                provider_version=self.provider_version,
                revision=self.revision,
                source=self.source,
                configuration=request.configuration,
            ),
            nfp=self.nfp,
            differentiable=self.differentiable,
        )
        return GeometryResult(parallel_grid=parallel, physical=physical, metadata=metadata)


@dataclass(frozen=True)
class SyntheticGeometryProvider:
    """Small differentiable provider for standalone solver and interface tests."""

    magnetic_field_amplitude: object = 0.1
    iota: object = 0.7
    shear: object = 0.0
    nfp: int = 1

    def get_geometry(self, request: GeometryRequest) -> GeometryResult:
        def arrays(_request: GeometryRequest, parallel: ParallelGrid):
            ones = jnp.ones_like(parallel.z)
            zeros = jnp.zeros_like(parallel.z)
            return {
                "theta": request.alpha + self.iota * parallel.z,
                "phi": parallel.z,
                "alpha": request.alpha,
                "rho": request.radial_value,
                "B": 1.0 + self.magnetic_field_amplitude * jnp.cos(parallel.z),
                "b_dot_grad_z": ones,
                "grad_psi_sq": ones,
                "grad_alpha_sq": (1.2 + 0.1 * self.shear**2) * ones,
                "grad_psi_dot_grad_alpha": 0.05 * self.shear * ones,
                "B_cross_gradB_dot_grad_psi": zeros,
                "B_cross_gradB_dot_grad_alpha": zeros,
                "b_cross_kappa_dot_grad_psi": zeros,
                "b_cross_kappa_dot_grad_alpha": zeros,
            }

        return PhysicalArrayGeometryProvider(
            arrays=arrays,
            provider_name="synthetic",
            provider_version="1",
            revision="builtin",
            source="builtin synthetic cosine-B geometry",
            iota=self.iota,
            shear=self.shear,
            nfp=self.nfp,
            differentiable=True,
        ).get_geometry(request)


def _parallel_grid(request: GeometryRequest) -> ParallelGrid:
    return build_parallel_grid(
        ParallelGridSpec(
            n_z=request.n_z,
            z_min=request.z_min,
            z_max=request.z_max,
            topology=request.topology,
            dtype=request.dtype,
        )
    )


def _required(values: Mapping[str, object], name: str):
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"physical-array provider is missing required field {name!r}") from exc
