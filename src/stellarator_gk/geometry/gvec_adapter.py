"""Optional in-memory pyGVEC field-line geometry provider."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ..grids import build_parallel_grid
from ..types import ParallelGridSpec
from .flux_tube import build_physical_flux_tube_geometry_from_coordinate_arrays
from .provider import (
    GeometryNormalization,
    GeometryProvenance,
    GeometryRequest,
    GeometryResult,
    build_geometry_metadata,
)


GVEC_GEOMETRY_COMPUTE_KEYS = (
    "grad_rho",
    "grad_theta_P",
    "grad_zeta",
    "iota",
    "diota_dr",
    "dPhi_dr",
    "Phi_edge",
    "r_minor",
    "B",
    "mod_B",
    "grad_mod_B",
    "kappa_B",
)


@dataclass(frozen=True)
class GvecGeometryProvider:
    """Evaluate a GVEC state on a PEST field line and return the common contract."""

    state: object | None = None
    parameter_file: str | Path | None = None
    state_file: str | Path | None = None
    state_factory: Callable | None = None
    evaluations_pest: Callable | None = None
    provider_version: str | None = None
    revision: str = "unknown"

    def __post_init__(self) -> None:
        if (self.state is None) == (self.parameter_file is None):
            raise ValueError(
                "GvecGeometryProvider requires exactly one of state or parameter_file"
            )
        if self.state_file is not None and self.parameter_file is None:
            raise ValueError("GVEC state_file requires parameter_file")

    def get_geometry(self, request: GeometryRequest) -> GeometryResult:
        if request.radial_coordinate != "rho":
            raise ValueError("GVEC provider currently requires radial_coordinate='rho'")
        if request.parallel_coordinate != "zeta" or request.parallel_coordinate_unit != "radian":
            raise ValueError("GVEC provider requires zeta in radians as the parallel coordinate")

        state = self.state
        if state is None:
            factory = self.state_factory or _import_gvec().State
            state = factory(self.parameter_file, self.state_file)
        zeta = np.linspace(request.z_min, request.z_max, request.n_z, endpoint=False)
        iota_data = state.evaluate(
            "iota",
            rho=np.asarray([request.radial_value]),
            theta=None,
            zeta=None,
        )
        iota = _scalar(iota_data, "iota")
        theta_pest = request.alpha + iota * zeta
        evaluations_pest = self.evaluations_pest or _import_gvec().EvaluationsPEST
        evaluated = evaluations_pest(
            rho=np.asarray([request.radial_value]),
            theta_P=theta_pest[None, None, :],
            zeta=zeta,
            state=state,
        )
        state.compute(evaluated, *GVEC_GEOMETRY_COMPUTE_KEYS)
        arrays = gvec_geometry_arrays_from_data(
            evaluated,
            zeta=zeta,
            rho=request.radial_value,
            alpha=request.alpha,
        )
        parallel = build_parallel_grid(
            ParallelGridSpec(
                n_z=request.n_z,
                z_min=request.z_min,
                z_max=request.z_max,
                topology=request.topology,
                dtype=request.dtype,
            )
        )
        source = (
            "in-memory GVEC state"
            if self.parameter_file is None
            else f"GVEC state {Path(self.parameter_file)}"
        )
        nfp = int(state.nfp)
        physical = build_physical_flux_tube_geometry_from_coordinate_arrays(
            parallel,
            **arrays,
            nfp=nfp,
            field_periods=request.field_periods,
            endpoint_policy=request.endpoint_policy,
            radial_coordinate=request.radial_coordinate,
            source=source,
            provider="gvec",
        )
        metadata = build_geometry_metadata(
            request,
            provenance=GeometryProvenance(
                provider="gvec",
                provider_version=self.provider_version or _distribution_version("gvec"),
                revision=self.revision,
                source=source,
                configuration=request.configuration,
                command="pyGVEC EvaluationsPEST + State.compute",
            ),
            nfp=nfp,
            normalization=GeometryNormalization(
                length_reference="GVEC effective minor radius r_minor (meter)",
                magnetic_field_reference="2*abs(GVEC Phi_edge)/r_minor^2 (tesla)",
                flux_reference="toroidal flux / (2*pi) (weber)",
            ),
            differentiable=False,
        )
        return GeometryResult(parallel_grid=parallel, physical=physical, metadata=metadata)


def gvec_geometry_arrays_from_data(
    data: Mapping[str, object],
    *,
    zeta,
    rho: float,
    alpha: float,
) -> dict[str, object]:
    """Map computed GVEC vectors to the physical flux-tube array contract."""

    zeta = np.asarray(zeta, dtype=float)
    shape = zeta.shape
    iota = _scalar(data, "iota")
    diota_dr = _scalar(data, "diota_dr")
    d_phi_dr = _scalar(data, "dPhi_dr")
    phi_edge = _scalar(data, "Phi_edge")
    length_reference = _scalar(data, "r_minor")
    if abs(d_phi_dr) < 1.0e-14:
        raise ValueError("GVEC dPhi_dr vanishes on the requested surface")
    if length_reference <= 0.0 or abs(phi_edge) < 1.0e-14:
        raise ValueError("GVEC r_minor and Phi_edge must define positive references")
    field_reference = 2.0 * abs(phi_edge) / length_reference**2

    grad_rho = _vector(data, "grad_rho", shape)
    grad_theta_pest = _vector(data, "grad_theta_P", shape)
    grad_zeta = _vector(data, "grad_zeta", shape)
    B_vector = _vector(data, "B", shape)
    B = _array(data, "mod_B", shape)
    grad_B = _vector(data, "grad_mod_B", shape)
    kappa = _vector(data, "kappa_B", shape)
    grad_psi = d_phi_dr * grad_rho
    grad_alpha = grad_theta_pest - iota * grad_zeta - (
        zeta * diota_dr
    )[:, None] * grad_rho
    b = B_vector / B[:, None]
    B_cross_grad_B = np.cross(B_vector, grad_B)
    b_cross_kappa = np.cross(b, kappa)

    return {
        "theta": alpha + iota * zeta,
        "phi": zeta,
        "alpha": alpha,
        "rho": rho,
        "iota": iota,
        "shear": -rho * diota_dr / iota,
        "B": B / field_reference,
        "b_dot_grad_z": length_reference * _dot(b, grad_zeta),
        "grad_psi_sq": _dot(grad_psi, grad_psi)
        / (length_reference**2 * field_reference**2),
        "grad_alpha_sq": length_reference**2 * _dot(grad_alpha, grad_alpha),
        "grad_psi_dot_grad_alpha": _dot(grad_psi, grad_alpha) / field_reference,
        "B_cross_gradB_dot_grad_psi": _dot(B_cross_grad_B, grad_psi)
        / field_reference**3,
        "B_cross_gradB_dot_grad_alpha": (
            length_reference**2
            * _dot(B_cross_grad_B, grad_alpha)
            / field_reference**2
        ),
        "b_cross_kappa_dot_grad_psi": _dot(b_cross_kappa, grad_psi)
        / field_reference,
        "b_cross_kappa_dot_grad_alpha": length_reference**2
        * _dot(b_cross_kappa, grad_alpha),
    }


def _array(data, name, shape):
    value = _value(data, name)
    array = np.asarray(value).squeeze()
    if array.shape == ():
        return np.full(shape, float(array))
    if array.shape != shape:
        raise ValueError(f"GVEC quantity {name!r} must have shape {shape}; got {array.shape}")
    return array


def _vector(data, name, shape):
    array = np.asarray(_value(data, name)).squeeze()
    expected = shape + (3,)
    if array.shape == (3,) + shape:
        array = np.moveaxis(array, 0, -1)
    if array.shape != expected:
        raise ValueError(f"GVEC quantity {name!r} must have shape {expected}; got {array.shape}")
    return array


def _scalar(data, name):
    array = np.asarray(_value(data, name)).squeeze()
    if array.shape != ():
        raise ValueError(f"GVEC quantity {name!r} must be scalar; got {array.shape}")
    return float(array)


def _value(data, name):
    try:
        value = data[name]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"GVEC data is missing required quantity {name!r}") from exc
    return getattr(value, "data", value)


def _dot(left, right):
    return np.einsum("ij,ij->i", left, right)


def _import_gvec():
    try:
        return importlib.import_module("gvec")
    except ImportError as exc:
        raise ImportError(
            "pyGVEC is required for GVEC geometry; install the 'gvec' or 'mhd' extra, "
            "or run scripts/bootstrap_dependencies.py --profile mhd"
        ) from exc


def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


__all__ = [
    "GVEC_GEOMETRY_COMPUTE_KEYS",
    "GvecGeometryProvider",
    "gvec_geometry_arrays_from_data",
]
