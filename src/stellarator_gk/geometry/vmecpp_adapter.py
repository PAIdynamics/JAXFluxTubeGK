"""Direct in-memory VMEC++ equilibrium and field-line geometry provider."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline

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


_MU_0 = 4.0e-7 * np.pi
_W7X_NAMES = {"w7-x", "w7x", "w7x-standard"}


@dataclass(frozen=True)
class VmecppGeometryProvider:
    """Run or consume VMEC++ in memory and reconstruct physical field-line arrays."""

    output: object | None = None
    vmec_input: object | None = None
    wout_path: str | Path | None = None
    runner: Callable | None = None
    named_loader: Callable | None = None
    wout_loader: Callable | None = None
    max_threads: int | None = None
    provider_version: str | None = None
    revision: str = "unknown"

    def __post_init__(self) -> None:
        sources = sum(
            value is not None for value in (self.output, self.vmec_input, self.wout_path)
        )
        if sources > 1:
            raise ValueError(
                "VmecppGeometryProvider accepts at most one of output, vmec_input, or wout_path"
            )

    def get_geometry(self, request: GeometryRequest) -> GeometryResult:
        if request.radial_coordinate != "rho":
            raise ValueError("VMEC++ provider currently requires radial_coordinate='rho'")
        if request.parallel_coordinate != "zeta" or request.parallel_coordinate_unit != "radian":
            raise ValueError("VMEC++ provider requires zeta in radians as the parallel coordinate")

        wout, source, command = self._resolve_wout(request)
        arrays = vmec_field_line_from_wout(wout, request)
        parallel = build_parallel_grid(
            ParallelGridSpec(
                n_z=request.n_z,
                z_min=request.z_min,
                z_max=request.z_max,
                topology=request.topology,
                dtype=request.dtype,
            )
        )
        nfp = int(wout.nfp)
        physical = build_physical_flux_tube_geometry_from_coordinate_arrays(
            parallel,
            **arrays,
            nfp=nfp,
            field_periods=request.field_periods,
            endpoint_policy=request.endpoint_policy,
            radial_coordinate=request.radial_coordinate,
            source=source,
            provider="vmecpp",
        )
        version = self.provider_version or _distribution_version("vmecpp")
        metadata = build_geometry_metadata(
            request,
            provenance=GeometryProvenance(
                provider="vmecpp",
                provider_version=version,
                revision=self.revision,
                source=source,
                configuration=request.configuration,
                command=command,
            ),
            nfp=nfp,
            normalization=GeometryNormalization(
                length_reference="VMEC Aminor_p (meter)",
                magnetic_field_reference="VMEC magnetic field (tesla)",
                flux_reference="toroidal flux / (2*pi) (weber)",
            ),
            differentiable=False,
        )
        return GeometryResult(parallel_grid=parallel, physical=physical, metadata=metadata)

    def _resolve_wout(self, request: GeometryRequest):
        if self.output is not None:
            return _as_wout(self.output), "in-memory VmecOutput", "consume in-memory output"
        if self.wout_path is not None:
            loader = self.wout_loader or _import_vmecpp().VmecWOut.from_wout_file
            path = Path(self.wout_path)
            return loader(path), str(path), "VmecWOut.from_wout_file(...)"

        vmec_input = self.vmec_input
        source = "in-memory VmecInput"
        if vmec_input is None:
            name = _canonical_configuration(request.configuration)
            loader = self.named_loader or _import_vmecpp().named_configuration
            vmec_input = loader(name)
            source = f"installed VMEC++ named configuration {name}"
        runner = self.runner or _import_vmecpp().run
        output = runner(vmec_input, max_threads=self.max_threads, verbose=False)
        return _as_wout(output), source, "vmecpp.run(...)"


def vmec_field_line_from_wout(wout, request: GeometryRequest) -> dict[str, object]:
    """Evaluate the physical contract directly from VMEC Fourier output in memory."""

    if bool(wout.lasym):
        raise NotImplementedError(
            "VMEC++ asymmetric equilibria are not supported yet; lasym must be false"
        )
    s = float(request.radial_value) ** 2
    if not 0.0 < s < 1.0:
        raise ValueError("VMEC++ field-line sampling requires 0 < rho < 1")

    zeta = np.linspace(request.z_min, request.z_max, request.n_z, endpoint=False)
    full_s = np.linspace(0.0, 1.0, int(wout.ns))
    iota = _half_grid_value(wout.iotas, full_s, s)
    d_iota_ds = _full_grid_derivative_on_half_grid(wout.iotaf, full_s, s)
    theta_pest = request.alpha + iota * zeta

    modes = np.asarray(wout.xm, dtype=float)
    toroidal_modes = np.asarray(wout.xn, dtype=float)
    rmnc, d_rmnc_ds = _full_grid_coefficients(wout.rmnc, full_s, s)
    zmns, d_zmns_ds = _full_grid_coefficients(wout.zmns, full_s, s)
    lmns, d_lmns_ds = _full_grid_coefficients(wout.lmns_full, full_s, s)

    theta = _invert_pest_angle(theta_pest, zeta, modes, toroidal_modes, lmns)
    phase = modes[:, None] * theta[None, :] - toroidal_modes[:, None] * zeta[None, :]
    cosine = np.cos(phase)
    sine = np.sin(phase)

    R = rmnc @ cosine
    R_s = d_rmnc_ds @ cosine
    R_theta = -(rmnc * modes) @ sine
    R_zeta = (rmnc * toroidal_modes) @ sine
    Z_s = d_zmns_ds @ sine
    Z_theta = (zmns * modes) @ cosine
    Z_zeta = -(zmns * toroidal_modes) @ cosine
    lambda_s = d_lmns_ds @ sine
    lambda_theta = (lmns * modes) @ cosine
    lambda_zeta = -(lmns * toroidal_modes) @ cosine
    pest_theta_factor = 1.0 + lambda_theta
    if np.any(np.abs(pest_theta_factor) < 1.0e-10):
        raise ValueError("VMEC theta-to-PEST map is singular on the requested field line")

    cos_zeta = np.cos(zeta)
    sin_zeta = np.sin(zeta)
    r_s = np.column_stack((R_s * cos_zeta, R_s * sin_zeta, Z_s))
    r_theta = np.column_stack((R_theta * cos_zeta, R_theta * sin_zeta, Z_theta))
    r_zeta = np.column_stack(
        (
            R_zeta * cos_zeta - R * sin_zeta,
            R_zeta * sin_zeta + R * cos_zeta,
            Z_zeta,
        )
    )

    orientation = float(getattr(wout, "signgs", 1.0))
    psi_s = orientation * _profile_derivative(
        np.asarray(wout.phi) / (2.0 * np.pi), full_s, s
    )
    if abs(psi_s) < 1.0e-14:
        raise ValueError("VMEC toroidal-flux derivative vanishes on the requested surface")
    r_theta_pest = r_theta / pest_theta_factor[:, None]
    r_psi = (
        r_s - r_theta * (lambda_s / pest_theta_factor)[:, None]
    ) / psi_s
    r_zeta_pest = r_zeta - r_theta * (lambda_zeta / pest_theta_factor)[:, None]

    jacobian = _dot(r_psi, np.cross(r_theta_pest, r_zeta_pest))
    nyq_m = np.asarray(wout.xm_nyq, dtype=float)
    nyq_n = np.asarray(wout.xn_nyq, dtype=float)
    nyq_phase = nyq_m[:, None] * theta[None, :] - nyq_n[:, None] * zeta[None, :]
    nyq_cosine = np.cos(nyq_phase)
    nyq_sine = np.sin(nyq_phase)
    if hasattr(wout, "gmnc"):
        gmnc, _ = _half_grid_coefficients(wout.gmnc, full_s, s)
        sqrt_g = gmnc @ nyq_cosine
        jacobian = sqrt_g / (psi_s * pest_theta_factor)
    if np.any(np.abs(jacobian) < 1.0e-14):
        raise ValueError("VMEC physical coordinate Jacobian vanishes on the field line")
    grad_psi = np.cross(r_theta_pest, r_zeta_pest) / jacobian[:, None]
    grad_theta_pest = np.cross(r_zeta_pest, r_psi) / jacobian[:, None]
    grad_zeta = np.cross(r_psi, r_theta_pest) / jacobian[:, None]

    d_iota_dpsi = d_iota_ds / psi_s
    grad_alpha = (
        grad_theta_pest
        - iota * grad_zeta
        - (zeta * d_iota_dpsi)[:, None] * grad_psi
    )

    B_vector = (r_zeta_pest + iota * r_theta_pest) / jacobian[:, None]
    b = B_vector / np.linalg.norm(B_vector, axis=1)[:, None]

    bmnc, d_bmnc_ds = _half_grid_coefficients(wout.bmnc, full_s, s)
    B = bmnc @ nyq_cosine
    B_s = d_bmnc_ds @ nyq_cosine
    B_theta = -(bmnc * nyq_m) @ nyq_sine
    B_zeta = (bmnc * nyq_n) @ nyq_sine
    B_psi = (B_s - B_theta * lambda_s / pest_theta_factor) / psi_s
    B_theta_pest = B_theta / pest_theta_factor
    B_zeta_pest = B_zeta - B_theta * lambda_zeta / pest_theta_factor
    grad_B = (
        B_psi[:, None] * grad_psi
        + B_theta_pest[:, None] * grad_theta_pest
        + B_zeta_pest[:, None] * grad_zeta
    )

    d_pressure_ds = _full_grid_derivative_on_half_grid(wout.presf, full_s, s)
    pressure_gradient = d_pressure_ds / psi_s
    b_cross_grad_B = np.cross(b, grad_B)
    b_cross_grad_psi = np.cross(b, grad_psi)
    b_cross_kappa = (
        b_cross_grad_B / B[:, None]
        + (_MU_0 * pressure_gradient / B**2)[:, None] * b_cross_grad_psi
    )
    B_cross_grad_B = B[:, None] * b_cross_grad_B

    shear = -2.0 * s * d_iota_ds / iota
    length_reference = float(wout.Aminor_p)
    field_reference = (
        2.0
        * abs(float(np.asarray(wout.phi)[-1]) / (2.0 * np.pi))
        / length_reference**2
    )
    if length_reference <= 0.0 or field_reference <= 0.0:
        raise ValueError("VMEC Aminor_p and edge toroidal flux must define positive references")
    return {
        "theta": theta_pest,
        "phi": zeta,
        "alpha": request.alpha,
        "rho": request.radial_value,
        "iota": iota,
        "shear": shear,
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


def _invert_pest_angle(theta_pest, zeta, modes, toroidal_modes, lmns):
    theta = np.asarray(theta_pest, dtype=float).copy()
    for _ in range(20):
        phase = modes[:, None] * theta[None, :] - toroidal_modes[:, None] * zeta[None, :]
        residual = theta + lmns @ np.sin(phase) - theta_pest
        derivative = 1.0 + (lmns * modes) @ np.cos(phase)
        step = residual / derivative
        theta -= step
        if np.max(np.abs(step)) < 2.0e-13:
            return theta
    raise RuntimeError("VMEC theta-to-PEST inversion did not converge")


def _full_grid_coefficients(values, grid, s):
    coefficients = np.asarray(values)
    value = _linear_interpolate(coefficients, grid, s)
    derivative = _linear_interpolate(
        np.diff(coefficients, axis=1) / (grid[1] - grid[0]),
        0.5 * (grid[1:] + grid[:-1]),
        s,
    )
    return value, derivative


def _half_grid_coefficients(values, full_grid, s):
    half_grid = 0.5 * (full_grid[1:] + full_grid[:-1])
    coefficients = np.asarray(values)[:, 1:]
    value = _linear_interpolate(coefficients, half_grid, s)
    derivative_on_full_grid = np.empty_like(np.asarray(values))
    derivative_on_full_grid[:, 1:-1] = (
        np.asarray(values)[:, 2:] - np.asarray(values)[:, 1:-1]
    ) / (full_grid[1] - full_grid[0])
    derivative_on_full_grid[:, 0] = derivative_on_full_grid[:, 1]
    derivative_on_full_grid[:, -1] = derivative_on_full_grid[:, -2]
    derivative = _linear_interpolate(derivative_on_full_grid, full_grid, s)
    return value, derivative


def _half_grid_value(values, full_grid, s):
    half_grid = 0.5 * (full_grid[1:] + full_grid[:-1])
    profile = np.asarray(values)[1:]
    return float(np.interp(s, half_grid, profile))


def _full_grid_derivative_on_half_grid(values, full_grid, s):
    derivative = np.diff(np.asarray(values)) / (full_grid[1] - full_grid[0])
    half_grid = 0.5 * (full_grid[1:] + full_grid[:-1])
    return float(np.interp(s, half_grid, derivative))


def _linear_interpolate(values, grid, s):
    upper = int(np.searchsorted(grid, s, side="right"))
    upper = min(max(upper, 1), len(grid) - 1)
    lower = upper - 1
    weight = (s - grid[lower]) / (grid[upper] - grid[lower])
    return values[:, lower] * (1.0 - weight) + values[:, upper] * weight


def _profile_value(values, grid, s):
    return float(CubicSpline(grid, np.asarray(values))(s))


def _profile_derivative(values, grid, s):
    return float(CubicSpline(grid, np.asarray(values))(s, 1))


def _dot(left, right):
    return np.einsum("ij,ij->i", left, right)


def _canonical_configuration(configuration: str) -> str:
    normalized = configuration.lower()
    if normalized in _W7X_NAMES:
        return "w7x-standard"
    return normalized


def _as_wout(output):
    return output.wout if hasattr(output, "wout") else output


def _import_vmecpp():
    try:
        return importlib.import_module("vmecpp")
    except ImportError as exc:
        raise ImportError(
            "VMEC++ is required for live equilibrium geometry; install the 'vmecpp' "
            "or 'mhd' extra, or run scripts/bootstrap_dependencies.py --profile mhd"
        ) from exc


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


__all__ = ["VmecppGeometryProvider", "vmec_field_line_from_wout"]
