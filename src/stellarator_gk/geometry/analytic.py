"""Differentiable analytic circular and s-alpha geometry fixtures.

The formulas follow the circular/s-alpha geometry used by GKW and mirrored by
Gyaradax.  They provide the internal geometry contract from ``main.tex``:
``B, F, G, E_y, D_x, D_y, g_xx, g_xy, g_yy, w_z``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import jax
import jax.numpy as jnp

from ..types import FourierGrid, GeometryScalarParams, ParallelGrid, _PyTreeDataclass

GeometryModel = Literal["circular", "circ", "s-alpha"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AnalyticGeometry(_PyTreeDataclass):
    """Internal analytic flux-tube geometry arrays on the parallel grid."""

    z: object
    w_z: object
    theta: object
    B: object
    F: object
    G: object
    E_y: object
    D_x: object
    D_y: object
    g_xx: object
    g_xy: object
    g_yy: object
    exb_tensor: object
    magnetic_drift_tensor: object
    model: str

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "z",
        "w_z",
        "theta",
        "B",
        "F",
        "G",
        "E_y",
        "D_x",
        "D_y",
        "g_xx",
        "g_xy",
        "g_yy",
        "exb_tensor",
        "magnetic_drift_tensor",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("model",)


def build_circular_geometry(
    parallel_grid: ParallelGrid,
    params: GeometryScalarParams,
    *,
    model: GeometryModel = "circular",
    sign_B: float = 1.0,
    sign_J: float = 1.0,
) -> AnalyticGeometry:
    """Build differentiable circular or s-alpha analytic geometry.

    ``model="circular"`` uses the finite-aspect-ratio circular model.  The
    accepted alias ``"circ"`` is retained for GKW/Gyaradax terminology.
    ``model="s-alpha"`` uses the simpler s-alpha circular limit.
    """

    normalized_model = _normalize_model(model)
    theta = _poloidal_angle(parallel_grid.z, params.eps, normalized_model)
    circular = _circular_geometry(
        theta,
        params.q,
        params.shat,
        params.eps,
        sign_B=sign_B,
        sign_J=sign_J,
        model=normalized_model,
    )
    exb_tensor, drift_tensor = _calc_geom_tensors(
        circular,
        sign_B=sign_B,
        sign_J=sign_J,
    )
    B = circular["B"]
    metric = circular["metric"]
    g_xy = circular["dzetadeps"]
    return AnalyticGeometry(
        z=parallel_grid.z,
        w_z=parallel_grid.w_z,
        theta=theta,
        B=B,
        F=circular["F"],
        G=circular["G"],
        E_y=-exb_tensor[:, 0, 1],
        D_x=drift_tensor[:, 0],
        D_y=drift_tensor[:, 1],
        g_xx=jnp.ones_like(B),
        g_xy=g_xy,
        g_yy=metric[:, 1, 1],
        exb_tensor=exb_tensor,
        magnetic_drift_tensor=drift_tensor,
        model=normalized_model,
    )


def build_s_alpha_geometry(
    parallel_grid: ParallelGrid,
    params: GeometryScalarParams,
    *,
    sign_B: float = 1.0,
    sign_J: float = 1.0,
) -> AnalyticGeometry:
    """Build the simplified s-alpha analytic geometry."""

    return build_circular_geometry(
        parallel_grid,
        params,
        model="s-alpha",
        sign_B=sign_B,
        sign_J=sign_J,
    )


def k_perp_squared(geometry: AnalyticGeometry, fourier_grid: FourierGrid):
    """Evaluate ``k_perp^2`` on ``(z, kx, ky)`` from the metric contract."""

    kx = fourier_grid.kx[None, :, None]
    ky = fourier_grid.ky[None, None, :]
    return (
        geometry.g_xx[:, None, None] * kx**2
        + 2.0 * geometry.g_xy[:, None, None] * kx * ky
        + geometry.g_yy[:, None, None] * ky**2
    )


def _normalize_model(model: GeometryModel) -> str:
    if model == "circ":
        return "circular"
    if model in ("circular", "s-alpha"):
        return model
    raise ValueError("model must be 'circular', 'circ', or 's-alpha'")


def _poloidal_angle(sgrid, eps, model: str, n_iter: int = 10):
    if model == "s-alpha":
        return 2.0 * jnp.pi * sgrid
    theta = 2.0 * jnp.pi * sgrid
    for _ in range(n_iter):
        theta = 2.0 * jnp.pi * sgrid - eps * jnp.sin(theta)
    return theta


def _dzetadeps(theta, q, shat, eps, sign_B: float, sign_J: float):
    dum2 = jnp.sqrt((1.0 - eps) / (1.0 + eps))
    raw = jnp.arctan(dum2 * jnp.tan(theta / 2.0))
    diffs = raw[1:] - raw[:-1]
    jumps = jnp.where(diffs < 0.0, jnp.pi, 0.0)
    corrections = jnp.concatenate([jnp.zeros(1, dtype=theta.dtype), jnp.cumsum(jumps)])
    dzde = raw + corrections
    dzde = dzde - jnp.pi * jnp.floor((dzde[0] - theta[0] / 2.0) / jnp.pi)

    tan_half = jnp.tan(theta / 2.0)
    correction = (
        eps
        / jnp.sqrt(1.0 - eps**2)
        * tan_half
        / (1.0 + tan_half**2 + eps * (1.0 - tan_half**2))
    )
    return sign_B * sign_J / jnp.pi * q / eps * (shat * dzde - correction)


def _psi_theta_to_psi_s(f_psi, f_theta, theta, eps):
    major_radius = 1.0 + eps * jnp.cos(theta)
    return (
        f_psi - jnp.sin(theta) / major_radius * f_theta,
        2.0 * jnp.pi * f_theta / major_radius,
    )


def _circular_geometry(
    theta,
    q,
    shat,
    eps,
    *,
    sign_B: float,
    sign_J: float,
    model: str,
):
    nz = theta.shape[0]
    finite_epsilon = model != "s-alpha"
    major_radius = 1.0 + eps * jnp.cos(theta)

    if model == "s-alpha":
        normalization = 1.0
        bups = sign_J / (2.0 * jnp.pi * q)
        dpfdpsi = eps / q
    else:
        normalization = jnp.sqrt(1.0 + eps**2 / q**2 / (1.0 - eps**2))
        bups = 1.0 / (2.0 * jnp.pi * q * jnp.sqrt(1.0 - eps**2))
        dpfdpsi = eps / (q * jnp.sqrt(1.0 - eps**2))
    B = normalization / major_radius
    F = bups / B if finite_epsilon else jnp.ones_like(theta) * bups

    if model == "s-alpha":
        dzde = _dzetadeps(theta, q, shat, eps, sign_B, sign_J)
        sgrid = theta / (2.0 * jnp.pi)
        metric = jnp.zeros((nz, 3, 3), dtype=theta.dtype)
        metric = metric.at[:, 0, 0].set(1.0)
        cross_01 = q * shat * sgrid / eps * sign_B * sign_J
        metric = metric.at[:, 0, 1].set(cross_01)
        metric = metric.at[:, 1, 0].set(cross_01)
        metric = metric.at[:, 1, 1].set(
            (q / (2.0 * jnp.pi * eps)) ** 2 * (1.0 + (2.0 * jnp.pi * shat * sgrid) ** 2)
        )
        cross_12 = q / (2.0 * jnp.pi * eps) ** 2 * sign_B * sign_J
        metric = metric.at[:, 1, 2].set(cross_12)
        metric = metric.at[:, 2, 1].set(cross_12)
        metric = metric.at[:, 2, 2].set(1.0 / (2.0 * jnp.pi * eps) ** 2)

        dBdpsi = -jnp.cos(theta)
        dBds = 2.0 * jnp.pi * eps * jnp.sin(theta)
        dRdpsi = jnp.cos(theta)
        dRds = -2.0 * jnp.pi * eps * jnp.sin(theta)
        dZdpsi = jnp.sin(theta)
        dZds = 2.0 * jnp.pi * eps * jnp.cos(theta)
        dzetadeps = metric[:, 0, 1]
    else:
        dzde = _dzetadeps(theta, q, shat, eps, sign_B, sign_J)
        sin_2pi = jnp.sin(theta) / (2.0 * jnp.pi)
        metric = jnp.zeros((nz, 3, 3), dtype=theta.dtype)
        metric = metric.at[:, 0, 0].set(1.0)
        metric = metric.at[:, 0, 1].set(dzde)
        metric = metric.at[:, 1, 0].set(dzde)
        metric = metric.at[:, 0, 2].set(sin_2pi)
        metric = metric.at[:, 2, 0].set(sin_2pi)
        metric = metric.at[:, 1, 1].set(
            (1.0 / (2.0 * jnp.pi * major_radius)) ** 2
            * (1.0 + (1.0 - eps**2) * (q / eps) ** 2)
            + dzde**2
        )
        cross_12 = (
            q * jnp.sqrt(1.0 - eps**2) / (2.0 * jnp.pi * eps) ** 2 * sign_B * sign_J
            + dzde * sin_2pi
        )
        metric = metric.at[:, 1, 2].set(cross_12)
        metric = metric.at[:, 2, 1].set(cross_12)
        metric = metric.at[:, 2, 2].set(
            (1.0 / (2.0 * jnp.pi)) ** 2
            * ((1.0 / eps + jnp.cos(theta)) ** 2 + jnp.sin(theta) ** 2)
        )

        dBdpsi_pt = B * (
            -jnp.cos(theta) / major_radius
            + eps * (1.0 - shat + eps**2 / (1.0 - eps**2))
            / (eps**2 + q**2 * (1.0 - eps**2))
        )
        dBds_pt = B * eps * jnp.sin(theta) / major_radius
        dBdpsi, dBds = _psi_theta_to_psi_s(dBdpsi_pt, dBds_pt, theta, eps)
        dRdpsi, dRds = _psi_theta_to_psi_s(jnp.cos(theta), -eps * jnp.sin(theta), theta, eps)
        dZdpsi, dZds = _psi_theta_to_psi_s(jnp.sin(theta), eps * jnp.cos(theta), theta, eps)
        dzetadeps = dzde

    G = F * dBds / B if finite_epsilon else F * B * dBds

    return {
        "B": B,
        "F": F,
        "G": G,
        "metric": metric,
        "dzetadeps": dzetadeps,
        "major_radius": major_radius,
        "bups": bups,
        "dpfdpsi": dpfdpsi,
        "dBdpsi": dBdpsi,
        "dBds": dBds,
        "dRdpsi": dRdpsi,
        "dRds": dRds,
        "dZdpsi": dZdpsi,
        "dZds": dZds,
        "finite_epsilon": finite_epsilon,
    }


def _calc_geom_tensors(circular, *, sign_B: float, sign_J: float):
    B = circular["B"]
    metric = circular["metric"]
    major_radius = circular["major_radius"]
    finite_epsilon = circular["finite_epsilon"]

    m0 = metric[:, 0, :]
    m1 = metric[:, 1, :]
    exb = m0[:, :, None] * m1[:, None, :] - m1[:, :, None] * m0[:, None, :]
    exb = exb * (sign_J * jnp.pi * circular["dpfdpsi"])
    if finite_epsilon:
        exb = exb / B[:, None, None] ** 2

    e0 = exb[:, :, 0]
    e2 = exb[:, :, 2]
    drift = (
        -2.0 * e0 * circular["dBdpsi"][:, None]
        - 2.0 * e2 * circular["dBds"][:, None]
    )
    if finite_epsilon:
        drift = drift / B[:, None]

    # The remaining tensors will be useful for rotation/extended physics later;
    # Phase 3 exposes the ExB and magnetic-drift tensors only.
    _ = major_radius, sign_B
    return exb, drift

