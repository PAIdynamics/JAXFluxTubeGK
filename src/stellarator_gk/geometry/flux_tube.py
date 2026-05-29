"""Boozer/stellarator flux-tube geometry data models and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import jax
import jax.numpy as jnp

from ..grids import build_parallel_grid
from ..types import ParallelGrid, ParallelGridSpec, _PyTreeDataclass

RadialCoordinate = Literal["rho", "psi", "x"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BoozerSurface(_PyTreeDataclass):
    """Minimal Boozer-surface Fourier model for field-line sampling tests."""

    iota: float
    B0: float = 1.0
    B_cos: object = ()
    B_sin: object = ()
    m_modes: tuple[int, ...] = ()
    n_modes: tuple[int, ...] = ()
    field_periods: int = 1
    radial_coordinate: str = "rho"

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("iota", "B0", "B_cos", "B_sin")
    _static_fields: ClassVar[tuple[str, ...]] = (
        "m_modes",
        "n_modes",
        "field_periods",
        "radial_coordinate",
    )

    def __post_init__(self):
        if self.field_periods < 1:
            raise ValueError("field_periods must be at least 1")
        if self.radial_coordinate not in ("rho", "psi", "x"):
            raise ValueError("radial_coordinate must be 'rho', 'psi', or 'x'")
        if len(self.m_modes) != len(self.n_modes):
            raise ValueError("m_modes and n_modes must have the same length")
        B_cos = jnp.asarray(self.B_cos if len(self.B_cos) else [], dtype=jnp.float64)
        B_sin = jnp.asarray(self.B_sin if len(self.B_sin) else [], dtype=jnp.float64)
        if B_cos.shape != B_sin.shape:
            raise ValueError("B_cos and B_sin must have the same shape")
        if B_cos.shape[0] != len(self.m_modes):
            raise ValueError("B mode coefficient length must match m_modes/n_modes")
        object.__setattr__(self, "B_cos", B_cos)
        object.__setattr__(self, "B_sin", B_sin)
        object.__setattr__(self, "B0", jnp.asarray(self.B0, dtype=jnp.float64))
        object.__setattr__(self, "iota", jnp.asarray(self.iota, dtype=jnp.float64))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FieldLineSpec(_PyTreeDataclass):
    """Continuous field-line selection on a Boozer surface."""

    rho: float
    alpha0: float = 0.0
    radial_coordinate: str = "rho"

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("rho", "alpha0")
    _static_fields: ClassVar[tuple[str, ...]] = ("radial_coordinate",)

    def __post_init__(self):
        if self.radial_coordinate not in ("rho", "psi", "x"):
            raise ValueError("radial_coordinate must be 'rho', 'psi', or 'x'")
        if self.radial_coordinate in ("rho", "x") and not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho/x radial coordinates must lie in [0, 1]")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BoozerFieldLine(_PyTreeDataclass):
    """Sampled Boozer field line."""

    z: object
    theta: object
    phi: object
    alpha: object
    rho: object
    w_z: object
    radial_coordinate: str = "rho"

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("z", "theta", "phi", "alpha", "rho", "w_z")
    _static_fields: ClassVar[tuple[str, ...]] = ("radial_coordinate",)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PhysicalFluxTubeGeometry(_PyTreeDataclass):
    """Physical stellarator quantities sampled along one field line."""

    z: object
    w_z: object
    theta: object
    phi: object
    rho: object
    B: object
    b_dot_grad_z: object
    grad_psi_sq: object
    grad_alpha_sq: object
    grad_psi_dot_grad_alpha: object
    B_cross_gradB_dot_grad_psi: object
    B_cross_gradB_dot_grad_alpha: object
    b_cross_kappa_dot_grad_psi: object
    b_cross_kappa_dot_grad_alpha: object
    radial_coordinate: str = "rho"
    source: str = "precomputed"

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "z",
        "w_z",
        "theta",
        "phi",
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
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("radial_coordinate", "source")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FluxTubeGeometry(_PyTreeDataclass):
    """Internal solver geometry produced by a Boozer/stellarator adapter."""

    z: object
    w_z: object
    theta: object
    phi: object
    rho: object
    B: object
    F: object
    G: object
    E_y: object
    D_x: object
    D_y: object
    g_xx: object
    g_xy: object
    g_yy: object
    radial_coordinate: str = "rho"
    source: str = "precomputed"

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "z",
        "w_z",
        "theta",
        "phi",
        "rho",
        "B",
        "F",
        "G",
        "E_y",
        "D_x",
        "D_y",
        "g_xx",
        "g_xy",
        "g_yy",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("radial_coordinate", "source")


def build_boozer_parallel_grid(
    n_z: int,
    n_turns: int = 1,
    *,
    center: float = 0.0,
    dtype: str = "float64",
) -> ParallelGrid:
    """Build a periodic toroidal-angle grid spanning ``n_turns`` full turns."""

    if n_turns < 1:
        raise ValueError("n_turns must be at least 1")
    span = 2.0 * jnp.pi * n_turns
    half_span = float(span / 2.0)
    return build_parallel_grid(
        ParallelGridSpec(
            n_z=n_z,
            z_min=center - half_span,
            z_max=center + half_span,
            topology="periodic",
            dtype=dtype,
        )
    )


def sample_boozer_field_line(
    surface: BoozerSurface,
    spec: FieldLineSpec,
    parallel_grid: ParallelGrid,
) -> BoozerFieldLine:
    """Trace a field line using ``alpha = theta - iota * phi - alpha0``."""

    phi = parallel_grid.z
    theta = spec.alpha0 + surface.iota * phi
    alpha = theta - surface.iota * phi - spec.alpha0
    return BoozerFieldLine(
        z=parallel_grid.z,
        theta=theta,
        phi=phi,
        alpha=alpha,
        rho=jnp.full_like(phi, spec.rho),
        w_z=parallel_grid.w_z,
        radial_coordinate=spec.radial_coordinate,
    )


def evaluate_boozer_magnetic_field(surface: BoozerSurface, field_line: BoozerFieldLine):
    """Evaluate a simple Boozer Fourier series for ``B(theta, phi)``."""

    B = jnp.full_like(field_line.theta, surface.B0)
    if len(surface.m_modes) == 0:
        return B
    m = jnp.asarray(surface.m_modes, dtype=field_line.theta.dtype)
    n = jnp.asarray(surface.n_modes, dtype=field_line.theta.dtype)
    phase = (
        m[:, None] * field_line.theta[None, :]
        - n[:, None] * surface.field_periods * field_line.phi[None, :]
    )
    return B + jnp.sum(
        surface.B_cos[:, None] * jnp.cos(phase)
        + surface.B_sin[:, None] * jnp.sin(phase),
        axis=0,
    )


def build_desc_geometry_from_arrays(
    parallel_grid: ParallelGrid,
    *,
    theta,
    phi,
    rho,
    B,
    b_dot_grad_z,
    grad_psi_sq,
    grad_alpha_sq,
    grad_psi_dot_grad_alpha,
    B_cross_gradB_dot_grad_psi,
    B_cross_gradB_dot_grad_alpha,
    b_cross_kappa_dot_grad_psi,
    b_cross_kappa_dot_grad_alpha,
    alpha=None,
    radial_coordinate: RadialCoordinate = "rho",
) -> FluxTubeGeometry:
    """Build solver geometry from DESC-sampled flux-tube arrays.

    DESC remains the equilibrium/geometry evaluator.  This adapter only assumes
    that DESC or a caller has sampled the physical quantities on
    ``parallel_grid.z`` and passes those arrays into the gyrokinetic geometry
    contract.  Scalars are broadcast to the parallel grid for reduced tests.
    """

    if radial_coordinate not in ("rho", "psi", "x"):
        raise ValueError("radial_coordinate must be 'rho', 'psi', or 'x'")
    shape = parallel_grid.z.shape
    theta_array = _coerce_geometry_array("theta", theta, shape)
    phi_array = _coerce_geometry_array("phi", phi, shape)
    rho_array = _coerce_geometry_array("rho", rho, shape)
    alpha_array = (
        jnp.zeros_like(parallel_grid.z)
        if alpha is None
        else _coerce_geometry_array("alpha", alpha, shape)
    )
    field_line = BoozerFieldLine(
        z=parallel_grid.z,
        theta=theta_array,
        phi=phi_array,
        alpha=alpha_array,
        rho=rho_array,
        w_z=parallel_grid.w_z,
        radial_coordinate=radial_coordinate,
    )
    physical = build_physical_flux_tube_geometry_from_arrays(
        field_line=field_line,
        B=_coerce_geometry_array("B", B, shape),
        b_dot_grad_z=_coerce_geometry_array("b_dot_grad_z", b_dot_grad_z, shape),
        grad_psi_sq=_coerce_geometry_array("grad_psi_sq", grad_psi_sq, shape),
        grad_alpha_sq=_coerce_geometry_array("grad_alpha_sq", grad_alpha_sq, shape),
        grad_psi_dot_grad_alpha=_coerce_geometry_array(
            "grad_psi_dot_grad_alpha",
            grad_psi_dot_grad_alpha,
            shape,
        ),
        B_cross_gradB_dot_grad_psi=_coerce_geometry_array(
            "B_cross_gradB_dot_grad_psi",
            B_cross_gradB_dot_grad_psi,
            shape,
        ),
        B_cross_gradB_dot_grad_alpha=_coerce_geometry_array(
            "B_cross_gradB_dot_grad_alpha",
            B_cross_gradB_dot_grad_alpha,
            shape,
        ),
        b_cross_kappa_dot_grad_psi=_coerce_geometry_array(
            "b_cross_kappa_dot_grad_psi",
            b_cross_kappa_dot_grad_psi,
            shape,
        ),
        b_cross_kappa_dot_grad_alpha=_coerce_geometry_array(
            "b_cross_kappa_dot_grad_alpha",
            b_cross_kappa_dot_grad_alpha,
            shape,
        ),
        source="desc",
    )
    return map_physical_to_internal_geometry(physical, parallel_grid)


def build_physical_flux_tube_geometry_from_arrays(
    *,
    field_line: BoozerFieldLine,
    B,
    b_dot_grad_z,
    grad_psi_sq,
    grad_alpha_sq,
    grad_psi_dot_grad_alpha,
    B_cross_gradB_dot_grad_psi,
    B_cross_gradB_dot_grad_alpha,
    b_cross_kappa_dot_grad_psi,
    b_cross_kappa_dot_grad_alpha,
    source: str = "precomputed",
) -> PhysicalFluxTubeGeometry:
    """Create a physical flux-tube geometry object from precomputed arrays."""

    return PhysicalFluxTubeGeometry(
        z=field_line.z,
        w_z=field_line.w_z,
        theta=field_line.theta,
        phi=field_line.phi,
        rho=field_line.rho,
        B=jnp.asarray(B),
        b_dot_grad_z=jnp.asarray(b_dot_grad_z),
        grad_psi_sq=jnp.asarray(grad_psi_sq),
        grad_alpha_sq=jnp.asarray(grad_alpha_sq),
        grad_psi_dot_grad_alpha=jnp.asarray(grad_psi_dot_grad_alpha),
        B_cross_gradB_dot_grad_psi=jnp.asarray(B_cross_gradB_dot_grad_psi),
        B_cross_gradB_dot_grad_alpha=jnp.asarray(B_cross_gradB_dot_grad_alpha),
        b_cross_kappa_dot_grad_psi=jnp.asarray(b_cross_kappa_dot_grad_psi),
        b_cross_kappa_dot_grad_alpha=jnp.asarray(b_cross_kappa_dot_grad_alpha),
        radial_coordinate=field_line.radial_coordinate,
        source=source,
    )


def map_physical_to_internal_geometry(
    physical: PhysicalFluxTubeGeometry,
    parallel_grid: ParallelGrid,
) -> FluxTubeGeometry:
    """Map physical Boozer/GX/GS2-like quantities to the solver contract."""

    dB_dz = parallel_grid.D_z @ physical.B
    F = physical.b_dot_grad_z
    G = -F * dB_dz / physical.B
    drift_scale = 1.0 / physical.B**2
    D_x = (
        physical.B_cross_gradB_dot_grad_psi
        + physical.B * physical.b_cross_kappa_dot_grad_psi
    ) * drift_scale
    D_y = (
        physical.B_cross_gradB_dot_grad_alpha
        + physical.B * physical.b_cross_kappa_dot_grad_alpha
    ) * drift_scale
    E_y = physical.B_cross_gradB_dot_grad_alpha * drift_scale
    return FluxTubeGeometry(
        z=physical.z,
        w_z=physical.w_z,
        theta=physical.theta,
        phi=physical.phi,
        rho=physical.rho,
        B=physical.B,
        F=F,
        G=G,
        E_y=E_y,
        D_x=D_x,
        D_y=D_y,
        g_xx=physical.grad_psi_sq,
        g_xy=physical.grad_psi_dot_grad_alpha,
        g_yy=physical.grad_alpha_sq,
        radial_coordinate=physical.radial_coordinate,
        source=physical.source,
    )


def _coerce_geometry_array(name: str, value, shape: tuple[int, ...]):
    array = jnp.asarray(value)
    if array.shape == ():
        return jnp.broadcast_to(array, shape)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    return array
