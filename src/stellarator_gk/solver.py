"""Matrix-free linear gyrokinetic residual assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from .physics.collisions import build_conserving_bgk_precompute, conserving_bgk_collision
from .physics.quasineutrality import (
    AdiabaticElectronParams,
    build_adiabatic_quasineutrality_precompute,
    build_kinetic_quasineutrality_precompute,
    solve_adiabatic_electron_phi,
    solve_kinetic_electron_phi,
)
from .physics.rhs_terms import (
    LinearRHSPrecompute,
    build_linear_rhs_precompute,
    linear_residual_from_phi,
)
from .types import FourierGrid, ParallelGrid, SpeciesParams, VelocityGrid, _PyTreeDataclass


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearResidualPrecompute(_PyTreeDataclass):
    """Coupled RHS and quasineutrality precompute for ``linear_residual``."""

    rhs: LinearRHSPrecompute
    field: object
    field_model: str = "adiabatic"
    n_species: int = 1
    collisions: object = None

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("rhs", "field", "collisions")
    _static_fields: ClassVar[tuple[str, ...]] = ("field_model", "n_species")

    def __post_init__(self):
        if self.field_model not in ("adiabatic", "kinetic"):
            raise ValueError("field_model must be 'adiabatic' or 'kinetic'")
        if self.n_species < 1:
            raise ValueError("n_species must be at least 1")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ImplicitParallelResponsePrecompute(_PyTreeDataclass):
    """Schur-complement data for a field-coupled midpoint parallel step."""

    left_inverse: object
    field_matrix: object
    field_inverse: object
    mass_matrix: object
    derivative: object
    streaming_coefficient: object
    field_maxwellian: object
    left_dt: object
    right_dt: object

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "left_inverse",
        "field_matrix",
        "field_inverse",
        "mass_matrix",
        "derivative",
        "streaming_coefficient",
        "field_maxwellian",
        "left_dt",
        "right_dt",
    )


def build_implicit_parallel_response_precompute(
    precompute: LinearResidualPrecompute,
    dt,
    *,
    spatial_scheme: str = "spectral",
    zed_upwind: float = 0.02,
    time_upwind: float = 0.02,
    periodic_parallel_boundary: bool = False,
) -> ImplicitParallelResponsePrecompute:
    """Build a midpoint Schur complement for streaming, field drive, and QN.

    ``dt`` is the duration of one application.  The distribution-space block
    is inverted independently for every parallel velocity, while the dense
    Schur complement has only the electrostatic-field dimension. The stella
    scheme defaults to its zero-incoming extended-domain boundary; periodic
    wrapping must be requested explicitly for zonal or otherwise periodic
    chains.
    """

    if precompute.n_species != 1:
        raise NotImplementedError("implicit parallel response currently supports one species")
    if precompute.rhs.parallel_derivative_model != "matrix":
        raise ValueError("implicit parallel response requires the matrix parallel derivative")

    derivative_base = jnp.asarray(precompute.rhs.D_z)
    coefficient = jnp.asarray(precompute.rhs.parallel_streaming_coeff)
    if coefficient.ndim == 3 and coefficient.shape[0] == 1:
        coefficient = coefficient[0]
    if derivative_base.ndim != 2 or derivative_base.shape[0] != derivative_base.shape[1]:
        raise ValueError("D_z must be square")
    if coefficient.ndim != 2 or coefficient.shape[1] != derivative_base.shape[0]:
        raise ValueError("parallel_streaming_coeff must have shape (vpar,z)")
    if spatial_scheme not in ("spectral", "stella_near_centered"):
        raise ValueError("spatial_scheme must be 'spectral' or 'stella_near_centered'")
    if not 0.0 <= zed_upwind <= 1.0 or not 0.0 <= time_upwind <= 1.0:
        raise ValueError("upwind parameters must lie in [0, 1]")

    dtype = jnp.result_type(derivative_base, coefficient, jnp.asarray(dt))
    identity_z = jnp.eye(derivative_base.shape[0], dtype=dtype)
    if spatial_scheme == "spectral":
        derivative = jnp.broadcast_to(
            derivative_base[None, :, :],
            (coefficient.shape[0],) + derivative_base.shape,
        )
        mass_matrix = jnp.broadcast_to(identity_z[None, :, :], derivative.shape)
        left_dt = right_dt = 0.5 * jnp.asarray(dt, dtype=dtype)
    else:
        spacing = jnp.sum(jnp.asarray(precompute.field.w_z)) / derivative_base.shape[0]
        forward = jnp.diag(jnp.ones(derivative_base.shape[0] - 1, dtype=dtype), 1)
        backward = jnp.diag(jnp.ones(derivative_base.shape[0] - 1, dtype=dtype), -1)
        if periodic_parallel_boundary:
            forward = forward.at[-1, 0].set(1.0)
            backward = backward.at[0, -1].set(1.0)
        plus = 0.5 * (1.0 + zed_upwind)
        minus = 0.5 * (1.0 - zed_upwind)
        positive_mass = plus * identity_z + minus * forward
        negative_mass = plus * identity_z + minus * backward
        positive_derivative = (forward - identity_z) / spacing
        negative_derivative = (identity_z - backward) / spacing
        positive_rhs = -jnp.mean(coefficient, axis=1) >= 0.0
        mass_matrix = jnp.where(
            positive_rhs[:, None, None], positive_mass, negative_mass
        )
        derivative = jnp.where(
            positive_rhs[:, None, None], positive_derivative, negative_derivative
        )
        left_dt = 0.5 * (1.0 + time_upwind) * jnp.asarray(dt, dtype=dtype)
        right_dt = 0.5 * (1.0 - time_upwind) * jnp.asarray(dt, dtype=dtype)

    streaming_coefficient = jnp.einsum("vij,vj->vi", mass_matrix, coefficient)
    field_maxwellian = jnp.einsum(
        "vij,vmj->vmi", mass_matrix, precompute.rhs.maxwellian[0]
    )
    operator = -streaming_coefficient[:, :, None] * derivative
    left_inverse = jnp.linalg.solve(
        mass_matrix - left_dt * operator,
        jnp.broadcast_to(identity_z, operator.shape),
    )

    bessel = jnp.asarray(precompute.rhs.flr_factors.bessel_j0)
    n_z, n_kx, n_ky = map(int, bessel.shape[-3:])
    n_field = n_z * n_kx * n_ky
    field_basis = jnp.eye(n_field, dtype=jnp.result_type(dtype, jnp.complex64)).reshape(
        n_field, n_z, n_kx, n_ky
    )

    def field_response_column(phi):
        drive = left_dt * _implicit_parallel_field_drive(
            phi,
            precompute,
            derivative,
            streaming_coefficient,
            field_maxwellian,
        )
        correction = _apply_parallel_left_inverse(drive, left_inverse)
        return _solve_phi(correction, precompute)

    response_columns = jax.vmap(field_response_column)(field_basis).reshape(
        n_field, n_field
    )
    field_matrix = jnp.eye(n_field, dtype=response_columns.dtype) - response_columns.T
    return ImplicitParallelResponsePrecompute(
        left_inverse=left_inverse,
        field_matrix=field_matrix,
        field_inverse=jnp.linalg.inv(field_matrix),
        mass_matrix=mass_matrix,
        derivative=derivative,
        streaming_coefficient=streaming_coefficient,
        field_maxwellian=field_maxwellian,
        left_dt=left_dt,
        right_dt=right_dt,
    )


def implicit_parallel_response_step(
    distribution,
    precompute: LinearResidualPrecompute,
    response: ImplicitParallelResponsePrecompute,
):
    """Apply a coupled implicit-midpoint parallel/QN response step."""

    distribution = jnp.asarray(distribution)
    if distribution.ndim != 5:
        raise ValueError("distribution must have shape (vpar,mu,z,kx,ky)")
    phi_old = _solve_phi(distribution, precompute)
    rhs_state = _apply_parallel_matrix(distribution, response.mass_matrix) + response.right_dt * (
        _implicit_parallel_streaming(distribution, response)
        + _implicit_parallel_field_drive(
            phi_old,
            precompute,
            response.derivative,
            response.streaming_coefficient,
            response.field_maxwellian,
        )
    )
    uncoupled = _apply_parallel_left_inverse(rhs_state, response.left_inverse)
    phi_shape = phi_old.shape
    phi_new = (
        response.field_inverse @ _solve_phi(uncoupled, precompute).reshape(-1)
    ).reshape(phi_shape)
    correction = _apply_parallel_left_inverse(
        response.left_dt
        * _implicit_parallel_field_drive(
            phi_new,
            precompute,
            response.derivative,
            response.streaming_coefficient,
            response.field_maxwellian,
        ),
        response.left_inverse,
    )
    return uncoupled + correction


def _apply_parallel_left_inverse(distribution, left_inverse):
    return jnp.einsum("vij,vmjxy->vmixy", left_inverse, distribution)


def _apply_parallel_matrix(distribution, matrix):
    return jnp.einsum("vij,vmjxy->vmixy", matrix, distribution)


def _implicit_parallel_streaming(distribution, response):
    differentiated = _apply_parallel_matrix(distribution, response.derivative)
    return -response.streaming_coefficient[:, None, :, None, None] * differentiated


def _implicit_parallel_field_drive(
    phi,
    precompute,
    derivative,
    streaming_coefficient,
    field_maxwellian,
):
    rhs = precompute.rhs
    gyro_phi = rhs.flr_factors.bessel_j0[0] * phi[None, :, :, :]
    differentiated = jnp.einsum("vij,mjxy->vmixy", derivative, gyro_phi)
    return (
        -rhs.charge_over_temperature[0]
        * streaming_coefficient[:, None, :, None, None]
        * field_maxwellian[..., None, None]
        * differentiated
    )


def build_linear_residual_precompute(
    velocity_grid: VelocityGrid,
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    geometry,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    *,
    field_model: str = "adiabatic",
    electron_params: AdiabaticElectronParams | None = None,
    field_precompute=None,
    flr_factors=None,
    perpendicular_damping=None,
    parallel_recurrence_rate: float = 0.0,
    parallel_recurrence_velocity_model: str = "rms",
    velocity_recurrence_rate: float = 0.0,
    velocity_recurrence_velocity_model: str = "rms",
    mode_connectivity=None,
    parallel_derivative_model: str = "matrix",
    phase_space_measure=None,
    collision_frequency=None,
) -> LinearResidualPrecompute:
    """Build the coupled linear RHS and electrostatic field precompute."""

    rhs = build_linear_rhs_precompute(
        velocity_grid,
        parallel_grid,
        fourier_grid,
        geometry,
        species,
        flr_factors=flr_factors,
        perpendicular_damping=perpendicular_damping,
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_recurrence_velocity_model=parallel_recurrence_velocity_model,
        velocity_recurrence_rate=velocity_recurrence_rate,
        velocity_recurrence_velocity_model=velocity_recurrence_velocity_model,
        mode_connectivity=mode_connectivity,
        parallel_derivative_model=parallel_derivative_model,
    )
    normalized_model = _normalize_field_model(field_model)
    field = field_precompute
    if field is None:
        if normalized_model == "adiabatic":
            field = build_adiabatic_quasineutrality_precompute(
                velocity_grid,
                geometry.B,
                rhs.flr_factors,
                species,
                electron_params,
                fourier_grid=fourier_grid,
                w_z=geometry.w_z,
                phase_space_measure=phase_space_measure,
            )
        else:
            field = build_kinetic_quasineutrality_precompute(
                velocity_grid,
                geometry.B,
                rhs.flr_factors,
                species,
                fourier_grid=fourier_grid,
                phase_space_measure=phase_space_measure,
            )
    collisions = None
    if collision_frequency is not None:
        collisions = build_conserving_bgk_precompute(
            velocity_grid,
            geometry.B,
            species,
            collision_frequency,
        )
    return LinearResidualPrecompute(
        rhs=rhs,
        field=field,
        field_model=normalized_model,
        n_species=rhs.n_species,
        collisions=collisions,
    )


def linear_residual(
    distribution,
    geometry=None,
    params=None,
    precomputed=None,
    *,
    phi=None,
):
    """Return the matrix-free linear electrostatic RHS.

    ``precomputed`` may be either a ``LinearResidualPrecompute`` for a
    self-consistent phi solve or a ``LinearRHSPrecompute`` when ``phi`` is
    supplied explicitly.  The ``geometry`` and ``params`` positional slots are
    accepted to match the project-level residual interface; Phase 7 keeps all
    expensive geometry/species work inside the precompute object.
    """

    precompute = _coerce_precompute(geometry, params, precomputed)
    if isinstance(precompute, LinearRHSPrecompute):
        if phi is None:
            raise ValueError("phi must be supplied when using LinearRHSPrecompute directly")
        return linear_residual_from_phi(distribution, phi, precompute)

    if not isinstance(precompute, LinearResidualPrecompute):
        raise TypeError("precomputed must be LinearResidualPrecompute or LinearRHSPrecompute")
    solved_phi = phi if phi is not None else _solve_phi(distribution, precompute)
    residual = linear_residual_from_phi(distribution, solved_phi, precompute.rhs)
    if precompute.collisions is not None:
        residual = residual + conserving_bgk_collision(distribution, precompute.collisions)
    return residual


@jax.jit
def jitted_linear_residual(distribution, precomputed: LinearResidualPrecompute):
    """JIT-compiled self-consistent linear residual for fixed grid topology."""

    return linear_residual(distribution, precomputed=precomputed)


def _solve_phi(distribution, precompute: LinearResidualPrecompute):
    if precompute.field_model == "adiabatic":
        return solve_adiabatic_electron_phi(distribution, precompute.field)
    if precompute.field_model == "kinetic":
        return solve_kinetic_electron_phi(distribution, precompute.field)
    raise ValueError(f"unsupported field_model {precompute.field_model!r}")


def _coerce_precompute(geometry, params, precomputed):
    if precomputed is not None:
        return precomputed
    if isinstance(geometry, (LinearResidualPrecompute, LinearRHSPrecompute)):
        return geometry
    if isinstance(params, (LinearResidualPrecompute, LinearRHSPrecompute)):
        return params
    raise ValueError("a Phase 7 precompute object is required")


def _normalize_field_model(field_model: str) -> str:
    if field_model not in ("adiabatic", "kinetic"):
        raise ValueError("field_model must be 'adiabatic' or 'kinetic'")
    return field_model
