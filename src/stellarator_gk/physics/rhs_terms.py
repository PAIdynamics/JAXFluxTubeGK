"""Linear electrostatic gyrokinetic RHS terms.

The functions here implement the term-level contract from ``main.tex`` using
matrix-free JAX array operations.  All topology and grid choices are assumed to
have been handled before this layer; the RHS sees only collocation derivative
matrices, Fourier mode arrays, and precomputed geometry/species coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from ..types import FourierGrid, ParallelGrid, SpeciesParams, VelocityGrid, _PyTreeDataclass
from .primitives import (
    FLRFactors,
    magnetic_drift_frequency as build_magnetic_drift_frequency,
    maxwellian as build_maxwellian,
    mirror_force_coefficient,
    normalized_energy,
    parallel_streaming_coefficient,
    species_flr_factors,
    thermodynamic_drive_factor,
)


def _center_9(stencil5):
    out = [0.0] * 9
    out[2:7] = stencil5
    return out


_GKW_D1_POS = np.asarray(
    [
        _center_9([0.0, 0.0, -18.0, 24.0, -6.0]),
        [0.0, 0.0, 0.0, -4.0, -6.0, 12.0, -2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, 0.0, 0.0, 0.0],
        _center_9([0.0, -6.0, 0.0, 0.0, 0.0]),
    ],
    dtype=float,
)
_GKW_D1_NEG = np.asarray(
    [
        _center_9([0.0, 0.0, 0.0, 6.0, 0.0]),
        [0.0, 0.0, 0.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, -12.0, 6.0, 4.0, 0.0, 0.0, 0.0],
        _center_9([6.0, -24.0, 18.0, 0.0, 0.0]),
    ],
    dtype=float,
)
_GKW_D4_POS = np.asarray(
    [
        [0.0] * 9,
        [0.0] * 9,
        _center_9([-1.0, 4.0, -6.0, 4.0, -1.0]),
        [0.0, 0.0, -1.0, 4.0, -6.0, 4.0, 0.0, 0.0, 0.0],
        _center_9([0.0, 12.0, -24.0, 0.0, 0.0]),
    ],
    dtype=float,
)
_GKW_D4_NEG = np.asarray(
    [
        _center_9([0.0, 0.0, -24.0, 12.0, 0.0]),
        [0.0, 0.0, 0.0, 4.0, -6.0, 4.0, -1.0, 0.0, 0.0],
        _center_9([-1.0, 4.0, -6.0, 4.0, -1.0]),
        [0.0] * 9,
        [0.0] * 9,
    ],
    dtype=float,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GKWParallelStencil(_PyTreeDataclass):
    """GKW/Gyaradax sign-dependent parallel stencil maps."""

    d1_pos: object
    d1_neg: object
    d4_pos: object
    d4_neg: object
    s_shift: object
    kx_shift: object
    valid_shift: object
    spacing: float

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "d1_pos",
        "d1_neg",
        "d4_pos",
        "d4_neg",
        "s_shift",
        "kx_shift",
        "valid_shift",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("spacing",)

    def __post_init__(self):
        if self.spacing <= 0.0:
            raise ValueError("spacing must be positive")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GKWArakawaIghStencil(_PyTreeDataclass):
    """Precomputed sparse shift table for GKW's fused ``igh`` operator."""

    coefficients: object
    valid_v_shift: object
    spacing_z: float
    spacing_vpar: float

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("coefficients", "valid_v_shift")
    _static_fields: ClassVar[tuple[str, ...]] = ("spacing_z", "spacing_vpar")

    def __post_init__(self):
        if self.spacing_z <= 0.0:
            raise ValueError("spacing_z must be positive")
        if self.spacing_vpar <= 0.0:
            raise ValueError("spacing_vpar must be positive")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearRHSPrecompute(_PyTreeDataclass):
    """Precomputed coefficients for the linear electrostatic RHS."""

    D_z: object
    D_vpar: object
    ky: object
    E_y: object
    flr_factors: FLRFactors
    maxwellian: object
    drive_factor: object
    parallel_streaming_coeff: object
    mirror_force_coeff: object
    magnetic_drift_frequency: object
    charge_over_temperature: object
    perpendicular_damping: object
    parallel_recurrence_operator: object
    parallel_recurrence_coeff: object
    velocity_recurrence_operator: object
    velocity_recurrence_coeff: object
    gkw_parallel_stencil: GKWParallelStencil
    gkw_igh_stencil: GKWArakawaIghStencil
    parallel_derivative_model: str
    n_species: int

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "D_z",
        "D_vpar",
        "ky",
        "E_y",
        "flr_factors",
        "maxwellian",
        "drive_factor",
        "parallel_streaming_coeff",
        "mirror_force_coeff",
        "magnetic_drift_frequency",
        "charge_over_temperature",
        "perpendicular_damping",
        "parallel_recurrence_operator",
        "parallel_recurrence_coeff",
        "velocity_recurrence_operator",
        "velocity_recurrence_coeff",
        "gkw_parallel_stencil",
        "gkw_igh_stencil",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("parallel_derivative_model", "n_species")

    def __post_init__(self):
        if self.n_species < 1:
            raise ValueError("n_species must be at least 1")
        if self.parallel_derivative_model not in ("matrix", "gkw_upwind", "gkw_igh"):
            raise ValueError(
                "parallel_derivative_model must be 'matrix', 'gkw_upwind', or 'gkw_igh'"
            )


def build_linear_rhs_precompute(
    velocity_grid: VelocityGrid,
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    geometry,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    *,
    flr_factors: FLRFactors | None = None,
    perpendicular_damping=None,
    parallel_recurrence_rate: float = 0.0,
    parallel_recurrence_velocity_model: str = "rms",
    velocity_recurrence_rate: float = 0.0,
    velocity_recurrence_velocity_model: str = "rms",
    mode_connectivity=None,
    parallel_derivative_model: str = "matrix",
) -> LinearRHSPrecompute:
    """Precompute geometry/species coefficients used by the linear RHS terms."""

    if parallel_derivative_model not in ("matrix", "gkw_upwind", "gkw_igh"):
        raise ValueError("parallel_derivative_model must be 'matrix', 'gkw_upwind', or 'gkw_igh'")
    species_tuple = _as_species_tuple(species)
    n_species = len(species_tuple)
    kperp2 = _k_perp_squared(geometry, fourier_grid)
    flr = flr_factors or species_flr_factors(species, velocity_grid.mu, geometry.B, kperp2)
    flr = _with_species_axis_flr(flr, n_species)

    energy = normalized_energy(velocity_grid.vpar, velocity_grid.mu, geometry.B, species)
    fmax = _with_species_axis(
        build_maxwellian(velocity_grid.vpar, velocity_grid.mu, geometry.B, species),
        n_species,
        "maxwellian",
        single_ndim=3,
    )
    drive = _with_species_axis(
        thermodynamic_drive_factor(energy, species),
        n_species,
        "drive_factor",
        single_ndim=3,
    )
    parallel = _with_species_axis(
        parallel_streaming_coefficient(velocity_grid.vpar, geometry.F, species),
        n_species,
        "parallel_streaming_coeff",
        single_ndim=2,
    )
    mirror = _with_species_axis(
        mirror_force_coefficient(velocity_grid.mu, geometry.B, geometry.G, species),
        n_species,
        "mirror_force_coeff",
        single_ndim=2,
    )
    drift = _with_species_axis(
        build_magnetic_drift_frequency(
            velocity_grid.vpar,
            velocity_grid.mu,
            geometry.B,
            geometry.D_x,
            geometry.D_y,
            fourier_grid.kx,
            fourier_grid.ky,
            species,
            D_x_gradB=getattr(geometry, "D_x_gradB", None),
            D_y_gradB=getattr(geometry, "D_y_gradB", None),
            D_x_curvature=getattr(geometry, "D_x_curvature", None),
            D_y_curvature=getattr(geometry, "D_y_curvature", None),
        ),
        n_species,
        "magnetic_drift_frequency",
        single_ndim=5,
    )
    charge_over_temperature = jnp.asarray(
        [item.charge / item.temperature for item in species_tuple],
        dtype=jnp.asarray(geometry.B).dtype,
    )

    damping = _normalize_perpendicular_damping(
        perpendicular_damping,
        fourier_grid.kx.shape[0],
        fourier_grid.ky.shape[0],
        dtype=jnp.asarray(geometry.B).dtype,
    )
    recurrence_operator = _gkw_parallel_recurrence_operator(
        parallel_grid,
        dtype=jnp.asarray(geometry.B).dtype,
    )
    recurrence_coeff = _parallel_recurrence_coefficient(
        velocity_grid.vpar,
        parallel,
        parallel_recurrence_rate,
        parallel_recurrence_velocity_model,
    )
    velocity_recurrence_operator = _gkw_velocity_recurrence_operator(
        velocity_grid,
        dtype=jnp.asarray(geometry.B).dtype,
    )
    velocity_recurrence_coeff = _velocity_recurrence_coefficient(
        velocity_grid.mu,
        mirror,
        velocity_recurrence_rate,
        velocity_recurrence_velocity_model,
    )
    if parallel_derivative_model in ("gkw_upwind", "gkw_igh"):
        if mode_connectivity is None:
            raise ValueError("mode_connectivity is required for GKW parallel derivative models")
        gkw_parallel_stencil = build_gkw_parallel_stencil(
            parallel_grid,
            fourier_grid,
            mode_connectivity,
            dtype=jnp.asarray(geometry.B).dtype,
        )
    else:
        gkw_parallel_stencil = _empty_gkw_parallel_stencil(
            parallel_grid,
            fourier_grid,
            dtype=jnp.asarray(geometry.B).dtype,
        )
    if parallel_derivative_model == "gkw_igh":
        gkw_igh_stencil = build_gkw_igh_stencil(
            velocity_grid,
            parallel_grid,
            fourier_grid,
            geometry,
            mode_connectivity,
            parallel,
            recurrence_coeff,
            velocity_recurrence_coeff,
            dtype=jnp.asarray(geometry.B).dtype,
        )
    else:
        gkw_igh_stencil = _empty_gkw_igh_stencil(
            velocity_grid,
            parallel_grid,
            fourier_grid,
            n_species,
            dtype=jnp.asarray(geometry.B).dtype,
        )

    return LinearRHSPrecompute(
        D_z=parallel_grid.D_z,
        D_vpar=velocity_grid.D_vpar,
        ky=fourier_grid.ky,
        E_y=geometry.E_y,
        flr_factors=flr,
        maxwellian=fmax,
        drive_factor=drive,
        parallel_streaming_coeff=parallel,
        mirror_force_coeff=mirror,
        magnetic_drift_frequency=drift,
        charge_over_temperature=charge_over_temperature,
        perpendicular_damping=damping,
        parallel_recurrence_operator=recurrence_operator,
        parallel_recurrence_coeff=recurrence_coeff,
        velocity_recurrence_operator=velocity_recurrence_operator,
        velocity_recurrence_coeff=velocity_recurrence_coeff,
        gkw_parallel_stencil=gkw_parallel_stencil,
        gkw_igh_stencil=gkw_igh_stencil,
        parallel_derivative_model=parallel_derivative_model,
        n_species=n_species,
    )


def build_gkw_parallel_stencil(
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    mode_connectivity,
    *,
    dtype=None,
) -> GKWParallelStencil:
    """Build the GKW/Gyaradax fourth-order upwind parallel stencil maps."""

    n_z = int(jnp.asarray(parallel_grid.z).shape[0])
    n_kx = int(jnp.asarray(fourier_grid.kx).shape[0])
    n_ky = int(jnp.asarray(fourier_grid.ky).shape[0])
    spacing = float(jnp.sum(jnp.asarray(parallel_grid.w_z)) / n_z)
    dtype = jnp.asarray(parallel_grid.z).dtype if dtype is None else dtype
    ixplus = np.asarray(mode_connectivity.ixplus, dtype=np.int32)
    ixminus = np.asarray(mode_connectivity.ixminus, dtype=np.int32)
    iyzero = int(mode_connectivity.iyzero)
    if ixplus.shape != (n_kx, n_ky) or ixminus.shape != (n_kx, n_ky):
        raise ValueError("mode_connectivity shape must match the Fourier grid")

    pos_class = _gkw_parallel_position_classes(ixplus, ixminus, iyzero, n_z)
    s_shift, kx_shift, valid_shift = _gkw_parallel_shift_maps(
        ixplus,
        ixminus,
        iyzero,
        n_z,
        max_shift=4,
    )

    def coefficients(table):
        idx = np.clip(pos_class + 2, 0, 4)
        selected = table[idx] / (12.0 * spacing)
        return jnp.asarray(np.moveaxis(selected, -1, 0), dtype=dtype)

    return GKWParallelStencil(
        d1_pos=coefficients(_GKW_D1_POS),
        d1_neg=coefficients(_GKW_D1_NEG),
        d4_pos=coefficients(_GKW_D4_POS),
        d4_neg=coefficients(_GKW_D4_NEG),
        s_shift=jnp.asarray(s_shift, dtype=jnp.int32),
        kx_shift=jnp.asarray(kx_shift, dtype=jnp.int32),
        valid_shift=jnp.asarray(valid_shift, dtype=bool),
        spacing=spacing,
    )


def _empty_gkw_parallel_stencil(
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    *,
    dtype,
) -> GKWParallelStencil:
    n_z = int(jnp.asarray(parallel_grid.z).shape[0])
    n_kx = int(jnp.asarray(fourier_grid.kx).shape[0])
    n_ky = int(jnp.asarray(fourier_grid.ky).shape[0])
    shape = (9, n_z, n_kx, n_ky)
    return GKWParallelStencil(
        d1_pos=jnp.zeros(shape, dtype=dtype),
        d1_neg=jnp.zeros(shape, dtype=dtype),
        d4_pos=jnp.zeros(shape, dtype=dtype),
        d4_neg=jnp.zeros(shape, dtype=dtype),
        s_shift=jnp.zeros(shape, dtype=jnp.int32),
        kx_shift=jnp.zeros(shape, dtype=jnp.int32),
        valid_shift=jnp.zeros(shape, dtype=bool),
        spacing=1.0,
    )


def build_gkw_igh_stencil(
    velocity_grid: VelocityGrid,
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    geometry,
    mode_connectivity,
    parallel_coeff,
    parallel_recurrence_coeff,
    velocity_recurrence_coeff,
    *,
    dtype=None,
) -> GKWArakawaIghStencil:
    """Build GKW's fused Arakawa ``igh`` shift table.

    This is a production-parity backend for the GKW/Gyaradax finite-difference
    path.  It precomputes the row-local Hamiltonian bracket coefficients and
    the in-operator ``disp_par``/``disp_vp`` recurrence terms as source-shift
    weights over ``delta_vpar, delta_z in [-2, 2]``.
    """

    if velocity_grid.backend != "finite_difference":
        raise ValueError("gkw_igh requires a finite_difference velocity grid")
    if parallel_grid.backend != "finite_difference":
        raise ValueError("gkw_igh requires a finite_difference parallel grid")

    vpar = np.asarray(velocity_grid.vpar, dtype=float)
    mu = np.asarray(velocity_grid.mu, dtype=float)
    z = np.asarray(parallel_grid.z, dtype=float)
    b_field = np.asarray(geometry.B, dtype=float)
    parallel_np = np.asarray(parallel_coeff, dtype=float)
    parallel_rec_np = np.asarray(parallel_recurrence_coeff, dtype=float)
    velocity_rec_np = np.asarray(velocity_recurrence_coeff, dtype=float)
    if parallel_np.ndim != 3:
        raise ValueError("parallel_coeff must have shape (n_species,n_vpar,n_z)")
    if parallel_rec_np.shape != parallel_np.shape:
        raise ValueError("parallel_recurrence_coeff must match parallel_coeff")
    if velocity_rec_np.ndim != 3:
        raise ValueError("velocity_recurrence_coeff must have shape (n_species,n_mu,n_z)")

    n_species, n_vpar, n_z = parallel_np.shape
    n_mu = mu.shape[0]
    n_kx = int(jnp.asarray(fourier_grid.kx).shape[0])
    n_ky = int(jnp.asarray(fourier_grid.ky).shape[0])
    if vpar.shape != (n_vpar,) or z.shape != (n_z,) or b_field.shape != (n_z,):
        raise ValueError("gkw_igh grid and geometry shapes are inconsistent")
    if velocity_rec_np.shape != (n_species, n_mu, n_z):
        raise ValueError("velocity_recurrence_coeff must have shape (n_species,n_mu,n_z)")
    if n_vpar < 5 or n_z < 5:
        raise ValueError("gkw_igh requires at least five vpar and z points")

    spacing_z = float(jnp.sum(jnp.asarray(parallel_grid.w_z)) / n_z)
    spacing_vpar = float(jnp.sum(jnp.asarray(velocity_grid.w_vpar)) / n_vpar)
    if not np.allclose(np.diff(vpar), spacing_vpar, rtol=2.0e-12, atol=2.0e-12):
        raise ValueError("gkw_igh requires a uniform finite-difference vpar grid")
    if not np.allclose(np.diff(z), spacing_z, rtol=2.0e-12, atol=2.0e-12):
        raise ValueError("gkw_igh requires a uniform finite-difference parallel grid")

    ixplus = np.asarray(mode_connectivity.ixplus, dtype=np.int32)
    ixminus = np.asarray(mode_connectivity.ixminus, dtype=np.int32)
    iyzero = int(mode_connectivity.iyzero)
    if ixplus.shape != (n_kx, n_ky) or ixminus.shape != (n_kx, n_ky):
        raise ValueError("mode_connectivity shape must match the Fourier grid")
    position_class = _gkw_parallel_position_classes(ixplus, ixminus, iyzero, n_z)

    dtype = jnp.asarray(geometry.B).dtype if dtype is None else dtype
    coefficients = np.zeros(
        (5, 5, n_species, n_vpar, n_mu, n_z, n_kx, n_ky),
        dtype=float,
    )
    valid_v_shift = np.zeros((5, n_vpar), dtype=bool)
    row_v = np.arange(n_vpar)
    for offset_index, delta_v in enumerate(range(-2, 3)):
        valid_v_shift[offset_index] = (0 <= row_v + delta_v) & (row_v + delta_v < n_vpar)

    parallel_scale = _gkw_parallel_scale_from_coeff(vpar, parallel_np)

    def vpar_at(index):
        return float(vpar[0] + index * spacing_vpar)

    def hh(iz, imu, iv):
        iz_ref = min(max(iz, 0), n_z - 1)
        return 0.5 * vpar_at(iv) ** 2 + mu[imu] * b_field[iz_ref]

    def add_value(ispecies, row_iv, row_imu, row_iz, ix, iy, col_iv, col_iz, value):
        delta_v = col_iv - row_iv
        delta_z = col_iz - row_iz
        if -2 <= delta_v <= 2 and -2 <= delta_z <= 2:
            coefficients[
                delta_v + 2,
                delta_z + 2,
                ispecies,
                row_iv,
                row_imu,
                row_iz,
                ix,
                iy,
            ] += value

    for ispecies in range(n_species):
        for iv in range(n_vpar):
            for imu in range(n_mu):
                for iz in range(n_z):
                    dum = parallel_scale[ispecies, iz] / (spacing_z * spacing_vpar)
                    dum2 = -parallel_np[ispecies, iv, iz]
                    for ix in range(n_kx):
                        for iy in range(n_ky):
                            position = int(position_class[iz, ix, iy])
                            if position in (-2, 2):
                                if dum2 * position < 0.0:
                                    _add_gkw_igh_zero_two_coefficients(
                                        add_value,
                                        ispecies,
                                        iv,
                                        imu,
                                        iz,
                                        ix,
                                        iy,
                                        dum,
                                        position,
                                        hh,
                                    )
                                elif dum2 * position > 0.0:
                                    _add_gkw_igh_jhg_coefficients(
                                        add_value,
                                        ispecies,
                                        iv,
                                        imu,
                                        iz,
                                        ix,
                                        iy,
                                        dum,
                                        hh,
                                    )
                            elif position in (-1, 1):
                                if dum2 * position < 0.0:
                                    _add_gkw_igh_two_coefficients(
                                        add_value,
                                        ispecies,
                                        iv,
                                        imu,
                                        iz,
                                        ix,
                                        iy,
                                        dum,
                                        position,
                                        hh,
                                    )
                                elif dum2 * position > 0.0:
                                    _add_gkw_igh_jhg_coefficients(
                                        add_value,
                                        ispecies,
                                        iv,
                                        imu,
                                        iz,
                                        ix,
                                        iy,
                                        dum,
                                        hh,
                                    )
                            else:
                                _add_gkw_igh_jhg_coefficients(
                                    add_value,
                                    ispecies,
                                    iv,
                                    imu,
                                    iz,
                                    ix,
                                    iy,
                                    dum,
                                    hh,
                                )
                                _add_gkw_igh_diffusion_coefficients(
                                    add_value,
                                    ispecies,
                                    iv,
                                    imu,
                                    iz,
                                    ix,
                                    iy,
                                    parallel_rec_np[ispecies, iv, iz],
                                    spacing_z,
                                    "z",
                                )
                                _add_gkw_igh_diffusion_coefficients(
                                    add_value,
                                    ispecies,
                                    iv,
                                    imu,
                                    iz,
                                    ix,
                                    iy,
                                    velocity_rec_np[ispecies, imu, iz],
                                    spacing_vpar,
                                    "vpar",
                                )

    return GKWArakawaIghStencil(
        coefficients=jnp.asarray(coefficients, dtype=dtype),
        valid_v_shift=jnp.asarray(valid_v_shift, dtype=bool),
        spacing_z=spacing_z,
        spacing_vpar=spacing_vpar,
    )


def _empty_gkw_igh_stencil(
    velocity_grid: VelocityGrid,
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    n_species: int,
    *,
    dtype,
) -> GKWArakawaIghStencil:
    n_vpar = int(jnp.asarray(velocity_grid.vpar).shape[0])
    n_mu = int(jnp.asarray(velocity_grid.mu).shape[0])
    n_z = int(jnp.asarray(parallel_grid.z).shape[0])
    n_kx = int(jnp.asarray(fourier_grid.kx).shape[0])
    n_ky = int(jnp.asarray(fourier_grid.ky).shape[0])
    return GKWArakawaIghStencil(
        coefficients=jnp.zeros((5, 5, n_species, n_vpar, n_mu, n_z, n_kx, n_ky), dtype=dtype),
        valid_v_shift=jnp.zeros((5, n_vpar), dtype=bool),
        spacing_z=1.0,
        spacing_vpar=1.0,
    )


def _gkw_parallel_scale_from_coeff(vpar: np.ndarray, parallel_coeff: np.ndarray) -> np.ndarray:
    valid = np.abs(vpar) > 1.0e-300
    if not np.any(valid):
        return np.zeros((parallel_coeff.shape[0], parallel_coeff.shape[2]), dtype=float)
    return np.mean(parallel_coeff[:, valid, :] / vpar[None, valid, None], axis=1)


def _add_gkw_igh_jhg_coefficients(add, ispecies, iv, imu, iz, ix, iy, dum, hh):
    d1 = dum / 6.0
    d2 = -dum / 24.0
    entries = (
        (iv, iz - 2, d2 * (hh(iz - 1, imu, iv + 1) - hh(iz - 1, imu, iv - 1))),
        (
            iv - 1,
            iz - 1,
            d1 * (hh(iz - 1, imu, iv) - hh(iz, imu, iv - 1))
            + d2
            * (
                hh(iz - 2, imu, iv)
                + hh(iz - 1, imu, iv + 1)
                - hh(iz, imu, iv - 2)
                - hh(iz + 1, imu, iv - 1)
            ),
        ),
        (
            iv,
            iz - 1,
            d1
            * (
                hh(iz - 1, imu, iv + 1)
                - hh(iz - 1, imu, iv - 1)
                - hh(iz, imu, iv - 1)
                + hh(iz, imu, iv + 1)
            ),
        ),
        (
            iv + 1,
            iz - 1,
            d1 * (hh(iz, imu, iv + 1) - hh(iz - 1, imu, iv))
            + d2
            * (
                hh(iz + 1, imu, iv + 1)
                - hh(iz - 1, imu, iv - 1)
                - hh(iz - 2, imu, iv)
                + hh(iz, imu, iv + 2)
            ),
        ),
        (iv - 2, iz, d2 * (hh(iz - 1, imu, iv - 1) - hh(iz + 1, imu, iv - 1))),
        (
            iv - 1,
            iz,
            d1
            * (
                hh(iz - 1, imu, iv - 1)
                + hh(iz - 1, imu, iv)
                - hh(iz + 1, imu, iv - 1)
                - hh(iz + 1, imu, iv)
            ),
        ),
        (
            iv + 1,
            iz,
            d1
            * (
                hh(iz + 1, imu, iv)
                + hh(iz + 1, imu, iv + 1)
                - hh(iz - 1, imu, iv)
                - hh(iz - 1, imu, iv + 1)
            ),
        ),
        (iv + 2, iz, d2 * (hh(iz + 1, imu, iv + 1) - hh(iz - 1, imu, iv + 1))),
        (
            iv - 1,
            iz + 1,
            d1 * (hh(iz, imu, iv - 1) - hh(iz + 1, imu, iv))
            + d2
            * (
                hh(iz - 1, imu, iv - 1)
                - hh(iz + 1, imu, iv + 1)
                - hh(iz + 2, imu, iv)
                + hh(iz, imu, iv - 2)
            ),
        ),
        (
            iv,
            iz + 1,
            d1
            * (
                hh(iz, imu, iv - 1)
                + hh(iz + 1, imu, iv - 1)
                - hh(iz, imu, iv + 1)
                - hh(iz + 1, imu, iv + 1)
            ),
        ),
        (
            iv + 1,
            iz + 1,
            d1 * (hh(iz + 1, imu, iv) - hh(iz, imu, iv + 1))
            + d2
            * (
                hh(iz + 1, imu, iv - 1)
                - hh(iz - 1, imu, iv + 1)
                + hh(iz + 2, imu, iv)
                - hh(iz, imu, iv + 2)
            ),
        ),
        (iv, iz + 2, d2 * (hh(iz + 1, imu, iv - 1) - hh(iz + 1, imu, iv + 1))),
    )
    for col_iv, col_iz, value in entries:
        add(ispecies, iv, imu, iz, ix, iy, col_iv, col_iz, value)


def _add_gkw_igh_zero_two_coefficients(add, ispecies, iv, imu, iz, ix, iy, dum, position, hh):
    fac = 0.25
    if position == 2:
        val = fac * dum * (3.0 * hh(iz, imu, iv) - 4.0 * hh(iz - 1, imu, iv) + hh(iz - 2, imu, iv))
        entries = ((iv - 1, iz, -val), (iv + 1, iz, val))
        val = fac * dum * (hh(iz, imu, iv + 1) - hh(iz, imu, iv - 1))
        entries += ((iv, iz, -3.0 * val), (iv, iz - 1, 4.0 * val), (iv, iz - 2, -val))
    elif position == -2:
        val = fac * dum * (-3.0 * hh(iz, imu, iv) + 4.0 * hh(iz + 1, imu, iv) - hh(iz + 2, imu, iv))
        entries = ((iv - 1, iz, -val), (iv + 1, iz, val))
        val = fac * dum * (hh(iz, imu, iv + 1) - hh(iz, imu, iv - 1))
        entries += ((iv, iz, 3.0 * val), (iv, iz + 1, -4.0 * val), (iv, iz + 2, val))
    else:
        raise ValueError("position must be -2 or 2")
    for col_iv, col_iz, value in entries:
        add(ispecies, iv, imu, iz, ix, iy, col_iv, col_iz, value)


def _add_gkw_igh_two_coefficients(add, ispecies, iv, imu, iz, ix, iy, dum, position, hh):
    fac = 1.0 / 72.0
    if position == -1:
        val = (
            fac
            * dum
            * (
                -2.0 * hh(iz - 1, imu, iv)
                - 3.0 * hh(iz, imu, iv)
                + 6.0 * hh(iz + 1, imu, iv)
                - hh(iz + 2, imu, iv)
            )
        )
        entries = ((iv - 2, iz, val), (iv - 1, iz, -8.0 * val))
        entries += ((iv + 1, iz, 8.0 * val), (iv + 2, iz, -val))
        val = (
            -fac
            * dum
            * (
                hh(iz, imu, iv - 2)
                - 8.0 * hh(iz, imu, iv - 1)
                + 8.0 * hh(iz, imu, iv + 1)
                - hh(iz, imu, iv + 2)
            )
        )
        entries += (
            (iv, iz - 1, -2.0 * val),
            (iv, iz, -3.0 * val),
            (iv, iz + 1, 6.0 * val),
            (iv, iz + 2, -val),
        )
    elif position == 1:
        val = (
            fac
            * dum
            * (
                hh(iz - 2, imu, iv)
                - 6.0 * hh(iz - 1, imu, iv)
                + 3.0 * hh(iz, imu, iv)
                + 2.0 * hh(iz + 1, imu, iv)
            )
        )
        entries = ((iv - 2, iz, val), (iv - 1, iz, -8.0 * val))
        entries += ((iv + 1, iz, 8.0 * val), (iv + 2, iz, -val))
        val = (
            -fac
            * dum
            * (
                hh(iz, imu, iv - 2)
                - 8.0 * hh(iz, imu, iv - 1)
                + 8.0 * hh(iz, imu, iv + 1)
                - hh(iz, imu, iv + 2)
            )
        )
        entries += (
            (iv, iz - 2, val),
            (iv, iz - 1, -6.0 * val),
            (iv, iz, 3.0 * val),
            (iv, iz + 1, 2.0 * val),
        )
    else:
        raise ValueError("position must be -1 or 1")
    for col_iv, col_iz, value in entries:
        add(ispecies, iv, imu, iz, ix, iy, col_iv, col_iz, value)


def _add_gkw_igh_diffusion_coefficients(add, ispecies, iv, imu, iz, ix, iy, speed, spacing, axis):
    if speed <= 0.0:
        return
    coefficient = -abs(float(speed)) / (12.0 * float(spacing))
    stencil = (1.0, -4.0, 6.0, -4.0, 1.0)
    for offset, weight in zip(range(-2, 3), stencil, strict=True):
        col_iv = iv + offset if axis == "vpar" else iv
        col_iz = iz + offset if axis == "z" else iz
        add(ispecies, iv, imu, iz, ix, iy, col_iv, col_iz, coefficient * weight)


def parallel_streaming(distribution, D_z, parallel_coefficient):
    """Return ``-a_parallel partial_z f``."""

    coefficient = jnp.asarray(parallel_coefficient)
    n_species = _coefficient_species_count(coefficient, single_ndim=2, name="parallel_coefficient")
    original_ndim = jnp.asarray(distribution).ndim
    distribution_s = _distribution_with_species_axis(distribution, n_species)
    coefficient_s = _with_species_axis(
        coefficient,
        n_species,
        "parallel_coefficient",
        single_ndim=2,
    )
    dz_distribution = _parallel_derivative(distribution_s, D_z)
    result = -coefficient_s[:, :, None, :, None, None] * dz_distribution
    return _restore_distribution_shape(result, original_ndim)


def magnetic_drift_advection(distribution, drift_frequency):
    """Return ``-i omega_d f``."""

    drift = jnp.asarray(drift_frequency)
    n_species = _coefficient_species_count(drift, single_ndim=5, name="drift_frequency")
    original_ndim = jnp.asarray(distribution).ndim
    distribution_s = _distribution_with_species_axis(distribution, n_species)
    drift_s = _with_species_axis(drift, n_species, "drift_frequency", single_ndim=5)
    result = -1j * drift_s * distribution_s
    return _restore_distribution_shape(result, original_ndim)


def mirror_force(distribution, D_vpar, mirror_coefficient):
    """Return ``a_mu partial_vparallel f``."""

    coefficient = jnp.asarray(mirror_coefficient)
    n_species = _coefficient_species_count(coefficient, single_ndim=2, name="mirror_coefficient")
    original_ndim = jnp.asarray(distribution).ndim
    distribution_s = _distribution_with_species_axis(distribution, n_species)
    coefficient_s = _with_species_axis(
        coefficient,
        n_species,
        "mirror_coefficient",
        single_ndim=2,
    )
    dv_distribution = _vpar_derivative(distribution_s, D_vpar)
    result = coefficient_s[:, None, :, :, None, None] * dv_distribution
    return _restore_distribution_shape(result, original_ndim)


def equilibrium_drive(phi, precompute: LinearRHSPrecompute):
    """Return ``i ky E_y J0 phi F_M Xi``."""

    gyro_phi = _gyroaveraged_potential(phi, precompute)
    coefficient = (
        1j
        * precompute.E_y[None, None, None, :, None, None]
        * precompute.ky[None, None, None, None, None, :]
    )
    result = (
        coefficient
        * gyro_phi[:, None, :, :, :, :]
        * precompute.maxwellian[..., None, None]
        * precompute.drive_factor[..., None, None]
    )
    return _drop_single_species(result, precompute.n_species)


def parallel_field_drive(phi, D_z, precompute: LinearRHSPrecompute):
    """Return ``-(Z/T) a_parallel F_M partial_z(J0 phi)``."""

    gyro_phi = _gyroaveraged_potential(phi, precompute)
    dz_gyro_phi = _parallel_derivative(gyro_phi, D_z)
    result = (
        -precompute.charge_over_temperature[:, None, None, None, None, None]
        * precompute.parallel_streaming_coeff[:, :, None, :, None, None]
        * precompute.maxwellian[..., None, None]
        * dz_gyro_phi[:, None, :, :, :, :]
    )
    return _drop_single_species(result, precompute.n_species)


def gkw_parallel_streaming(distribution, precompute: LinearRHSPrecompute):
    """Return GKW upwinded Term I plus fused ``disp_par`` recurrence control."""

    coefficient = jnp.asarray(precompute.parallel_streaming_coeff)
    n_species = _coefficient_species_count(coefficient, single_ndim=2, name="parallel_coefficient")
    original_ndim = jnp.asarray(distribution).ndim
    distribution_s = _distribution_with_species_axis(distribution, n_species)
    coefficient_s = _with_species_axis(
        coefficient,
        n_species,
        "parallel_coefficient",
        single_ndim=2,
    )
    upar = -coefficient_s
    d1_pos = _apply_gkw_parallel_stencil_table(
        distribution_s, precompute.gkw_parallel_stencil.d1_pos, precompute.gkw_parallel_stencil
    )
    d1_neg = _apply_gkw_parallel_stencil_table(
        distribution_s, precompute.gkw_parallel_stencil.d1_neg, precompute.gkw_parallel_stencil
    )
    d4_pos = _apply_gkw_parallel_stencil_table(
        distribution_s, precompute.gkw_parallel_stencil.d4_pos, precompute.gkw_parallel_stencil
    )
    d4_neg = _apply_gkw_parallel_stencil_table(
        distribution_s, precompute.gkw_parallel_stencil.d4_neg, precompute.gkw_parallel_stencil
    )
    sign = upar[:, :, None, :, None, None]
    d1 = jnp.where(sign > 0.0, d1_pos, d1_neg)
    d4 = jnp.where(sign > 0.0, d4_pos, d4_neg)
    recurrence = jnp.asarray(precompute.parallel_recurrence_coeff)
    recurrence_s = _with_species_axis(
        recurrence,
        n_species,
        "parallel_recurrence_coeff",
        single_ndim=2,
    )
    result = sign * d1 + recurrence_s[:, :, None, :, None, None] * d4
    return _restore_distribution_shape(result, original_ndim)


def gkw_igh_streaming_mirror(distribution, precompute: LinearRHSPrecompute):
    """Return GKW's fused ``igh`` Term I/IV plus ``disp_par``/``disp_vp``."""

    coefficients = jnp.asarray(precompute.gkw_igh_stencil.coefficients)
    n_species = int(coefficients.shape[2])
    original_ndim = jnp.asarray(distribution).ndim
    distribution_s = _distribution_with_species_axis(distribution, n_species)
    result = jnp.zeros_like(distribution_s)
    for v_shift_index, delta_v in enumerate(range(-2, 3)):
        shifted_v = _shift_vpar_zero(
            distribution_s,
            delta_v,
            precompute.gkw_igh_stencil.valid_v_shift[v_shift_index],
        )
        for z_shift_index, delta_z in enumerate(range(-2, 3)):
            shifted = _apply_gkw_parallel_single_shift(
                shifted_v,
                delta_z,
                precompute.gkw_parallel_stencil,
            )
            result = result + coefficients[v_shift_index, z_shift_index] * shifted
    return _restore_distribution_shape(result, original_ndim)


def gkw_parallel_field_drive(phi, precompute: LinearRHSPrecompute):
    """Return GKW upwinded Term VII for the gyroaveraged potential."""

    gyro_phi = _gyroaveraged_potential(phi, precompute)
    d1_pos = _apply_gkw_parallel_stencil_table(
        gyro_phi,
        precompute.gkw_parallel_stencil.d1_pos,
        precompute.gkw_parallel_stencil,
    )
    d1_neg = _apply_gkw_parallel_stencil_table(
        gyro_phi,
        precompute.gkw_parallel_stencil.d1_neg,
        precompute.gkw_parallel_stencil,
    )
    coefficient = (
        -precompute.charge_over_temperature[:, None, None, None, None, None]
        * precompute.parallel_streaming_coeff[:, :, None, :, None, None]
        * precompute.maxwellian[..., None, None]
    )
    selected_derivative = jnp.where(
        coefficient < 0.0,
        d1_pos[:, None, :, :, :, :],
        d1_neg[:, None, :, :, :, :],
    )
    result = coefficient * selected_derivative
    return _drop_single_species(result, precompute.n_species)


def drift_field_drive(phi, precompute: LinearRHSPrecompute):
    """Return ``-(Z/T) i omega_d F_M J0 phi``."""

    gyro_phi = _gyroaveraged_potential(phi, precompute)
    result = (
        -precompute.charge_over_temperature[:, None, None, None, None, None]
        * 1j
        * precompute.magnetic_drift_frequency
        * precompute.maxwellian[..., None, None]
        * gyro_phi[:, None, :, :, :, :]
    )
    return _drop_single_species(result, precompute.n_species)


def dissipation(distribution, damping_rate=None):
    """Return optional linear perpendicular damping, zero by default."""

    distribution = jnp.asarray(distribution)
    if damping_rate is None:
        return jnp.zeros_like(distribution)
    damping = jnp.asarray(damping_rate)
    if damping.ndim == 2:
        damping = damping.reshape((1,) * (distribution.ndim - 2) + damping.shape)
    return -damping * distribution


def parallel_recurrence_control(distribution, operator, coefficient):
    """Return the GKW-scaled parallel fourth-order recurrence-control term."""

    distribution = jnp.asarray(distribution)
    coefficient = jnp.asarray(coefficient)
    if coefficient.ndim == 2:
        n_species = 1
        coefficient_s = coefficient[None, ...]
    elif coefficient.ndim == 3:
        n_species = int(coefficient.shape[0])
        coefficient_s = coefficient
    else:
        raise ValueError(
            "parallel_recurrence_coeff must have shape (n_vpar,n_z) or (n_species,n_vpar,n_z)"
        )
    original_ndim = distribution.ndim
    distribution_s = _distribution_with_species_axis(distribution, n_species)
    d4_distribution = _parallel_derivative(distribution_s, operator)
    result = coefficient_s[:, :, None, :, None, None] * d4_distribution
    return _restore_distribution_shape(result, original_ndim)


def velocity_recurrence_control(distribution, operator, coefficient):
    """Return the GKW-scaled parallel-velocity recurrence-control term."""

    distribution = jnp.asarray(distribution)
    coefficient = jnp.asarray(coefficient)
    if coefficient.ndim == 2:
        n_species = 1
        coefficient_s = coefficient[None, ...]
    elif coefficient.ndim == 3:
        n_species = int(coefficient.shape[0])
        coefficient_s = coefficient
    else:
        raise ValueError(
            "velocity_recurrence_coeff must have shape (n_mu,n_z) or (n_species,n_mu,n_z)"
        )
    original_ndim = distribution.ndim
    distribution_s = _distribution_with_species_axis(distribution, n_species)
    d4_distribution = _vpar_derivative(distribution_s, operator)
    result = coefficient_s[:, None, :, :, None, None] * d4_distribution
    return _restore_distribution_shape(result, original_ndim)


def linear_residual_from_phi(distribution, phi, precompute: LinearRHSPrecompute):
    """Assemble the linear RHS for a supplied electrostatic potential."""

    if precompute.parallel_derivative_model == "gkw_igh":
        parallel_term = gkw_igh_streaming_mirror(distribution, precompute)
        parallel_field_term = gkw_parallel_field_drive(phi, precompute)
        mirror_term = jnp.zeros_like(distribution)
        parallel_recurrence_term = jnp.zeros_like(distribution)
        velocity_recurrence_term = jnp.zeros_like(distribution)
    elif precompute.parallel_derivative_model == "gkw_upwind":
        parallel_term = gkw_parallel_streaming(distribution, precompute)
        parallel_field_term = gkw_parallel_field_drive(phi, precompute)
        mirror_term = mirror_force(distribution, precompute.D_vpar, precompute.mirror_force_coeff)
        parallel_recurrence_term = jnp.zeros_like(distribution)
        velocity_recurrence_term = velocity_recurrence_control(
            distribution,
            precompute.velocity_recurrence_operator,
            precompute.velocity_recurrence_coeff,
        )
    else:
        parallel_term = parallel_streaming(
            distribution,
            precompute.D_z,
            precompute.parallel_streaming_coeff,
        )
        parallel_field_term = parallel_field_drive(phi, precompute.D_z, precompute)
        mirror_term = mirror_force(distribution, precompute.D_vpar, precompute.mirror_force_coeff)
        parallel_recurrence_term = parallel_recurrence_control(
            distribution,
            precompute.parallel_recurrence_operator,
            precompute.parallel_recurrence_coeff,
        )
        velocity_recurrence_term = velocity_recurrence_control(
            distribution,
            precompute.velocity_recurrence_operator,
            precompute.velocity_recurrence_coeff,
        )
    return (
        parallel_term
        + magnetic_drift_advection(distribution, precompute.magnetic_drift_frequency)
        + mirror_term
        + equilibrium_drive(phi, precompute)
        + parallel_field_term
        + drift_field_drive(phi, precompute)
        + dissipation(distribution, precompute.perpendicular_damping)
        + parallel_recurrence_term
        + velocity_recurrence_term
    )


def _gyroaveraged_potential(phi, precompute: LinearRHSPrecompute):
    phi = jnp.asarray(phi)
    if phi.ndim != 3:
        raise ValueError("phi must have shape (n_z,n_kx,n_ky)")
    return precompute.flr_factors.bessel_j0 * phi[None, None, :, :, :]


def _parallel_derivative(values, D_z):
    axis = jnp.asarray(values).ndim - 3
    return _apply_matrix_along_axis(D_z, values, axis)


def _vpar_derivative(values, D_vpar):
    return _apply_matrix_along_axis(D_vpar, values, axis=1)


def _apply_matrix_along_axis(matrix, values, axis: int):
    values = jnp.asarray(values)
    matrix = jnp.asarray(matrix)
    moved = jnp.moveaxis(values, axis, 0)
    differentiated = jnp.tensordot(matrix, moved, axes=((1,), (0,)))
    return jnp.moveaxis(differentiated, 0, axis)


def _apply_gkw_parallel_stencil_table(values, coefficients, stencil: GKWParallelStencil):
    values = jnp.asarray(values)
    prefix_shape = values.shape[:-3]
    n_z, n_kx, n_ky = values.shape[-3:]
    flat = jnp.reshape(values, (-1, n_z, n_kx, n_ky))
    output = jnp.zeros_like(flat)
    ky_idx = jnp.reshape(jnp.arange(n_ky, dtype=jnp.int32), (1, 1, n_ky))
    for shift_index in range(9):
        shifted = flat[
            :,
            stencil.s_shift[shift_index],
            stencil.kx_shift[shift_index],
            ky_idx,
        ]
        shifted = jnp.where(stencil.valid_shift[shift_index][None, :, :, :], shifted, 0.0)
        output = output + coefficients[shift_index][None, :, :, :] * shifted
    return jnp.reshape(output, prefix_shape + (n_z, n_kx, n_ky))


def _apply_gkw_parallel_single_shift(values, delta_z: int, stencil: GKWParallelStencil):
    values = jnp.asarray(values)
    prefix_shape = values.shape[:-3]
    n_z, n_kx, n_ky = values.shape[-3:]
    flat = jnp.reshape(values, (-1, n_z, n_kx, n_ky))
    shift_index = int(delta_z) + 4
    ky_idx = jnp.reshape(jnp.arange(n_ky, dtype=jnp.int32), (1, 1, n_ky))
    shifted = flat[
        :,
        stencil.s_shift[shift_index],
        stencil.kx_shift[shift_index],
        ky_idx,
    ]
    shifted = jnp.where(stencil.valid_shift[shift_index][None, :, :, :], shifted, 0.0)
    return jnp.reshape(shifted, prefix_shape + (n_z, n_kx, n_ky))


def _shift_vpar_zero(values, delta_v: int, valid_v_shift):
    values = jnp.asarray(values)
    n_vpar = values.shape[1]
    indices = jnp.arange(n_vpar, dtype=jnp.int32) + int(delta_v)
    clipped = jnp.clip(indices, 0, n_vpar - 1)
    shifted = jnp.take(values, clipped, axis=1)
    valid = jnp.asarray(valid_v_shift, dtype=bool)[None, :, None, None, None, None]
    return jnp.where(valid, shifted, 0.0)


def _k_perp_squared(geometry, fourier_grid: FourierGrid):
    kx = fourier_grid.kx[None, :, None]
    ky = fourier_grid.ky[None, None, :]
    return (
        geometry.g_xx[:, None, None] * kx**2
        + 2.0 * geometry.g_xy[:, None, None] * kx * ky
        + geometry.g_yy[:, None, None] * ky**2
    )


def _with_species_axis_flr(flr: FLRFactors, n_species: int) -> FLRFactors:
    return FLRFactors(
        bessel_argument=_with_species_axis(
            flr.bessel_argument,
            n_species,
            "flr.bessel_argument",
            single_ndim=4,
        ),
        bessel_j0=_with_species_axis(
            flr.bessel_j0,
            n_species,
            "flr.bessel_j0",
            single_ndim=4,
        ),
        polarization_argument=_with_species_axis(
            flr.polarization_argument,
            n_species,
            "flr.polarization_argument",
            single_ndim=3,
        ),
        gamma0=_with_species_axis(
            flr.gamma0,
            n_species,
            "flr.gamma0",
            single_ndim=3,
        ),
    )


def _with_species_axis(array, n_species: int, name: str, *, single_ndim: int):
    array = jnp.asarray(array)
    if n_species == 1 and array.ndim == single_ndim:
        return array[None, ...]
    if array.ndim == single_ndim + 1 and array.shape[0] == n_species:
        return array
    raise ValueError(
        f"{name} must have a single-species shape with {single_ndim} dimensions or "
        f"a leading species axis with {single_ndim + 1} dimensions; got {array.shape}"
    )


def _distribution_with_species_axis(distribution, n_species: int):
    distribution = jnp.asarray(distribution)
    if n_species == 1 and distribution.ndim == 5:
        return distribution[None, ...]
    if distribution.ndim == 6 and distribution.shape[0] == n_species:
        return distribution
    raise ValueError(
        "distribution must have shape (n_vpar,n_mu,n_z,n_kx,n_ky) "
        "or (n_species,n_vpar,n_mu,n_z,n_kx,n_ky)"
    )


def _restore_distribution_shape(distribution, original_ndim: int):
    if original_ndim == 5:
        return distribution[0]
    if original_ndim == 6:
        return distribution
    raise ValueError(
        "distribution must have shape (n_vpar,n_mu,n_z,n_kx,n_ky) "
        "or (n_species,n_vpar,n_mu,n_z,n_kx,n_ky)"
    )


def _drop_single_species(values, n_species: int):
    if n_species == 1:
        return values[0]
    return values


def _coefficient_species_count(array, *, single_ndim: int, name: str) -> int:
    if array.ndim == single_ndim:
        return 1
    if array.ndim == single_ndim + 1:
        return int(array.shape[0])
    raise ValueError(
        f"{name} must have {single_ndim} dimensions or a leading species axis; got {array.shape}"
    )


def _normalize_perpendicular_damping(damping, n_kx: int, n_ky: int, *, dtype):
    if damping is None:
        return jnp.zeros((n_kx, n_ky), dtype=dtype)
    damping = jnp.asarray(damping, dtype=dtype)
    if damping.ndim == 0:
        return jnp.full((n_kx, n_ky), damping, dtype=dtype)
    if damping.shape != (n_kx, n_ky):
        raise ValueError("perpendicular_damping must be scalar or have shape (n_kx,n_ky)")
    return damping


def _gkw_parallel_recurrence_operator(parallel_grid: ParallelGrid, *, dtype):
    """Return the negative-semidefinite GKW-scaled fourth derivative operator.

    GKW's ``disp_par`` stencil adds
    ``[-1, 4, -6, 4, -1] / (12 * ds)`` to the parallel streaming stencil.  For
    the spectral target backend, the equivalent low-wavenumber scaling is
    ``-(ds**3 / 12) * d^4/dz^4``; this keeps recurrence control inside the
    residual while preserving the benchmark discretization's resolution
    scaling.
    """

    if parallel_grid.backend == "finite_difference":
        spacing = float(jnp.sum(jnp.asarray(parallel_grid.w_z)) / parallel_grid.D_z.shape[0])
        return _finite_difference_recurrence_operator(
            parallel_grid.D_z.shape[0],
            spacing,
            periodic=True,
            dtype=dtype,
        )
    d_z = jnp.asarray(parallel_grid.D_z, dtype=dtype)
    d4 = d_z @ d_z @ d_z @ d_z
    spacing = jnp.sum(jnp.asarray(parallel_grid.w_z, dtype=dtype)) / d_z.shape[0]
    return -(spacing**3 / 12.0) * d4


def _gkw_velocity_recurrence_operator(velocity_grid: VelocityGrid, *, dtype):
    """Return the GKW-scaled fourth derivative operator in ``v_parallel``."""

    if velocity_grid.backend == "finite_difference":
        spacing = float(jnp.sum(jnp.asarray(velocity_grid.w_vpar)) / velocity_grid.vpar.shape[0])
        return _finite_difference_recurrence_operator(
            velocity_grid.vpar.shape[0],
            spacing,
            periodic=False,
            dtype=dtype,
        )
    d_vpar = jnp.asarray(velocity_grid.D_vpar, dtype=dtype)
    d4 = d_vpar @ d_vpar @ d_vpar @ d_vpar
    vpar = jnp.asarray(velocity_grid.vpar, dtype=dtype)
    spacing = (jnp.max(vpar) - jnp.min(vpar)) / jnp.asarray(vpar.shape[0] - 1, dtype=dtype)
    if velocity_grid.backend == "finite_difference":
        spacing = jnp.sum(jnp.asarray(velocity_grid.w_vpar, dtype=dtype)) / vpar.shape[0]
    return -(spacing**3 / 12.0) * d4


def _finite_difference_recurrence_operator(n: int, spacing: float, *, periodic: bool, dtype):
    matrix = np.zeros((n, n), dtype=float)
    coefficients = {
        -2: -1.0 / (12.0 * spacing),
        -1: 4.0 / (12.0 * spacing),
        0: -6.0 / (12.0 * spacing),
        1: 4.0 / (12.0 * spacing),
        2: -1.0 / (12.0 * spacing),
    }
    for row in range(n):
        for offset, value in coefficients.items():
            col = row + offset
            if periodic:
                matrix[row, col % n] += value
            elif 0 <= col < n:
                matrix[row, col] += value
    return jnp.asarray(matrix, dtype=dtype)


def _gkw_parallel_position_classes(
    ixplus: np.ndarray,
    ixminus: np.ndarray,
    iyzero: int,
    n_z: int,
) -> np.ndarray:
    pos = np.zeros((n_z,) + ixplus.shape, dtype=np.int8)
    zonal = np.zeros(ixplus.shape, dtype=bool)
    if 0 <= iyzero < ixplus.shape[1]:
        zonal[:, iyzero] = True
    left_open = np.logical_and(ixminus < 0, ~zonal)
    right_open = np.logical_and(ixplus < 0, ~zonal)
    if n_z >= 1:
        pos[0, left_open] = -2
        pos[n_z - 1, right_open] = 2
    if n_z >= 2:
        pos[1, left_open] = -1
        pos[n_z - 2, right_open] = 1
    return pos


def _gkw_parallel_shift_maps(
    ixplus: np.ndarray,
    ixminus: np.ndarray,
    iyzero: int,
    n_z: int,
    *,
    max_shift: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_kx, n_ky = ixplus.shape
    offsets = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
    shape = (offsets.size, n_z, n_kx, n_ky)
    s_shift = np.zeros(shape, dtype=np.int32)
    kx_shift = np.zeros(shape, dtype=np.int32)
    valid = np.zeros(shape, dtype=bool)
    for offset_index, delta_s in enumerate(offsets):
        for iz in range(n_z):
            for ix in range(n_kx):
                for iy in range(n_ky):
                    target_z = iz + int(delta_s)
                    target_kx = ix
                    ok = True
                    if target_z < 0:
                        if iy == iyzero:
                            target_z += n_z
                        else:
                            connected = int(ixminus[ix, iy])
                            if connected >= 0:
                                target_kx = connected
                                target_z += n_z
                            else:
                                ok = False
                    elif target_z >= n_z:
                        if iy == iyzero:
                            target_z -= n_z
                        else:
                            connected = int(ixplus[ix, iy])
                            if connected >= 0:
                                target_kx = connected
                                target_z -= n_z
                            else:
                                ok = False
                    if ok and 0 <= target_z < n_z:
                        s_shift[offset_index, iz, ix, iy] = target_z
                        kx_shift[offset_index, iz, ix, iy] = target_kx
                        valid[offset_index, iz, ix, iy] = True
    return s_shift, kx_shift, valid


def _parallel_recurrence_coefficient(vpar, parallel_coeff, rate: float, velocity_model: str):
    if rate < 0.0:
        raise ValueError("parallel_recurrence_rate must be nonnegative")
    if velocity_model not in ("local", "rms"):
        raise ValueError("parallel_recurrence_velocity_model must be 'local' or 'rms'")
    parallel_coeff = jnp.asarray(parallel_coeff)
    if velocity_model == "local":
        speed = jnp.abs(parallel_coeff)
    else:
        vpar = jnp.asarray(vpar, dtype=parallel_coeff.dtype)
        rms = jnp.sqrt(jnp.mean(vpar**2))
        eps = jnp.asarray(1.0e-300, dtype=parallel_coeff.dtype)
        if parallel_coeff.ndim == 2:
            v_abs = jnp.abs(vpar)[:, None]
            safe_v_abs = jnp.where(v_abs > eps, v_abs, 1.0)
            scale = jnp.where(v_abs > eps, jnp.abs(parallel_coeff) / safe_v_abs, 0.0)
            speed = rms * jnp.max(scale, axis=0, keepdims=True)
        elif parallel_coeff.ndim == 3:
            v_abs = jnp.abs(vpar)[None, :, None]
            safe_v_abs = jnp.where(v_abs > eps, v_abs, 1.0)
            scale = jnp.where(v_abs > eps, jnp.abs(parallel_coeff) / safe_v_abs, 0.0)
            speed = rms * jnp.max(scale, axis=1, keepdims=True)
        else:
            raise ValueError(
                "parallel_coeff must have shape (n_vpar,n_z) or (n_species,n_vpar,n_z)"
            )
        speed = jnp.broadcast_to(speed, parallel_coeff.shape)
    return jnp.asarray(rate, dtype=parallel_coeff.dtype) * speed


def _velocity_recurrence_coefficient(mu, mirror_coeff, rate: float, velocity_model: str):
    if rate < 0.0:
        raise ValueError("velocity_recurrence_rate must be nonnegative")
    if velocity_model not in ("local", "rms"):
        raise ValueError("velocity_recurrence_velocity_model must be 'local' or 'rms'")
    mirror_coeff = jnp.asarray(mirror_coeff)
    if velocity_model == "local":
        speed = jnp.abs(mirror_coeff)
    else:
        mu = jnp.asarray(mu, dtype=mirror_coeff.dtype)
        rms = jnp.sqrt(jnp.mean(mu**2))
        eps = jnp.asarray(1.0e-300, dtype=mirror_coeff.dtype)
        if mirror_coeff.ndim == 2:
            mu_abs = jnp.abs(mu)[:, None]
            safe_mu_abs = jnp.where(mu_abs > eps, mu_abs, 1.0)
            scale = jnp.where(mu_abs > eps, jnp.abs(mirror_coeff) / safe_mu_abs, 0.0)
            speed = rms * jnp.max(scale, axis=0, keepdims=True)
        elif mirror_coeff.ndim == 3:
            mu_abs = jnp.abs(mu)[None, :, None]
            safe_mu_abs = jnp.where(mu_abs > eps, mu_abs, 1.0)
            scale = jnp.where(mu_abs > eps, jnp.abs(mirror_coeff) / safe_mu_abs, 0.0)
            speed = rms * jnp.max(scale, axis=1, keepdims=True)
        else:
            raise ValueError("mirror_coeff must have shape (n_mu,n_z) or (n_species,n_mu,n_z)")
        speed = jnp.broadcast_to(speed, mirror_coeff.shape)
    return jnp.asarray(rate, dtype=mirror_coeff.dtype) * speed


def _as_species_tuple(species: SpeciesParams | tuple[SpeciesParams, ...]):
    return species if isinstance(species, tuple) else (species,)
