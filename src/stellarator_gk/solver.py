"""Matrix-free linear gyrokinetic residual assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax

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

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("rhs", "field")
    _static_fields: ClassVar[tuple[str, ...]] = ("field_model", "n_species")

    def __post_init__(self):
        if self.field_model not in ("adiabatic", "kinetic"):
            raise ValueError("field_model must be 'adiabatic' or 'kinetic'")
        if self.n_species < 1:
            raise ValueError("n_species must be at least 1")


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
            )
        else:
            field = build_kinetic_quasineutrality_precompute(
                velocity_grid,
                geometry.B,
                rhs.flr_factors,
                species,
                fourier_grid=fourier_grid,
            )
    return LinearResidualPrecompute(
        rhs=rhs,
        field=field,
        field_model=normalized_model,
        n_species=rhs.n_species,
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
    return linear_residual_from_phi(distribution, solved_phi, precompute.rhs)


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
