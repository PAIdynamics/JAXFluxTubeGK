"""Differentiable linear-growth and quasilinear objective helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from .operators import mode_chain_mask
from .solver import LinearResidualPrecompute, linear_residual
from .time_advance import integrate_fixed_step, linear_growth_diagnostics
from .types import ModeConnectivity, _PyTreeDataclass


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearObjectiveValues(_PyTreeDataclass):
    """Collected per-``ky`` and scalar linear objective values."""

    growth_rate: object
    frequency: object
    kperp2_average: object
    mode_structure: object
    max_growth_rate: object
    selected_growth_rate: object
    quasilinear_proxy: object
    mode_structure_penalty: object

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "growth_rate",
        "frequency",
        "kperp2_average",
        "mode_structure",
        "max_growth_rate",
        "selected_growth_rate",
        "quasilinear_proxy",
        "mode_structure_penalty",
    )


def max_growth_objective(growth_rates, active_mask=None):
    """Return the maximum active linear growth rate."""

    growth_rates = jnp.asarray(growth_rates)
    if active_mask is None:
        return jnp.max(growth_rates)
    active = jnp.asarray(active_mask, dtype=bool)
    return jnp.max(jnp.where(active, growth_rates, -jnp.inf))


def selected_growth_objective(growth_rates, ky_index: int):
    """Return the growth rate for one ``ky`` index."""

    return jnp.asarray(growth_rates)[ky_index]


def weighted_quasilinear_proxy(
    growth_rates,
    kperp2_average_values,
    *,
    active_mask=None,
    epsilon: float = 1.0e-12,
    positive_only: bool = True,
    softplus_temperature: float | None = None,
):
    """Return ``sum max(gamma,0) / (<k_perp^2> + epsilon)`` over active modes."""

    growth_rates = jnp.asarray(growth_rates)
    kperp = jnp.asarray(kperp2_average_values)
    if softplus_temperature is not None:
        tau = jnp.asarray(softplus_temperature, dtype=growth_rates.dtype)
        positive_growth = tau * jnp.logaddexp(0.0, growth_rates / tau)
    elif positive_only:
        positive_growth = jnp.maximum(growth_rates, 0.0)
    else:
        positive_growth = growth_rates
    contribution = positive_growth / (kperp + jnp.asarray(epsilon, dtype=kperp.dtype))
    if active_mask is not None:
        contribution = jnp.where(jnp.asarray(active_mask, dtype=bool), contribution, 0.0)
    return jnp.sum(contribution)


def kperp2_weighted_average(
    kperp_squared,
    field,
    *,
    w_z=None,
    connectivity: ModeConnectivity | None = None,
    amplitude_floor: float = 1.0e-300,
):
    """Return per-``ky`` mode-structure weighted ``<k_perp^2>``."""

    kperp = jnp.asarray(kperp_squared)
    field = jnp.asarray(field)
    if kperp.shape != field.shape:
        raise ValueError("kperp_squared and field must both have shape (n_z,n_kx,n_ky)")
    weights = _z_weights(field, w_z)
    mask = mode_chain_mask(field.shape[1], field.shape[2], connectivity, dtype=field.real.dtype)
    power = mask[None, :, :] * jnp.abs(field) ** 2
    numerator = jnp.sum(weights[:, None, None] * kperp * power, axis=(0, 1))
    denominator = jnp.sum(weights[:, None, None] * power, axis=(0, 1))
    floor = jnp.asarray(amplitude_floor, dtype=denominator.dtype)
    return numerator / jnp.maximum(denominator, floor)


def mode_structure_penalty(
    mode_structure,
    target=None,
    *,
    w_z=None,
    connectivity: ModeConnectivity | None = None,
    phase_align: bool = False,
):
    """Return a weighted squared-distance penalty for a mode structure.

    When ``phase_align`` is true, each ``ky`` target is first multiplied by its
    optimal unit complex phase. This removes the arbitrary eigenfunction phase
    without allowing an amplitude rescaling.
    """

    mode_structure = jnp.asarray(mode_structure)
    target = jnp.zeros_like(mode_structure) if target is None else jnp.asarray(target)
    if target.shape != mode_structure.shape:
        raise ValueError("target must have the same shape as mode_structure")
    weights = _z_weights(mode_structure, w_z)
    mask = mode_chain_mask(
        mode_structure.shape[1],
        mode_structure.shape[2],
        connectivity,
        dtype=mode_structure.real.dtype,
    )
    if phase_align:
        overlap = jnp.sum(
            weights[:, None, None]
            * mask[None, :, :]
            * jnp.conj(target)
            * mode_structure,
            axis=(0, 1),
        )
        overlap_abs = jnp.abs(overlap)
        phase = jnp.where(overlap_abs > 0.0, overlap / overlap_abs, 1.0 + 0.0j)
        target = target * phase[None, None, :]
    error = jnp.abs(mode_structure - target) ** 2
    numerator = jnp.sum(weights[:, None, None] * mask[None, :, :] * error)
    denominator = jnp.sum(weights) * jnp.maximum(jnp.sum(mask), 1.0)
    return numerator / denominator


def linear_growth_objectives(
    field_start,
    field_end,
    t_start,
    t_end,
    kperp_squared,
    *,
    selected_ky: int | None = None,
    target_mode_structure=None,
    w_z=None,
    connectivity: ModeConnectivity | None = None,
    active_mask=None,
    amplitude_floor: float = 1.0e-300,
    kperp_epsilon: float = 1.0e-12,
    softplus_temperature: float | None = None,
    phase_align_mode_structure: bool = False,
) -> LinearObjectiveValues:
    """Compute scalar optimization objectives from two potential snapshots."""

    diagnostics = linear_growth_diagnostics(
        field_start,
        field_end,
        t_start,
        t_end,
        w_z=w_z,
        connectivity=connectivity,
        amplitude_floor=amplitude_floor,
    )
    kperp_average = kperp2_weighted_average(
        kperp_squared,
        field_end,
        w_z=w_z,
        connectivity=connectivity,
        amplitude_floor=amplitude_floor,
    )
    selected = (
        max_growth_objective(diagnostics.growth_rate, active_mask=active_mask)
        if selected_ky is None
        else selected_growth_objective(diagnostics.growth_rate, selected_ky)
    )
    penalty = mode_structure_penalty(
        diagnostics.mode_structure,
        target=target_mode_structure,
        w_z=w_z,
        connectivity=connectivity,
        phase_align=phase_align_mode_structure,
    )
    return LinearObjectiveValues(
        growth_rate=diagnostics.growth_rate,
        frequency=diagnostics.frequency,
        kperp2_average=kperp_average,
        mode_structure=diagnostics.mode_structure,
        max_growth_rate=max_growth_objective(diagnostics.growth_rate, active_mask=active_mask),
        selected_growth_rate=selected,
        quasilinear_proxy=weighted_quasilinear_proxy(
            diagnostics.growth_rate,
            kperp_average,
            active_mask=active_mask,
            epsilon=kperp_epsilon,
            softplus_temperature=softplus_temperature,
        ),
        mode_structure_penalty=penalty,
    )


def initial_value_growth_objectives(
    initial_state,
    precomputed: LinearResidualPrecompute,
    dt,
    n_steps: int,
    kperp_squared,
    *,
    selected_ky: int | None = None,
    target_mode_structure=None,
    w_z=None,
    connectivity: ModeConnectivity | None = None,
    active_mask=None,
    amplitude_floor: float = 1.0e-300,
    kperp_epsilon: float = 1.0e-12,
    softplus_temperature: float | None = None,
    store_history: bool = True,
    phase_align_mode_structure: bool = False,
) -> LinearObjectiveValues:
    """Run a short fixed-step linear solve and return growth objectives."""

    result = integrate_fixed_step(
        initial_state,
        dt,
        n_steps,
        linear_residual,
        precomputed,
        store_history=store_history,
    )
    field_start = solve_field_from_state(result.history[0], precomputed)
    field_end = solve_field_from_state(result.state, precomputed)
    return linear_growth_objectives(
        field_start,
        field_end,
        result.times[0],
        result.times[-1],
        kperp_squared,
        selected_ky=selected_ky,
        target_mode_structure=target_mode_structure,
        w_z=w_z,
        connectivity=connectivity,
        active_mask=active_mask,
        amplitude_floor=amplitude_floor,
        kperp_epsilon=kperp_epsilon,
        softplus_temperature=softplus_temperature,
        phase_align_mode_structure=phase_align_mode_structure,
    )


def solve_field_from_state(state, precomputed: LinearResidualPrecompute):
    """Solve the electrostatic field for a state using a residual precompute."""

    if precomputed.field_model == "adiabatic":
        from .physics.quasineutrality import solve_adiabatic_electron_phi

        return solve_adiabatic_electron_phi(state, precomputed.field)
    if precomputed.field_model == "kinetic":
        from .physics.quasineutrality import solve_kinetic_electron_phi

        return solve_kinetic_electron_phi(state, precomputed.field)
    raise ValueError(f"unsupported field_model {precomputed.field_model!r}")


def _z_weights(field, w_z):
    if w_z is None:
        return jnp.ones((field.shape[0],), dtype=field.real.dtype)
    return jnp.asarray(w_z, dtype=field.real.dtype)
