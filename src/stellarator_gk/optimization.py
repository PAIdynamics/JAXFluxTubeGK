"""Optimization-facing wrappers for fixed-topology linear flux-tube solves."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import jax
import jax.numpy as jnp

from .geometry import build_circular_geometry, build_s_alpha_geometry, k_perp_squared
from .objectives import LinearObjectiveValues, initial_value_growth_objectives
from .physics import AdiabaticElectronParams, default_adiabatic_electron_params
from .solver import build_linear_residual_precompute
from .types import (
    FourierGrid,
    GeometryScalarParams,
    ModeConnectivity,
    ParallelGrid,
    SpeciesParams,
    VelocityGrid,
    _PyTreeDataclass,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OptimizationKnobs(_PyTreeDataclass):
    """Differentiable local knobs for a single flux-tube optimization solve."""

    density: float = 1.0
    temperature: float = 1.0
    density_gradient: float = 0.8
    temperature_gradient: float = 2.5
    q: float = 1.4
    shat: float = 0.8
    eps: float = 0.18
    rho: float = 0.5
    alpha: float = 0.0
    beta: float = 0.0
    pressure_gradient: float = 0.0
    equilibrium_coefficients: object = ()

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "density",
        "temperature",
        "density_gradient",
        "temperature_gradient",
        "q",
        "shat",
        "eps",
        "rho",
        "alpha",
        "beta",
        "pressure_gradient",
        "equilibrium_coefficients",
    )

    def __post_init__(self):
        coefficients = jnp.asarray(self.equilibrium_coefficients, dtype=jnp.float64)
        if coefficients.ndim != 1:
            raise ValueError("equilibrium_coefficients must be one-dimensional")
        object.__setattr__(self, "equilibrium_coefficients", coefficients)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SingleSurfaceOptimizationConfig(_PyTreeDataclass):
    """Static controls for a single-surface/single-alpha objective evaluation."""

    geometry_model: str = "circular"
    dt: float = 0.01
    n_steps: int = 2
    selected_ky: int | None = None
    objective_kind: str = "growth_plus_quasilinear"
    quasilinear_weight: float = 0.01
    mode_structure_weight: float = 0.0
    softplus_temperature: float | None = 0.1
    store_history: bool = False

    _static_fields: ClassVar[tuple[str, ...]] = (
        "geometry_model",
        "dt",
        "n_steps",
        "selected_ky",
        "objective_kind",
        "quasilinear_weight",
        "mode_structure_weight",
        "softplus_temperature",
        "store_history",
    )

    def __post_init__(self):
        if self.geometry_model not in ("circular", "circ", "s-alpha"):
            raise ValueError("geometry_model must be 'circular', 'circ', or 's-alpha'")
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.objective_kind not in (
            "selected_growth",
            "max_growth",
            "quasilinear_proxy",
            "growth_plus_quasilinear",
        ):
            raise ValueError("unsupported objective_kind")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SingleSurfaceOptimizationResult(_PyTreeDataclass):
    """Result of one differentiable single-surface objective evaluation."""

    scalar_objective: object
    values: LinearObjectiveValues
    geometry: object
    species: SpeciesParams
    kperp_squared: object

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "scalar_objective",
        "values",
        "geometry",
        "species",
        "kperp_squared",
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OptimizationScanResult(_PyTreeDataclass):
    """Static-grid scan over surface labels, field-line labels, and ky choices."""

    objectives: object
    rho_values: object
    alpha_values: object
    ky_indices: object

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "objectives",
        "rho_values",
        "alpha_values",
        "ky_indices",
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ToyOptimizationStep(_PyTreeDataclass):
    """One gradient-descent step for reduced optimization examples."""

    value: object
    gradient: OptimizationKnobs
    updated_knobs: OptimizationKnobs

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("value", "gradient", "updated_knobs")


def build_optimization_species(
    knobs: OptimizationKnobs,
    base_species: SpeciesParams | None = None,
) -> SpeciesParams:
    """Map differentiable profile knobs to a kinetic species parameter object."""

    species = base_species or SpeciesParams(
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        density_gradient=0.0,
        temperature_gradient=0.0,
    )
    return _species_from_optimization_knobs(species, knobs)


def build_optimization_geometry(
    parallel_grid: ParallelGrid,
    knobs: OptimizationKnobs,
    config: SingleSurfaceOptimizationConfig | None = None,
):
    """Build differentiable analytic geometry and apply toy equilibrium knobs.

    The coefficient modulation is intentionally low-amplitude and algebraic. It
    exercises the optimization plumbing until DESC or Boozer-file adapters
    provide physically complete precomputed geometry arrays.
    """

    config = config or SingleSurfaceOptimizationConfig()
    params = _geometry_params_from_optimization_knobs(knobs)
    if config.geometry_model == "s-alpha":
        geometry = build_s_alpha_geometry(parallel_grid, params)
    else:
        geometry = build_circular_geometry(parallel_grid, params)
    return _apply_toy_equilibrium_coefficients(geometry, knobs)


def single_surface_objective(
    knobs: OptimizationKnobs,
    velocity_grid: VelocityGrid,
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    initial_state,
    *,
    base_species: SpeciesParams | None = None,
    electron_params: AdiabaticElectronParams | None = None,
    connectivity: ModeConnectivity | None = None,
    config: SingleSurfaceOptimizationConfig | None = None,
) -> SingleSurfaceOptimizationResult:
    """Evaluate a differentiable single-surface linear objective."""

    config = config or SingleSurfaceOptimizationConfig()
    species = build_optimization_species(knobs, base_species)
    geometry = build_optimization_geometry(parallel_grid, knobs, config)
    electrons = electron_params or default_adiabatic_electron_params()
    precompute = build_linear_residual_precompute(
        velocity_grid,
        parallel_grid,
        fourier_grid,
        geometry,
        species,
        electron_params=electrons,
    )
    kperp2 = k_perp_squared(geometry, fourier_grid)
    values = initial_value_growth_objectives(
        initial_state,
        precompute,
        config.dt,
        config.n_steps,
        kperp2,
        selected_ky=config.selected_ky,
        w_z=geometry.w_z,
        connectivity=connectivity,
        softplus_temperature=config.softplus_temperature,
        store_history=config.store_history,
    )
    scalar = _scalar_from_objective_values(values, config)
    return SingleSurfaceOptimizationResult(
        scalar_objective=scalar,
        values=values,
        geometry=geometry,
        species=species,
        kperp_squared=kperp2,
    )


def scan_single_surface_objective(
    knobs: OptimizationKnobs,
    velocity_grid: VelocityGrid,
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    initial_state,
    *,
    rho_values,
    alpha_values,
    ky_indices,
    base_species: SpeciesParams | None = None,
    electron_params: AdiabaticElectronParams | None = None,
    connectivity: ModeConnectivity | None = None,
    config: SingleSurfaceOptimizationConfig | None = None,
) -> OptimizationScanResult:
    """Evaluate the scalar objective on a static ``rho``/``alpha``/``ky`` scan."""

    base_config = config or SingleSurfaceOptimizationConfig()
    rho_values = jnp.asarray(rho_values)
    alpha_values = jnp.asarray(alpha_values)
    ky_indices_array = jnp.asarray(ky_indices, dtype=jnp.int32)
    objectives = []
    for rho in rho_values:
        alpha_objectives = []
        for alpha in alpha_values:
            ky_objectives = []
            scan_knobs = replace(knobs, rho=rho, alpha=alpha)
            for ky_index in tuple(int(value) for value in ky_indices_array):
                scan_config = replace(base_config, selected_ky=ky_index)
                result = single_surface_objective(
                    scan_knobs,
                    velocity_grid,
                    parallel_grid,
                    fourier_grid,
                    initial_state,
                    base_species=base_species,
                    electron_params=electron_params,
                    connectivity=connectivity,
                    config=scan_config,
                )
                ky_objectives.append(result.scalar_objective)
            alpha_objectives.append(jnp.stack(ky_objectives))
        objectives.append(jnp.stack(alpha_objectives))
    return OptimizationScanResult(
        objectives=jnp.stack(objectives),
        rho_values=rho_values,
        alpha_values=alpha_values,
        ky_indices=ky_indices_array,
    )


def toy_gradient_descent_step(
    objective_fn,
    knobs: OptimizationKnobs,
    *,
    learning_rate: float = 1.0e-2,
) -> ToyOptimizationStep:
    """Return one generic gradient-descent step for a scalar knob objective."""

    value, gradient = jax.value_and_grad(objective_fn)(knobs)
    updated = jax.tree_util.tree_map(
        lambda parameter, grad: parameter - learning_rate * grad,
        knobs,
        gradient,
    )
    return ToyOptimizationStep(value=value, gradient=gradient, updated_knobs=updated)


def _scalar_from_objective_values(
    values: LinearObjectiveValues,
    config: SingleSurfaceOptimizationConfig,
):
    if config.objective_kind == "selected_growth":
        return values.selected_growth_rate
    if config.objective_kind == "max_growth":
        return values.max_growth_rate
    if config.objective_kind == "quasilinear_proxy":
        return values.quasilinear_proxy
    return (
        values.selected_growth_rate
        + config.quasilinear_weight * values.quasilinear_proxy
        + config.mode_structure_weight * values.mode_structure_penalty
    )


def _apply_toy_equilibrium_coefficients(geometry, knobs: OptimizationKnobs):
    coefficients = jnp.asarray(knobs.equilibrium_coefficients, dtype=geometry.B.dtype)
    modulation = _coefficient_modulation(geometry.theta + knobs.alpha, coefficients)
    beta_scale = 1.0 + 0.01 * knobs.beta
    pressure_scale = 1.0 + 0.005 * knobs.pressure_gradient * (1.0 + knobs.rho)
    magnetic_scale = beta_scale * (1.0 + 0.02 * modulation)
    metric_scale = pressure_scale * (1.0 + 0.01 * modulation)
    drift_scale = 1.0 + 0.01 * knobs.beta + 0.005 * knobs.pressure_gradient
    return replace(
        geometry,
        B=geometry.B * magnetic_scale,
        D_x=geometry.D_x * drift_scale,
        D_y=geometry.D_y * drift_scale,
        g_xy=geometry.g_xy * metric_scale,
        g_yy=geometry.g_yy * metric_scale,
    )


def _species_from_optimization_knobs(
    base_species: SpeciesParams,
    knobs: OptimizationKnobs,
) -> SpeciesParams:
    species = SpeciesParams.__new__(SpeciesParams)
    for name, value in (
        ("charge", base_species.charge),
        ("mass", base_species.mass),
        ("density", knobs.density),
        ("temperature", knobs.temperature),
        ("density_gradient", knobs.density_gradient),
        ("temperature_gradient", knobs.temperature_gradient),
        ("kinetic", base_species.kinetic),
    ):
        object.__setattr__(species, name, value)
    return species


def _geometry_params_from_optimization_knobs(knobs: OptimizationKnobs) -> GeometryScalarParams:
    params = GeometryScalarParams.__new__(GeometryScalarParams)
    for name, value in (
        ("q", knobs.q),
        ("shat", knobs.shat),
        ("eps", knobs.eps),
        ("iota", 1.0 / knobs.q),
    ):
        object.__setattr__(params, name, value)
    return params


def _coefficient_modulation(theta, coefficients):
    if coefficients.shape[0] == 0:
        return jnp.zeros_like(theta)
    modes = jnp.arange(1, coefficients.shape[0] + 1, dtype=theta.dtype)
    phase = modes[:, None] * theta[None, :]
    return jnp.sum(coefficients[:, None] * jnp.cos(phase), axis=0)
