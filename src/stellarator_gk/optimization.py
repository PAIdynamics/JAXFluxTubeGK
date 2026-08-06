"""Optimization-facing wrappers for fixed-topology linear flux-tube solves."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import jax
import jax.numpy as jnp

from .benchmarks import BenchmarkTarget, benchmark_target_cost, benchmark_target_residual
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

_ANALYTIC_GEOMETRY_MODELS = ("circular", "circ", "s-alpha")
_IMPORTED_GEOMETRY_MODELS = ("precomputed", "desc", "desc-precomputed")
_GEOMETRY_MODELS = _ANALYTIC_GEOMETRY_MODELS + _IMPORTED_GEOMETRY_MODELS


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
        if self.geometry_model not in _GEOMETRY_MODELS:
            raise ValueError(
                "geometry_model must be one of "
                "'circular', 'circ', 's-alpha', 'precomputed', 'desc', "
                "or 'desc-precomputed'"
            )
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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BenchmarkOptimizationResult(_PyTreeDataclass):
    """Single-surface objective recast as a least-squares benchmark-target error."""

    scalar_objective: object
    target_residual: object
    observed_value: object
    surface_result: SingleSurfaceOptimizationResult

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "scalar_objective",
        "target_residual",
        "observed_value",
        "surface_result",
    )


@dataclass(frozen=True)
class DesignObjectiveSpec:
    """Stable composition contract for a differentiable linear design objective."""

    selected_ky: int | None = None
    growth_aggregation: str = "selected"
    growth_weight: float = 1.0
    frequency_weight: float = 0.0
    mode_structure_weight: float = 0.0
    quasilinear_weight: float = 0.0
    frequency_target: float = 0.0

    def __post_init__(self) -> None:
        if self.growth_aggregation not in ("selected", "max"):
            raise ValueError("growth_aggregation must be 'selected' or 'max'")
        if self.growth_aggregation == "selected" and self.selected_ky is None:
            raise ValueError("selected growth aggregation requires selected_ky")
        if self.selected_ky is not None and self.selected_ky < 0:
            raise ValueError("selected_ky must be nonnegative")
        if self.frequency_weight != 0.0 and self.selected_ky is None:
            raise ValueError("a frequency objective requires selected_ky")
        for name in (
            "growth_weight",
            "frequency_weight",
            "mode_structure_weight",
            "quasilinear_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DesignObjectiveResult(_PyTreeDataclass):
    """Scalar design loss together with all named physical components."""

    scalar_objective: object
    growth_objective: object
    selected_growth_rate: object
    selected_frequency: object
    frequency_penalty: object
    max_growth_rate: object
    quasilinear_proxy: object
    mode_structure_penalty: object
    mode_structure: object
    surface_result: SingleSurfaceOptimizationResult

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "scalar_objective",
        "growth_objective",
        "selected_growth_rate",
        "selected_frequency",
        "frequency_penalty",
        "max_growth_rate",
        "quasilinear_proxy",
        "mode_structure_penalty",
        "mode_structure",
        "surface_result",
    )


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
    if config.geometry_model in _IMPORTED_GEOMETRY_MODELS:
        raise ValueError(
            "imported geometry models require a supplied geometry object; "
            "build one with build_desc_geometry_from_arrays or "
            "map_physical_to_internal_geometry"
        )
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
    geometry=None,
    target_mode_structure=None,
    phase_align_mode_structure: bool = False,
) -> SingleSurfaceOptimizationResult:
    """Evaluate a differentiable single-surface linear objective."""

    config = config or SingleSurfaceOptimizationConfig()
    species = build_optimization_species(knobs, base_species)
    geometry = _objective_geometry(parallel_grid, knobs, config, geometry)
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
        target_mode_structure=target_mode_structure,
        w_z=geometry.w_z,
        connectivity=connectivity,
        softplus_temperature=config.softplus_temperature,
        store_history=config.store_history,
        phase_align_mode_structure=phase_align_mode_structure,
    )
    scalar = _scalar_from_objective_values(values, config)
    return SingleSurfaceOptimizationResult(
        scalar_objective=scalar,
        values=values,
        geometry=geometry,
        species=species,
        kperp_squared=kperp2,
    )


def design_objective(
    knobs: OptimizationKnobs,
    velocity_grid: VelocityGrid,
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    initial_state,
    spec: DesignObjectiveSpec,
    *,
    base_species: SpeciesParams | None = None,
    electron_params: AdiabaticElectronParams | None = None,
    connectivity: ModeConnectivity | None = None,
    config: SingleSurfaceOptimizationConfig | None = None,
    geometry=None,
    target_mode_structure=None,
) -> DesignObjectiveResult:
    """Evaluate the stable growth/frequency/shape/quasilinear design contract."""

    base_config = config or SingleSurfaceOptimizationConfig()
    selected_ky = spec.selected_ky
    objective_config = replace(base_config, selected_ky=selected_ky)
    surface = single_surface_objective(
        knobs,
        velocity_grid,
        parallel_grid,
        fourier_grid,
        initial_state,
        base_species=base_species,
        electron_params=electron_params,
        connectivity=connectivity,
        config=objective_config,
        geometry=geometry,
        target_mode_structure=target_mode_structure,
        phase_align_mode_structure=True,
    )
    values = surface.values
    if selected_ky is None:
        selected_index = jnp.argmax(values.growth_rate)
    else:
        if selected_ky >= fourier_grid.ky.shape[0]:
            raise ValueError(
                f"selected_ky {selected_ky} is outside the {fourier_grid.ky.shape[0]}-mode grid"
            )
        selected_index = selected_ky
    selected_growth = values.growth_rate[selected_index]
    selected_frequency = values.frequency[selected_index]
    growth = (
        selected_growth if spec.growth_aggregation == "selected" else values.max_growth_rate
    )
    frequency_penalty = (selected_frequency - spec.frequency_target) ** 2
    scalar = (
        spec.growth_weight * growth
        + spec.frequency_weight * frequency_penalty
        + spec.mode_structure_weight * values.mode_structure_penalty
        + spec.quasilinear_weight * values.quasilinear_proxy
    )
    return DesignObjectiveResult(
        scalar_objective=scalar,
        growth_objective=growth,
        selected_growth_rate=selected_growth,
        selected_frequency=selected_frequency,
        frequency_penalty=frequency_penalty,
        max_growth_rate=values.max_growth_rate,
        quasilinear_proxy=values.quasilinear_proxy,
        mode_structure_penalty=values.mode_structure_penalty,
        mode_structure=values.mode_structure,
        surface_result=surface,
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
    geometry=None,
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
                    geometry=geometry,
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


def single_surface_benchmark_objective(
    knobs: OptimizationKnobs,
    velocity_grid: VelocityGrid,
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    initial_state,
    target: BenchmarkTarget,
    *,
    base_species: SpeciesParams | None = None,
    electron_params: AdiabaticElectronParams | None = None,
    connectivity: ModeConnectivity | None = None,
    config: SingleSurfaceOptimizationConfig | None = None,
    geometry=None,
    normalize_by_tolerance: bool = True,
) -> BenchmarkOptimizationResult:
    """Evaluate a reduced objective as a benchmark-target least-squares error."""

    result = single_surface_objective(
        knobs,
        velocity_grid,
        parallel_grid,
        fourier_grid,
        initial_state,
        base_species=base_species,
        electron_params=electron_params,
        connectivity=connectivity,
        config=config,
        geometry=geometry,
    )
    observed = _benchmark_observed_value(result, target.quantity)
    residual = benchmark_target_residual(
        observed,
        target,
        normalize_by_tolerance=normalize_by_tolerance,
    )
    cost = benchmark_target_cost(
        observed,
        target,
        normalize_by_tolerance=normalize_by_tolerance,
    )
    return BenchmarkOptimizationResult(
        scalar_objective=cost,
        target_residual=residual,
        observed_value=observed,
        surface_result=result,
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


def _objective_geometry(
    parallel_grid: ParallelGrid,
    knobs: OptimizationKnobs,
    config: SingleSurfaceOptimizationConfig,
    geometry,
):
    if geometry is None:
        return build_optimization_geometry(parallel_grid, knobs, config)
    _validate_objective_geometry(parallel_grid, geometry)
    return geometry


def _validate_objective_geometry(parallel_grid: ParallelGrid, geometry):
    target_shape = parallel_grid.z.shape
    for name in ("w_z", "B", "F", "G", "E_y", "D_x", "D_y", "g_xx", "g_xy", "g_yy"):
        array = jnp.asarray(getattr(geometry, name))
        if array.shape != target_shape:
            raise ValueError(
                f"supplied geometry.{name} must have shape {target_shape}; got {array.shape}"
            )


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


def _benchmark_observed_value(result: SingleSurfaceOptimizationResult, quantity: str):
    aliases = {
        "growth_rate": "selected_growth_rate",
        "selected_growth": "selected_growth_rate",
        "max_growth": "max_growth_rate",
        "quasilinear": "quasilinear_proxy",
        "mode_structure": "mode_structure_penalty",
    }
    quantity = aliases.get(quantity, quantity)
    if quantity == "scalar_objective":
        return result.scalar_objective
    if hasattr(result.values, quantity):
        return getattr(result.values, quantity)
    raise ValueError(f"unsupported benchmark target quantity {quantity!r}")


def _apply_toy_equilibrium_coefficients(geometry, knobs: OptimizationKnobs):
    coefficients = jnp.asarray(knobs.equilibrium_coefficients, dtype=geometry.B.dtype)
    modulation = _coefficient_modulation(geometry.theta + knobs.alpha, coefficients)
    beta_scale = 1.0 + 0.01 * knobs.beta
    pressure_scale = 1.0 + 0.005 * knobs.pressure_gradient * (1.0 + knobs.rho)
    magnetic_scale = beta_scale * (1.0 + 0.02 * modulation)
    metric_scale = pressure_scale * (1.0 + 0.01 * modulation)
    drift_scale = 1.0 + 0.01 * knobs.beta + 0.005 * knobs.pressure_gradient
    updates = {
        "B": geometry.B * magnetic_scale,
        "D_x": geometry.D_x * drift_scale,
        "D_y": geometry.D_y * drift_scale,
        "g_xy": geometry.g_xy * metric_scale,
        "g_yy": geometry.g_yy * metric_scale,
    }
    for name in ("D_x_gradB", "D_y_gradB", "D_x_curvature", "D_y_curvature"):
        value = getattr(geometry, name, None)
        if value is not None:
            updates[name] = value * drift_scale
    return replace(geometry, **updates)


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
