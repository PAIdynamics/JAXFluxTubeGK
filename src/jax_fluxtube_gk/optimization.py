"""Optimization-facing wrappers for fixed-topology linear flux-tube solves."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import build_circular_geometry, build_s_alpha_geometry, k_perp_squared
from .objectives import LinearObjectiveValues, initial_value_growth_objectives
from .physics import AdiabaticElectronParams, default_adiabatic_electron_params
from .solver import build_linear_residual_precompute
from .targets import BenchmarkTarget, benchmark_target_cost, benchmark_target_residual
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
        """Coerce ``equilibrium_coefficients`` to a one-dimensional array."""

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
        """Validate geometry_model, n_steps, dt, and objective_kind choices."""

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
    softmax_temperature: float = 0.01
    branch_gap_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        """Validate aggregation choice, selected_ky requirements, and weight signs."""

        if self.growth_aggregation not in ("selected", "max", "softmax"):
            raise ValueError("growth_aggregation must be 'selected', 'max', or 'softmax'")
        if self.growth_aggregation == "selected" and self.selected_ky is None:
            raise ValueError("selected growth aggregation requires selected_ky")
        if self.selected_ky is not None and self.selected_ky < 0:
            raise ValueError("selected_ky must be nonnegative")
        if self.frequency_weight != 0.0 and self.selected_ky is None:
            raise ValueError("a frequency objective requires selected_ky")
        if self.softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive")
        if self.branch_gap_tolerance < 0.0:
            raise ValueError("branch_gap_tolerance must be nonnegative")
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
    growth_branch_gap: object
    near_degenerate_branch: object
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
        "growth_branch_gap",
        "near_degenerate_branch",
        "surface_result",
    )


@dataclass(frozen=True)
class DesignGradientAudit:
    """Autodiff/finite-difference agreement for one scalar design parameter."""

    autodiff_gradient: float
    finite_difference_gradient: float
    absolute_error: float
    relative_error: float
    passed: bool
    step: float
    branch_gap: float | None
    near_degenerate_branch: bool


@dataclass(frozen=True)
class RobustAggregationSpec:
    """Fixed policy for reducing several surface/field-line objectives."""

    method: str = "softmax"
    softmax_temperature: float = 0.05

    def __post_init__(self) -> None:
        """Validate the aggregation method and softmax_temperature."""

        if self.method not in ("weighted_mean", "worst_case", "softmax"):
            raise ValueError(
                "aggregation method must be 'weighted_mean', 'worst_case', or 'softmax'"
            )
        if self.softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RobustAggregationResult(_PyTreeDataclass):
    """Differentiable reduction and diagnostics over fixed design samples."""

    scalar_objective: object
    weighted_mean: object
    worst_case: object
    softmax_objective: object
    sample_objectives: object
    normalized_weights: object

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "scalar_objective",
        "weighted_mean",
        "worst_case",
        "softmax_objective",
        "sample_objectives",
        "normalized_weights",
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
    electrons = electron_params or default_adiabatic_electron_params(species)
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
    if spec.growth_aggregation == "selected":
        growth = selected_growth
    elif spec.growth_aggregation == "softmax":
        temperature = jnp.asarray(spec.softmax_temperature, dtype=values.growth_rate.dtype)
        growth = temperature * jax.scipy.special.logsumexp(values.growth_rate / temperature)
    else:
        growth = values.max_growth_rate
    branch_gap = _growth_branch_gap(values.growth_rate)
    frequency_penalty = (selected_frequency - spec.frequency_target) ** 2
    scalar = spec.growth_weight * growth
    if spec.frequency_weight != 0.0:
        scalar = scalar + spec.frequency_weight * frequency_penalty
    if spec.mode_structure_weight != 0.0:
        scalar = scalar + spec.mode_structure_weight * values.mode_structure_penalty
    if spec.quasilinear_weight != 0.0:
        scalar = scalar + spec.quasilinear_weight * values.quasilinear_proxy
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
        growth_branch_gap=branch_gap,
        near_degenerate_branch=branch_gap <= spec.branch_gap_tolerance,
        surface_result=surface,
    )


def audit_design_gradient(
    objective_fn,
    parameter: float,
    *,
    step: float = 1.0e-5,
    relative_tolerance: float = 1.0e-3,
    absolute_tolerance: float = 1.0e-6,
    branch_gap: float | None = None,
    branch_gap_tolerance: float = 1.0e-3,
) -> DesignGradientAudit:
    """Compare reverse-mode AD with a central difference at a scalar parameter."""

    if step <= 0.0:
        raise ValueError("gradient-audit step must be positive")
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("gradient-audit tolerances must be nonnegative")
    autodiff = float(jax.grad(objective_fn)(parameter))
    finite_difference = float(
        (objective_fn(parameter + step) - objective_fn(parameter - step)) / (2.0 * step)
    )
    absolute_error = abs(autodiff - finite_difference)
    scale = max(abs(autodiff), abs(finite_difference), absolute_tolerance)
    relative_error = absolute_error / scale
    passed = absolute_error <= absolute_tolerance + relative_tolerance * scale
    return DesignGradientAudit(
        autodiff_gradient=autodiff,
        finite_difference_gradient=finite_difference,
        absolute_error=absolute_error,
        relative_error=relative_error,
        passed=passed,
        step=float(step),
        branch_gap=None if branch_gap is None else float(branch_gap),
        near_degenerate_branch=(
            False if branch_gap is None else abs(float(branch_gap)) <= branch_gap_tolerance
        ),
    )


def aggregate_design_objectives(
    sample_objectives,
    spec: RobustAggregationSpec | None = None,
    *,
    sample_weights=None,
) -> RobustAggregationResult:
    """Reduce fixed surface/field-line samples without breaking JAX gradients.

    The sample axis may have any shape and is flattened in a deterministic
    row-major order. Weights are static optimization controls rather than
    differentiated parameters.
    """

    spec = spec or RobustAggregationSpec()
    values = jnp.ravel(jnp.asarray(sample_objectives))
    if values.size == 0:
        raise ValueError("at least one design sample is required")
    if sample_weights is None:
        weights = jnp.full(values.shape, 1.0 / values.size, dtype=values.dtype)
    else:
        raw_weights = np.ravel(np.asarray(sample_weights, dtype=float))
        if raw_weights.shape != values.shape:
            raise ValueError("sample_weights must have the same size as sample_objectives")
        if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights < 0.0):
            raise ValueError("sample_weights must be finite and nonnegative")
        total_weight = float(np.sum(raw_weights))
        if total_weight <= 0.0:
            raise ValueError("sample_weights must contain a positive weight")
        weights = jnp.asarray(raw_weights / total_weight, dtype=values.dtype)
    weighted_mean = jnp.sum(weights * values)
    worst_case = jnp.max(values)
    temperature = jnp.asarray(spec.softmax_temperature, dtype=values.dtype)
    log_weights = jnp.where(weights > 0.0, jnp.log(weights), -jnp.inf)
    softmax_objective = temperature * jax.scipy.special.logsumexp(
        values / temperature + log_weights
    )
    if spec.method == "weighted_mean":
        scalar = weighted_mean
    elif spec.method == "worst_case":
        scalar = worst_case
    else:
        scalar = softmax_objective
    return RobustAggregationResult(
        scalar_objective=scalar,
        weighted_mean=weighted_mean,
        worst_case=worst_case,
        softmax_objective=softmax_objective,
        sample_objectives=values,
        normalized_weights=weights,
    )


def robust_scan_objective(
    scan: OptimizationScanResult,
    spec: RobustAggregationSpec | None = None,
    *,
    sample_weights=None,
) -> RobustAggregationResult:
    """Aggregate a fixed rho/alpha/ky scan as one robust design objective."""

    expected_shape = (
        scan.rho_values.shape[0],
        scan.alpha_values.shape[0],
        scan.ky_indices.shape[0],
    )
    if scan.objectives.shape != expected_shape:
        raise ValueError(
            f"scan objectives must have rho/alpha/ky shape {expected_shape}; "
            f"got {scan.objectives.shape}"
        )
    if sample_weights is not None and np.asarray(sample_weights).shape != expected_shape:
        raise ValueError(f"scan weights must have rho/alpha/ky shape {expected_shape}")
    return aggregate_design_objectives(
        scan.objectives,
        spec,
        sample_weights=sample_weights,
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
    """Build geometry from knobs, or validate and pass through a supplied geometry."""

    if geometry is None:
        return build_optimization_geometry(parallel_grid, knobs, config)
    _validate_objective_geometry(parallel_grid, geometry)
    return geometry


def _validate_objective_geometry(parallel_grid: ParallelGrid, geometry):
    """Check that a supplied geometry's field arrays match the parallel grid shape."""

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
    """Select or combine growth/quasilinear/mode-structure terms per objective_kind."""

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
    """Resolve a benchmark target quantity name (with aliases) to its observed value."""

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
    """Apply a toy algebraic beta/pressure-gradient/coefficient modulation to geometry."""

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
    """Build a SpeciesParams from base charge/mass/kinetic and knob profile values."""

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
    """Build GeometryScalarParams from knob q/shat/eps, deriving iota as 1/q."""

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
    """Return the cosine-series modulation sum_n coefficients[n] * cos(n * theta)."""

    if coefficients.shape[0] == 0:
        return jnp.zeros_like(theta)
    modes = jnp.arange(1, coefficients.shape[0] + 1, dtype=theta.dtype)
    phase = modes[:, None] * theta[None, :]
    return jnp.sum(coefficients[:, None] * jnp.cos(phase), axis=0)


def _growth_branch_gap(growth_rates):
    """Return the gap between the largest and second-largest growth rate."""

    growth_rates = jnp.asarray(growth_rates)
    if growth_rates.shape[0] < 2:
        return jnp.asarray(jnp.inf, dtype=growth_rates.dtype)
    ordered = jnp.sort(growth_rates)
    return ordered[-1] - ordered[-2]
