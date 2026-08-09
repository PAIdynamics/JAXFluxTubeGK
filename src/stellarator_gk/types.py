"""Immutable public data containers for grids, species, and solver controls.

The static/dynamic split is explicit so JAX transforms see only continuous
arrays and differentiable scalar parameters as leaves. Topology, enum choices,
sizes, and validation flags stay in auxiliary data.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from typing import ClassVar

import jax


class DerivativeBackend(StrEnum):
    """Derivative backend names used by grid builders."""

    CHEBYSHEV = "chebyshev"
    FOURIER = "fourier"
    FINITE_DIFFERENCE = "finite_difference"


class ParallelTopology(StrEnum):
    """Parallel field-line topology used by the spectral grid builder."""

    PERIODIC = "periodic"
    OPEN = "open"


class _PyTreeDataclass:
    """Small helper for frozen dataclasses registered as JAX PyTrees."""

    _dynamic_fields: ClassVar[tuple[str, ...]] = ()
    _static_fields: ClassVar[tuple[str, ...]] = ()

    def tree_flatten(self):
        leaves = tuple(getattr(self, name) for name in self._dynamic_fields)
        aux = tuple((name, getattr(self, name)) for name in self._static_field_names())
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        values = {name: value for name, value in aux}
        values.update(zip(cls._dynamic_fields, leaves, strict=True))
        obj = cls.__new__(cls)
        for name, value in values.items():
            object.__setattr__(obj, name, value)
        return obj

    @classmethod
    def _static_field_names(cls) -> tuple[str, ...]:
        explicit_static = set(cls._static_fields)
        dynamic = set(cls._dynamic_fields)
        declared = tuple(field.name for field in fields(cls))
        return tuple(name for name in declared if name in explicit_static or name not in dynamic)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpeciesParams(_PyTreeDataclass):
    """Dimensionless species/profile parameters."""

    charge: float
    mass: float
    density: float
    temperature: float
    density_gradient: float
    temperature_gradient: float
    kinetic: bool = True

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "charge",
        "mass",
        "density",
        "temperature",
        "density_gradient",
        "temperature_gradient",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("kinetic",)

    def __post_init__(self):
        if self.charge == 0:
            raise ValueError("charge must be nonzero")
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SolverControls(_PyTreeDataclass):
    """Solver-wide controls that are static under JAX tracing."""

    dtype: str = "float64"
    derivative_backend: str = DerivativeBackend.CHEBYSHEV.value
    require_x64: bool = True
    validate_inputs: bool = True

    _static_fields: ClassVar[tuple[str, ...]] = (
        "dtype",
        "derivative_backend",
        "require_x64",
        "validate_inputs",
    )

    def __post_init__(self):
        backend = DerivativeBackend(self.derivative_backend)
        object.__setattr__(self, "derivative_backend", backend.value)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GeometryScalarParams(_PyTreeDataclass):
    """Continuous scalar geometry parameters used by analytic fixtures."""

    q: float = 1.0
    shat: float = 0.0
    eps: float = 0.1
    iota: float = 1.0

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("q", "shat", "eps", "iota")

    def __post_init__(self):
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.q <= 0:
            raise ValueError("q must be positive")
        if self.iota <= 0:
            raise ValueError("iota must be positive")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VelocityGridSpec(_PyTreeDataclass):
    """Configuration for spectral velocity-space collocation."""

    n_vpar: int
    n_mu: int
    vpar_max: float
    mu_max: float
    backend: str = DerivativeBackend.CHEBYSHEV.value
    dtype: str = "float64"

    _static_fields: ClassVar[tuple[str, ...]] = (
        "n_vpar",
        "n_mu",
        "vpar_max",
        "mu_max",
        "backend",
        "dtype",
    )

    def __post_init__(self):
        backend = DerivativeBackend(self.backend)
        object.__setattr__(self, "backend", backend.value)
        if backend not in (DerivativeBackend.CHEBYSHEV, DerivativeBackend.FINITE_DIFFERENCE):
            raise ValueError("VelocityGridSpec supports chebyshev or finite_difference")
        if self.n_vpar < 2:
            raise ValueError("n_vpar must be at least 2")
        if self.n_mu < 2:
            raise ValueError("n_mu must be at least 2")
        if self.vpar_max <= 0:
            raise ValueError("vpar_max must be positive")
        if self.mu_max <= 0:
            raise ValueError("mu_max must be positive")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ParallelGridSpec(_PyTreeDataclass):
    """Configuration for the parallel field-line grid."""

    n_z: int
    z_min: float
    z_max: float
    topology: str = ParallelTopology.PERIODIC.value
    backend: str = "auto"
    dtype: str = "float64"

    _static_fields: ClassVar[tuple[str, ...]] = (
        "n_z",
        "z_min",
        "z_max",
        "topology",
        "backend",
        "dtype",
    )

    def __post_init__(self):
        topology = ParallelTopology(self.topology)
        object.__setattr__(self, "topology", topology.value)
        if self.n_z < 2:
            raise ValueError("n_z must be at least 2")
        if self.z_max <= self.z_min:
            raise ValueError("z_max must be greater than z_min")
        if self.backend not in ("auto", DerivativeBackend.FOURIER.value, DerivativeBackend.CHEBYSHEV.value):
            raise ValueError("parallel backend must be auto, fourier, or chebyshev")
        if self.backend == "auto":
            backend = (
                DerivativeBackend.FOURIER.value
                if topology is ParallelTopology.PERIODIC
                else DerivativeBackend.CHEBYSHEV.value
            )
            object.__setattr__(self, "backend", backend)
        if topology is ParallelTopology.PERIODIC and self.backend != DerivativeBackend.FOURIER.value:
            raise ValueError("periodic parallel grids use the fourier backend")
        if topology is ParallelTopology.OPEN and self.backend != DerivativeBackend.CHEBYSHEV.value:
            raise ValueError("open parallel grids use the chebyshev backend")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FourierGridSpec(_PyTreeDataclass):
    """Configuration for perpendicular Fourier modes."""

    n_kx: int
    n_ky: int
    kx_max: float
    ky_max: float | None = None
    ky_values: tuple[float, ...] | None = None
    ikxspace: int = 5
    q: float | None = None
    shat: float | None = None
    eps: float | None = None
    use_gkw_shear_spacing: bool = False
    dtype: str = "float64"

    _static_fields: ClassVar[tuple[str, ...]] = (
        "n_kx",
        "n_ky",
        "kx_max",
        "ky_max",
        "ky_values",
        "ikxspace",
        "q",
        "shat",
        "eps",
        "use_gkw_shear_spacing",
        "dtype",
    )

    def __post_init__(self):
        if self.n_kx < 1 or self.n_kx % 2 == 0:
            raise ValueError("n_kx must be a positive odd integer")
        if self.n_ky < 1:
            raise ValueError("n_ky must be at least 1")
        if self.kx_max < 0:
            raise ValueError("kx_max must be nonnegative")
        if self.ikxspace < 1:
            raise ValueError("ikxspace must be at least 1")
        if self.ky_values is not None:
            ky_values = tuple(float(value) for value in self.ky_values)
            if len(ky_values) != self.n_ky:
                raise ValueError("ky_values length must match n_ky")
            if any(value < 0 for value in ky_values):
                raise ValueError("ky_values must be nonnegative")
            if any(left > right for left, right in zip(ky_values, ky_values[1:], strict=False)):
                raise ValueError("ky_values must be nondecreasing")
            object.__setattr__(self, "ky_values", ky_values)
        elif self.ky_max is None:
            raise ValueError("ky_max is required when ky_values is absent")
        elif self.ky_max < 0:
            raise ValueError("ky_max must be nonnegative")
        if self.use_gkw_shear_spacing and None in (self.q, self.shat, self.eps):
            raise ValueError("q, shat, and eps are required for GKW shear spacing")
        if self.use_gkw_shear_spacing and self.eps == 0:
            raise ValueError("eps must be nonzero for GKW shear spacing")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VelocityGrid(_PyTreeDataclass):
    """Spectral velocity grid and operators."""

    vpar: object
    mu: object
    w_vpar: object
    w_mu: object
    D_vpar: object
    D_mu: object
    vpar_modal_transform: object
    vpar_inverse_modal_transform: object
    mu_modal_transform: object
    mu_inverse_modal_transform: object
    backend: str = DerivativeBackend.CHEBYSHEV.value

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "vpar",
        "mu",
        "w_vpar",
        "w_mu",
        "D_vpar",
        "D_mu",
        "vpar_modal_transform",
        "vpar_inverse_modal_transform",
        "mu_modal_transform",
        "mu_inverse_modal_transform",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("backend",)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ParallelGrid(_PyTreeDataclass):
    """Parallel grid and derivative operators."""

    z: object
    w_z: object
    D_z: object
    modal_transform: object
    inverse_modal_transform: object
    backend: str
    topology: str

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "z",
        "w_z",
        "D_z",
        "modal_transform",
        "inverse_modal_transform",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("backend", "topology")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FourierGrid(_PyTreeDataclass):
    """Perpendicular Fourier mode grid."""

    kx: object
    ky: object
    parseval: object
    ixzero: int
    iyzero: int
    ikxspace: int

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("kx", "ky", "parseval")
    _static_fields: ClassVar[tuple[str, ...]] = ("ixzero", "iyzero", "ikxspace")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ModeConnectivity(_PyTreeDataclass):
    """Static mode labels and boundary connectivity maps."""

    mode_label: object
    ixplus: object
    ixminus: object
    kx_shift: object
    valid_shift: object
    ixzero: int
    iyzero: int
    max_shift: int

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "mode_label",
        "ixplus",
        "ixminus",
        "kx_shift",
        "valid_shift",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("ixzero", "iyzero", "max_shift")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FiniteDifferenceOperators(_PyTreeDataclass):
    """Uniform-grid finite-difference fallback operators."""

    D1: object
    D4: object
    n: int
    spacing: float
    periodic: bool

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("D1", "D4")
    _static_fields: ClassVar[tuple[str, ...]] = ("n", "spacing", "periodic")
