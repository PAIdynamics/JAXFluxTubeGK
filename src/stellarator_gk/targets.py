"""Stable scalar target contracts shared by optimization and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from .types import _PyTreeDataclass


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BenchmarkTarget(_PyTreeDataclass):
    """Scalar validation or optimization target from an external benchmark."""

    name: str
    quantity: str
    reference_value: object
    tolerance: float
    source: str
    metadata: tuple[tuple[str, object], ...] = ()

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("reference_value",)
    _static_fields: ClassVar[tuple[str, ...]] = (
        "name",
        "quantity",
        "tolerance",
        "source",
        "metadata",
    )

    def __post_init__(self):
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        object.__setattr__(
            self,
            "reference_value",
            jnp.asarray(self.reference_value, dtype=jnp.float64),
        )
        object.__setattr__(self, "metadata", tuple(self.metadata))


def benchmark_target_residual(
    value,
    target: BenchmarkTarget,
    *,
    normalize_by_tolerance: bool = True,
):
    """Return the signed residual between an observation and a scalar target."""
    residual = jnp.asarray(value) - target.reference_value
    if normalize_by_tolerance:
        residual = residual / jnp.asarray(target.tolerance, dtype=residual.dtype)
    return residual


def benchmark_target_cost(
    value,
    target: BenchmarkTarget,
    *,
    normalize_by_tolerance: bool = True,
):
    """Return ``0.5 * residual**2`` for a scalar target."""
    residual = benchmark_target_residual(
        value,
        target,
        normalize_by_tolerance=normalize_by_tolerance,
    )
    return 0.5 * residual**2
