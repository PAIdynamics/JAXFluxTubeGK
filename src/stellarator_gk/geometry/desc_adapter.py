"""Optional DESC equilibrium adapters for flux-tube geometry arrays."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np

from ..types import ParallelGrid
from .flux_tube import build_desc_geometry_from_arrays

DESC_GEOMETRY_COMPUTE_KEYS = (
    "|B|",
    "B",
    "B^zeta",
    "grad(|B|)",
    "grad(psi)",
    "grad(alpha)",
    "b",
    "kappa",
    "theta_PEST",
    "phi",
    "alpha",
)


def load_desc_equilibrium(
    path,
    *,
    file_format: str | None = None,
    index: int = -1,
    loader: Callable | None = None,
):
    """Load a DESC equilibrium or select one equilibrium from a saved family."""

    loader = loader or _import_desc_load()
    loaded = loader(str(Path(path)), file_format=file_format)
    return _select_desc_equilibrium(loaded, index=index)


def desc_geometry_arrays_from_data(
    data: dict,
    *,
    zeta,
    rho,
    alpha,
) -> dict[str, object]:
    """Map DESC ``eq.compute`` output to the physical flux-tube array contract."""

    zeta = _coerce_scalar_or_1d("zeta", zeta)
    shape = zeta.shape
    B = _require_1d(data, "|B|", shape)
    B_vector = _require_vector(data, "B", shape)
    grad_B = _require_vector(data, "grad(|B|)", shape)
    grad_psi = _require_vector(data, "grad(psi)", shape)
    grad_alpha = _require_vector(data, "grad(alpha)", shape)
    b = _require_vector(data, "b", shape, default=B_vector / B[:, None])
    kappa = _require_vector(data, "kappa", shape)

    B_cross_gradB = jnp.cross(B_vector, grad_B, axis=-1)
    b_cross_kappa = jnp.cross(b, kappa, axis=-1)
    return {
        "theta": _optional_1d(data, "theta_PEST", shape, default=zeta),
        "phi": _optional_1d(data, "phi", shape, default=zeta),
        "rho": _optional_1d(data, "rho", shape, default=rho),
        "alpha": _optional_1d(data, "alpha", shape, default=alpha),
        "B": B,
        "b_dot_grad_z": _require_1d(data, "B^zeta", shape) / B,
        "grad_psi_sq": _dot(grad_psi, grad_psi),
        "grad_alpha_sq": _dot(grad_alpha, grad_alpha),
        "grad_psi_dot_grad_alpha": _dot(grad_psi, grad_alpha),
        "B_cross_gradB_dot_grad_psi": _dot(B_cross_gradB, grad_psi),
        "B_cross_gradB_dot_grad_alpha": _dot(B_cross_gradB, grad_alpha),
        "b_cross_kappa_dot_grad_psi": _dot(b_cross_kappa, grad_psi),
        "b_cross_kappa_dot_grad_alpha": _dot(b_cross_kappa, grad_alpha),
    }


def desc_geometry_arrays_from_equilibrium(
    eq,
    parallel_grid: ParallelGrid,
    *,
    rho: float,
    alpha: float = 0.0,
    iota: float | None = None,
    zeta_offset: float = 0.0,
    get_rtz_grid: Callable | None = None,
    linear_grid_cls: type | None = None,
) -> dict[str, object]:
    """Evaluate required DESC geometry arrays on ``parallel_grid.z``."""

    zeta = jnp.asarray(parallel_grid.z) + zeta_offset
    iota = _desc_iota(eq, rho, linear_grid_cls=linear_grid_cls) if iota is None else iota
    grid = _desc_field_line_grid(
        eq,
        rho=rho,
        alpha=alpha,
        zeta=zeta,
        iota=iota,
        get_rtz_grid=get_rtz_grid,
    )
    data = eq.compute(list(DESC_GEOMETRY_COMPUTE_KEYS), grid=grid)
    return desc_geometry_arrays_from_data(data, zeta=zeta, rho=rho, alpha=alpha)


def desc_geometry_arrays_from_path(
    path,
    parallel_grid: ParallelGrid,
    *,
    rho: float,
    alpha: float = 0.0,
    iota: float | None = None,
    zeta_offset: float = 0.0,
    file_format: str | None = None,
    index: int = -1,
    loader: Callable | None = None,
    get_rtz_grid: Callable | None = None,
    linear_grid_cls: type | None = None,
) -> dict[str, object]:
    """Load a DESC equilibrium file and evaluate geometry on ``parallel_grid.z``."""

    eq = load_desc_equilibrium(
        path,
        file_format=file_format,
        index=index,
        loader=loader,
    )
    return desc_geometry_arrays_from_equilibrium(
        eq,
        parallel_grid,
        rho=rho,
        alpha=alpha,
        iota=iota,
        zeta_offset=zeta_offset,
        get_rtz_grid=get_rtz_grid,
        linear_grid_cls=linear_grid_cls,
    )


def build_desc_geometry_from_equilibrium(
    eq,
    parallel_grid: ParallelGrid,
    *,
    rho: float,
    alpha: float = 0.0,
    iota: float | None = None,
    zeta_offset: float = 0.0,
    get_rtz_grid: Callable | None = None,
    linear_grid_cls: type | None = None,
):
    """Build internal solver geometry by sampling a DESC equilibrium."""

    arrays = desc_geometry_arrays_from_equilibrium(
        eq,
        parallel_grid,
        rho=rho,
        alpha=alpha,
        iota=iota,
        zeta_offset=zeta_offset,
        get_rtz_grid=get_rtz_grid,
        linear_grid_cls=linear_grid_cls,
    )
    return build_desc_geometry_from_arrays(parallel_grid, **arrays)


def build_desc_geometry_from_path(
    path,
    parallel_grid: ParallelGrid,
    *,
    rho: float,
    alpha: float = 0.0,
    iota: float | None = None,
    zeta_offset: float = 0.0,
    file_format: str | None = None,
    index: int = -1,
    loader: Callable | None = None,
    get_rtz_grid: Callable | None = None,
    linear_grid_cls: type | None = None,
):
    """Load a DESC equilibrium file and build internal solver geometry."""

    arrays = desc_geometry_arrays_from_path(
        path,
        parallel_grid,
        rho=rho,
        alpha=alpha,
        iota=iota,
        zeta_offset=zeta_offset,
        file_format=file_format,
        index=index,
        loader=loader,
        get_rtz_grid=get_rtz_grid,
        linear_grid_cls=linear_grid_cls,
    )
    return build_desc_geometry_from_arrays(parallel_grid, **arrays)


def _desc_field_line_grid(
    eq,
    *,
    rho,
    alpha,
    zeta,
    iota,
    get_rtz_grid: Callable | None,
):
    get_rtz_grid = get_rtz_grid or _import_desc_get_rtz_grid()
    return get_rtz_grid(eq, rho, alpha, np.asarray(zeta), coordinates="raz", iota=iota)


def _desc_iota(eq, rho, *, linear_grid_cls):
    linear_grid_cls = linear_grid_cls or _import_desc_linear_grid()
    grid = linear_grid_cls(rho=np.asarray([rho]), theta=np.asarray([0.0]), zeta=np.asarray([0.0]))
    data = eq.compute(["iota"], grid=grid)
    iota = data["iota"]
    if hasattr(grid, "compress"):
        iota = grid.compress(iota)
    return jnp.ravel(jnp.asarray(iota))[0]


def _select_desc_equilibrium(loaded, *, index: int):
    if hasattr(loaded, "compute"):
        return loaded
    try:
        eq = loaded[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise TypeError(
            "DESC file must contain an equilibrium or indexable equilibrium family"
        ) from exc
    if not hasattr(eq, "compute"):
        raise TypeError("selected DESC object does not provide an eq.compute method")
    return eq


def _import_desc_load():
    try:
        module = importlib.import_module("desc.io")
    except ImportError as exc:
        raise ImportError(
            "DESC is required for HDF5/path equilibrium loading. Install desc-opt "
            "or pass --desc-root to the extraction script."
        ) from exc
    return module.load


def _import_desc_get_rtz_grid():
    try:
        module = importlib.import_module("desc.equilibrium.coords")
    except ImportError as exc:
        raise ImportError(
            "DESC is required for direct equilibrium extraction. Install desc-opt "
            "or pass --desc-root to the extraction script."
        ) from exc
    return module.get_rtz_grid


def _import_desc_linear_grid():
    try:
        module = importlib.import_module("desc.grid")
    except ImportError as exc:
        raise ImportError(
            "DESC is required to compute iota automatically. Install desc-opt, "
            "pass --desc-root to the extraction script, or supply iota explicitly."
        ) from exc
    return module.LinearGrid


def _coerce_scalar_or_1d(name: str, value):
    array = jnp.asarray(value)
    if array.ndim == 0:
        return array[None]
    if array.ndim != 1:
        raise ValueError(f"{name} must be scalar or one-dimensional; got shape {array.shape}")
    return array


def _require_1d(data: dict, name: str, shape: tuple[int, ...]):
    if name not in data:
        raise KeyError(f"DESC data is missing required key {name!r}")
    return _optional_1d(data, name, shape, default=None)


def _optional_1d(data: dict, name: str, shape: tuple[int, ...], *, default):
    value = data.get(name, default)
    array = jnp.asarray(value)
    if array.shape == ():
        return jnp.broadcast_to(array, shape)
    if array.shape != shape:
        raise ValueError(f"DESC data {name!r} must have shape {shape}; got {array.shape}")
    return array


def _require_vector(data: dict, name: str, shape: tuple[int, ...], *, default=None):
    if name not in data and default is None:
        raise KeyError(f"DESC data is missing required key {name!r}")
    array = jnp.asarray(data.get(name, default))
    expected = shape + (3,)
    if array.shape != expected:
        raise ValueError(f"DESC data {name!r} must have shape {expected}; got {array.shape}")
    return array


def _dot(left, right):
    return jnp.sum(left * right, axis=-1)
