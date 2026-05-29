"""Matrix-free linear-operator helpers for residual and eigenvalue workflows."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .solver import linear_residual
from .types import ModeConnectivity


def mode_chain_mask(n_kx: int, n_ky: int, connectivity: ModeConnectivity | None, dtype=None):
    """Return the ``kx``-chain mask containing ``kx=0`` for each ``ky``."""

    dtype = jnp.float64 if dtype is None else dtype
    if connectivity is None:
        return jnp.ones((n_kx, n_ky), dtype=dtype)
    labels = jnp.asarray(connectivity.mode_label)
    targets = labels[connectivity.ixzero, :]
    return (labels == targets[None, :]).astype(dtype)


def project_to_ky(state, ky_index: int | None):
    """Project a phase-space or field array to one retained ``ky`` index."""

    state = jnp.asarray(state)
    if ky_index is None:
        return state
    mask = jnp.zeros((state.shape[-1],), dtype=state.dtype)
    mask = mask.at[ky_index].set(1.0)
    return state * mask.reshape((1,) * (state.ndim - 1) + (state.shape[-1],))


def project_to_mode_chain(
    state,
    connectivity: ModeConnectivity | None,
    *,
    ky_index: int | None = None,
):
    """Project an array to the connected ``kx`` chain containing ``kx=0``."""

    state = jnp.asarray(state)
    mask = mode_chain_mask(
        state.shape[-2],
        state.shape[-1],
        connectivity,
        dtype=state.real.dtype,
    ).astype(state.dtype)
    projected = state * mask.reshape((1,) * (state.ndim - 2) + mask.shape)
    return project_to_ky(projected, ky_index)


def linear_operator_action(
    state,
    precomputed,
    *,
    ky_index: int | None = None,
    connectivity: ModeConnectivity | None = None,
    project_output: bool = True,
):
    """Apply the Phase 7 matrix-free linear residual, optionally restricted by mode."""

    input_state = project_to_mode_chain(state, connectivity, ky_index=ky_index)
    output = linear_residual(input_state, precomputed=precomputed)
    if project_output:
        output = project_to_mode_chain(output, connectivity, ky_index=ky_index)
    return output


def flatten_state(state):
    """Flatten a state for external linear algebra packages."""

    return jnp.ravel(jnp.asarray(state))


def unflatten_state(vector, template_or_shape):
    """Unflatten a vector using a template array or explicit shape tuple."""

    shape = template_or_shape if isinstance(template_or_shape, tuple) else jnp.asarray(template_or_shape).shape
    return jnp.reshape(jnp.asarray(vector), shape)


def dense_matrix_from_action(action, template, *, dtype=None, max_size: int | None = 4096):
    """Build a dense matrix from a matrix-free action for tiny validation problems."""

    template = jnp.asarray(template)
    flat_template = flatten_state(template)
    size = int(flat_template.shape[0])
    if max_size is not None and size > max_size:
        raise ValueError("dense matrix construction is intended only for reduced test problems")
    dtype = _operator_dtype(template, dtype)
    basis = jnp.eye(size, dtype=dtype)

    def apply_basis(vector):
        state = unflatten_state(vector, template.shape)
        return flatten_state(action(state))

    responses = jax.vmap(apply_basis)(basis)
    return jnp.swapaxes(responses, 0, 1)


def dense_linear_operator_matrix(
    template,
    precomputed,
    *,
    ky_index: int | None = None,
    connectivity: ModeConnectivity | None = None,
    max_size: int | None = 4096,
):
    """Build a dense matrix for the restricted Phase 7 residual on a small template."""

    def action(state):
        return linear_operator_action(
            state,
            precomputed,
            ky_index=ky_index,
            connectivity=connectivity,
        )

    return dense_matrix_from_action(action, template, max_size=max_size)


def dense_eigensystem(matrix_or_action, template=None, *, max_size: int | None = 4096):
    """Return eigenvalues/eigenvectors for a dense matrix or tiny matrix-free action."""

    if template is None:
        matrix = jnp.asarray(matrix_or_action)
    else:
        matrix = dense_matrix_from_action(matrix_or_action, template, max_size=max_size)
    return jnp.linalg.eig(matrix)


def _operator_dtype(template, dtype):
    if dtype is not None:
        return jnp.dtype(dtype)
    real_dtype = jnp.asarray(template).real.dtype
    return jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64
