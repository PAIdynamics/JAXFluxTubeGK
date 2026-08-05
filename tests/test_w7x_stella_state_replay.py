from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.audit_w7x_ky03_rhs_model_balance import RHSTermSplit
from scripts.replay_w7x_stella_state_in_solver import (
    apply_stella_native_drift_algebra,
    apply_stella_coefficient_contract,
    bundled_solver_rhs,
    phase_space_to_solver,
    replay_cases,
    selected_mode_from_solver,
    stella_third_order_upwind_matrix,
    stella_second_order_centered_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def test_replay_velocity_grids_are_inside_stella_trace_domain():
    from stellarator_gk import VelocityGridSpec, build_velocity_grid

    for case in replay_cases():
        if case.name.startswith("replay_stella_native_"):
            continue
        grid = build_velocity_grid(
            VelocityGridSpec(
                n_vpar=case.n_vpar,
                n_mu=case.n_mu,
                vpar_max=case.vpar_max,
                mu_max=case.mu_max,
                backend=case.velocity_backend,
            )
        )
        assert np.min(grid.vpar) >= -3.0
        assert np.max(grid.vpar) <= 3.0
        assert np.min(grid.mu) >= 0.0366213
        assert np.max(grid.mu) <= 4.91707


def test_replay_includes_source_derived_stella_coefficient_discriminator():
    cases = {case.name: case for case in replay_cases()}
    assert cases["replay_stella_coefficients_16x4"].parallel_derivative_model == "gkw_upwind"
    high_resolution = cases["replay_stella_coefficients_32x4"]
    assert high_resolution.parallel_derivative_model == "gkw_upwind"
    np.testing.assert_allclose(high_resolution.vpar_max * 31.0 / 32.0, 3.0)
    native = cases["replay_stella_native_32x8"]
    assert (native.n_vpar, native.n_mu) == (32, 8)


def test_non_discriminator_keeps_precompute_identity():
    marker = object()
    case = next(case for case in replay_cases() if case.name == "replay_open_16x4")
    assert apply_stella_coefficient_contract(case, marker, None) is marker


def test_committed_same_state_result_passes_native_parity_gate():
    path = (
        ROOT
        / "fixtures/w7x_ky03_stella_state_replay/same_state_rhs_replay_status.json"
    )
    status = json.loads(path.read_text(encoding="utf-8"))
    assert status["status"] == "same_state_rhs_parity_passed"
    assert status["acceptance_case"] == "replay_stella_native_32x8"
    assert status["best_by_quantity"]["equilibrium_drive"]["relative_l2_error"] < 0.1
    assert status["best_by_quantity"]["total_rhs"]["relative_l2_error"] < 0.3
    assert status["best_by_quantity"]["mirror_force"]["relative_l2_error"] < 4.0e-3
    assert (
        status["native_grid_mirror_reconstruction"]["max_relative_l2_error"]
        < 4.0e-3
    )
    assert (
        status["native_grid_quasineutrality_reconstruction"]["max_relative_l2_error"]
        < 2.0e-15
    )


def test_phase_space_order_round_trip():
    expected = np.arange(3 * 4 * 2).reshape(3, 4, 2).astype(complex)
    actual = selected_mode_from_solver(phase_space_to_solver(expected))
    np.testing.assert_array_equal(actual, expected)


def test_stella_upwind_matrix_matches_zero_boundary_source_stencil():
    values = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    positive = stella_third_order_upwind_matrix(5, 0.5, 1) @ values
    negative = stella_third_order_upwind_matrix(5, 0.5, -1) @ values
    assert positive[-1] == -32.0
    assert positive[0] == 2.0
    assert negative[0] == 2.0
    assert negative[-1] == 16.0


def test_stella_upwind_matrix_is_exact_for_interior_cubic():
    nodes = np.arange(8, dtype=float)
    values = nodes**3
    exact = 3.0 * nodes**2
    for sign in (-1, 1):
        actual = stella_third_order_upwind_matrix(8, 1.0, sign) @ values
        interior = slice(2, -2)
        np.testing.assert_allclose(actual[interior], exact[interior], atol=1.0e-13)


def test_stella_centered_matrix_matches_open_source_boundaries():
    values = np.asarray([1.0, 2.0, 4.0, 8.0])
    positive = stella_second_order_centered_matrix(4, 0.5, 1) @ values
    negative = stella_second_order_centered_matrix(4, 0.5, -1) @ values
    np.testing.assert_allclose(positive, [2.0, 3.0, 6.0, -4.0])
    np.testing.assert_allclose(negative, [2.0, 3.0, 6.0, 8.0])


def test_semantic_bundles_include_field_drive():
    shape = (2, 2, 3, 1, 1)

    def term(value):
        return np.full(shape, value, dtype=complex)

    split = RHSTermSplit(
        names=(
            "parallel_streaming",
            "parallel_field_drive",
            "mirror_force",
            "magnetic_drift",
            "drift_field_drive",
            "equilibrium_drive",
        ),
        terms=(term(1), term(2), term(3), term(4), term(5), term(6)),
    )
    bundles = bundled_solver_rhs(split, term(7))
    assert set(bundles) == {
        "parallel_streaming",
        "mirror_force",
        "magnetic_drift",
        "equilibrium_drive",
        "total_rhs",
    }
    np.testing.assert_array_equal(bundles["parallel_streaming"], 3)
    np.testing.assert_array_equal(bundles["magnetic_drift"], 9)


def test_native_stella_drift_uses_traced_g_and_phi_coefficients():
    case = next(case for case in replay_cases() if case.name == "replay_stella_native_32x8")
    shape = (2, 1, 3, 1, 1)
    state = np.full(shape, 2.0 + 0.0j)
    phi = np.full((3, 1, 1), 5.0 + 0.0j)
    split = RHSTermSplit(
        names=("magnetic_drift", "drift_field_drive"),
        terms=(np.zeros(shape, dtype=complex), np.zeros(shape, dtype=complex)),
    )
    coefficients = {
        "magnetic_drift_g_y": np.full((3, 2, 1), 7.0),
        "magnetic_drift_phi_y": np.full((3, 2, 1), 11.0),
        "gyroaverage_j0": np.full((3, 2, 1), 0.5),
    }

    actual = apply_stella_native_drift_algebra(case, state, phi, split, coefficients)

    np.testing.assert_allclose(actual.terms[0], 1j * 0.3 * 7.0 * 2.0)
    np.testing.assert_allclose(actual.terms[1], 1j * 0.3 * 11.0 * 0.5 * 5.0)
