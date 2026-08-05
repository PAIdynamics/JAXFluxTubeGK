from __future__ import annotations

import numpy as np

from scripts.audit_w7x_ky03_rhs_model_balance import RHSTermSplit
from scripts.replay_w7x_stella_state_in_solver import (
    bundled_solver_rhs,
    phase_space_to_solver,
    replay_cases,
    selected_mode_from_solver,
)


def test_replay_velocity_grids_are_inside_stella_trace_domain():
    from stellarator_gk import VelocityGridSpec, build_velocity_grid

    for case in replay_cases():
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


def test_phase_space_order_round_trip():
    expected = np.arange(3 * 4 * 2).reshape(3, 4, 2).astype(complex)
    actual = selected_mode_from_solver(phase_space_to_solver(expected))
    np.testing.assert_array_equal(actual, expected)


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
