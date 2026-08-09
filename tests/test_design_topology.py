from __future__ import annotations

from dataclasses import replace

import pytest

from jax_fluxtube_gk import (
    FourierGridSpec,
    ParallelGridSpec,
    TopologyChangeError,
    VelocityGridSpec,
    assert_fixed_optimization_topology,
    build_fourier_grid,
    build_mode_connectivity,
    build_optimization_topology_contract,
    build_parallel_grid,
    build_velocity_grid,
    optimization_topology_changes,
)


def _contract(*, n_z=8, ky_values=(0.1, 0.3), max_shift=4):
    velocity = build_velocity_grid(
        VelocityGridSpec(n_vpar=4, n_mu=3, vpar_max=2.0, mu_max=1.5)
    )
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=n_z, z_min=-0.5, z_max=0.5)
    )
    fourier = build_fourier_grid(
        FourierGridSpec(
            n_kx=3,
            n_ky=len(ky_values),
            kx_max=0.5,
            ky_values=ky_values,
            ikxspace=2,
        )
    )
    connectivity = build_mode_connectivity(fourier, max_shift=max_shift)
    return build_optimization_topology_contract(
        velocity,
        parallel,
        fourier,
        connectivity=connectivity,
    )


def test_identical_design_topology_can_reuse_compiled_objective():
    reference = _contract()
    candidate = _contract()

    assert optimization_topology_changes(reference, candidate) == ()
    assert_fixed_optimization_topology(reference, candidate)


@pytest.mark.parametrize(
    ("candidate", "expected_field"),
    [
        (lambda: _contract(n_z=10), "parallel_size"),
        (lambda: _contract(ky_values=(0.1, 0.2, 0.3)), "fourier_shape"),
        (lambda: _contract(max_shift=2), "connectivity_digest"),
    ],
)
def test_design_topology_detects_remeshing_and_relinking(candidate, expected_field):
    reference = _contract()
    changed = optimization_topology_changes(reference, candidate())

    assert expected_field in changed
    with pytest.raises(TopologyChangeError) as error:
        assert_fixed_optimization_topology(reference, candidate())
    assert expected_field in error.value.changed_fields
    assert "rebuild grids" in str(error.value)


def test_provider_topology_is_part_of_design_contract():
    reference = _contract()
    provider_reference = replace(
        reference,
        provider_topology=(2, 5, 1.0, "periodic", "exclude", "periodic", False, ()),
    )
    provider_remeshed = replace(
        provider_reference,
        provider_topology=(2, 5, 2.0, "periodic", "exclude", "periodic", False, ()),
    )

    assert optimization_topology_changes(provider_reference, provider_remeshed) == (
        "provider_topology",
    )
