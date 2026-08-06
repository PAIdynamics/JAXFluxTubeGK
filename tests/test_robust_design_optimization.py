import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stellarator_gk import (
    DesignObjectiveSpec,
    OptimizationScanResult,
    OptimizationTopologyContract,
    RobustAggregationSpec,
    TopologyChangeError,
    aggregate_design_objectives,
    assert_checkpoint_topology,
    build_optimization_checkpoint,
    load_optimization_checkpoint,
    robust_scan_objective,
    write_optimization_checkpoint,
)


def _topology(*, parallel_size=8):
    return OptimizationTopologyContract(
        schema_version=1,
        velocity_shape=(3, 2),
        velocity_backend="gauss-legendre",
        velocity_nodes_digest="velocity",
        parallel_size=parallel_size,
        parallel_backend="fourier",
        parallel_topology="periodic",
        parallel_nodes_digest="parallel",
        fourier_shape=(1, 2),
        ikxspace=1,
        fourier_modes_digest="fourier",
        connectivity_digest="connectivity",
        provider_topology=(1, 5, 1.0, "periodic", "exclude", "periodic", False, ()),
    )


def test_robust_aggregation_reports_all_reductions_and_weights():
    values = jnp.asarray([1.0, 2.0, 4.0])
    result = aggregate_design_objectives(
        values,
        RobustAggregationSpec(method="weighted_mean"),
        sample_weights=[1.0, 2.0, 1.0],
    )

    np.testing.assert_allclose(result.normalized_weights, [0.25, 0.5, 0.25])
    assert float(result.scalar_objective) == pytest.approx(2.25)
    assert float(result.weighted_mean) == pytest.approx(2.25)
    assert float(result.worst_case) == pytest.approx(4.0)
    assert 2.25 < float(result.softmax_objective) < 4.0


def test_softmax_aggregation_has_finite_gradient_at_degenerate_samples():
    spec = RobustAggregationSpec(method="softmax", softmax_temperature=0.1)

    gradient = jax.grad(
        lambda shift: aggregate_design_objectives(
            jnp.asarray([shift, shift]), spec
        ).scalar_objective
    )(1.0)

    assert float(gradient) == pytest.approx(1.0)
    assert np.isfinite(float(gradient))


def test_robust_scan_preserves_rho_alpha_ky_sample_order():
    scan = OptimizationScanResult(
        objectives=jnp.arange(8.0).reshape(2, 2, 2),
        rho_values=jnp.asarray([0.4, 0.8]),
        alpha_values=jnp.asarray([0.0, 0.5]),
        ky_indices=jnp.asarray([0, 1]),
    )
    result = robust_scan_objective(
        scan,
        RobustAggregationSpec(method="worst_case"),
    )

    np.testing.assert_array_equal(result.sample_objectives, np.arange(8.0))
    assert float(result.scalar_objective) == pytest.approx(7.0)


@pytest.mark.parametrize(
    "weights",
    ([1.0], [-1.0, 2.0], [0.0, 0.0], [np.nan, 1.0]),
)
def test_robust_aggregation_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        aggregate_design_objectives([1.0, 2.0], sample_weights=weights)


def test_checkpoint_round_trip_preserves_reproducibility_contract(tmp_path):
    checkpoint = build_optimization_checkpoint(
        iteration=3,
        objective_spec=DesignObjectiveSpec(selected_ky=0, frequency_weight=0.2),
        aggregation_spec=RobustAggregationSpec(
            method="softmax", softmax_temperature=0.03
        ),
        rho_values=[0.4, 0.8],
        alpha_values=[0.0, np.pi / 5.0],
        ky_indices=[0, 1],
        topology_contracts=[_topology()],
        provider_provenance=[
            {"provider": "vmecpp", "revision": "abc123", "configuration": "w7x-standard"}
        ],
        parameters={"boundary_coefficients": np.asarray([1.0, 0.2])},
        sample_objectives=np.arange(8.0),
        aggregate_objective=6.5,
        history=[{"iteration": 2, "objective": 6.8}],
        code_revision="deadbeef",
        dependency_revisions={"vmecpp": "abc123", "jax": "0.7.2"},
        command=["python", "optimize.py", "--seed", "17"],
        random_seed=17,
    )
    path = write_optimization_checkpoint(tmp_path / "checkpoint.json", checkpoint)
    loaded = load_optimization_checkpoint(path)

    assert loaded == checkpoint
    assert json.loads(path.read_text())["samples"]["rho_values"] == [0.4, 0.8]
    assert_checkpoint_topology(loaded, [_topology(), _topology()])
    with pytest.raises(FileExistsError):
        write_optimization_checkpoint(path, checkpoint)


def test_checkpoint_rejects_changed_restart_topology(tmp_path):
    checkpoint = build_optimization_checkpoint(
        iteration=0,
        objective_spec=DesignObjectiveSpec(selected_ky=0),
        aggregation_spec=RobustAggregationSpec(),
        rho_values=[0.8],
        alpha_values=[0.0],
        ky_indices=[0],
        topology_contracts=[_topology()],
        provider_provenance=[{"provider": "vmecpp", "revision": "abc"}],
        parameters={"scale": 1.0},
        sample_objectives=[0.2],
        aggregate_objective=0.2,
        code_revision="deadbeef",
        dependency_revisions={"vmecpp": "abc"},
        command=["optimize"],
    )

    with pytest.raises(TopologyChangeError, match="parallel_size"):
        assert_checkpoint_topology(checkpoint, [_topology(parallel_size=12)])


def test_checkpoint_requires_complete_sample_tensor():
    with pytest.raises(ValueError, match="rho/alpha/ky"):
        build_optimization_checkpoint(
            iteration=0,
            objective_spec=DesignObjectiveSpec(selected_ky=0),
            aggregation_spec=RobustAggregationSpec(),
            rho_values=[0.4, 0.8],
            alpha_values=[0.0],
            ky_indices=[0],
            topology_contracts=[_topology()],
            provider_provenance=[{"provider": "synthetic"}],
            parameters={},
            sample_objectives=[0.2],
            aggregate_objective=0.2,
            code_revision="deadbeef",
            dependency_revisions={},
            command=["optimize"],
        )
