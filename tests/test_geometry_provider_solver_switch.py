from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stellarator_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    GxEikGeometryProvider,
    GeometryRequest,
    PhysicalArrayGeometryProvider,
    SpeciesParams,
    StellaGeometryProvider,
    SyntheticGeometryProvider,
    VelocityGridSpec,
    build_fourier_grid,
    build_linear_residual_precompute,
    build_velocity_grid,
    internal_geometry_from_result,
    linear_residual,
    load_stella_geometry_data,
    resolve_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
GX_EIK = ROOT / "fixtures/gx_desc_dshape_rho05_alpha0.eik.out"
STELLA_GEOMETRY = (
    ROOT / "fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.geometry"
)


def _physical_arrays(_request, parallel):
    ones = jnp.ones_like(parallel.z)
    zeros = jnp.zeros_like(parallel.z)
    return {
        "B": 1.0 + 0.04 * jnp.cos(parallel.z),
        "b_dot_grad_z": 0.3 * ones,
        "grad_psi_sq": ones,
        "grad_alpha_sq": 1.2 * ones,
        "grad_psi_dot_grad_alpha": zeros,
        "B_cross_gradB_dot_grad_psi": zeros,
        "B_cross_gradB_dot_grad_alpha": zeros,
        "b_cross_kappa_dot_grad_psi": zeros,
        "b_cross_kappa_dot_grad_alpha": zeros,
    }


def _provider_cases():
    base_request = GeometryRequest(configuration="provider-switch", n_z=8)
    for name in ("desc", "vmecpp", "gvec"):
        yield (
            name,
            PhysicalArrayGeometryProvider(
                arrays=_physical_arrays,
                provider_name=name,
                differentiable=True,
            ),
            base_request,
        )
    yield "synthetic", SyntheticGeometryProvider(), base_request
    yield (
        "gx-eik",
        GxEikGeometryProvider(GX_EIK, iota=1.0 / 1.2012012012012012),
        GeometryRequest(configuration="gx-eik", n_z=32),
    )
    stella = load_stella_geometry_data(STELLA_GEOMETRY)
    yield (
        "stella-geometry",
        StellaGeometryProvider(STELLA_GEOMETRY, nfp=5),
        GeometryRequest(
            configuration="stella-W7X",
            radial_value=0.8,
            parallel_coordinate="zed_over_2pi",
            parallel_coordinate_unit="turn",
            n_z=256,
            z_min=-0.5,
            z_max=0.5,
            field_periods=stella.field_periods,
        ),
    )


@pytest.mark.parametrize("name,provider,geometry_request", tuple(_provider_cases()))
def test_solver_construction_is_provider_independent(name, provider, geometry_request):
    result = resolve_geometry(provider, geometry_request)
    geometry = internal_geometry_from_result(result)
    velocity = build_velocity_grid(
        VelocityGridSpec(n_vpar=2, n_mu=2, vpar_max=1.5, mu_max=1.0)
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.2,))
    )
    species = SpeciesParams(
        charge=1.0,
        mass=2.0,
        density=1.0,
        temperature=1.0,
        density_gradient=1.0,
        temperature_gradient=2.0,
    )
    precomputed = build_linear_residual_precompute(
        velocity,
        result.parallel_grid,
        fourier,
        geometry,
        species,
        electron_params=AdiabaticElectronParams(density=1.0, temperature=1.0),
    )
    state = jnp.zeros((2, 2, geometry_request.n_z, 1, 1), dtype=jnp.complex128)

    rhs = linear_residual(state, precomputed=precomputed)

    assert result.metadata.provenance.provider == name
    assert rhs.shape == state.shape
    assert np.all(np.isfinite(np.asarray(rhs)))


def test_synthetic_provider_retains_gradient_to_continuous_control():
    request = GeometryRequest(configuration="synthetic-gradient", n_z=16)

    def objective(amplitude):
        result = resolve_geometry(
            SyntheticGeometryProvider(magnetic_field_amplitude=amplitude),
            request,
            validate=False,
        )
        return jnp.sum(internal_geometry_from_result(result).B**2)

    derivative = jax.jit(jax.grad(objective))(0.1)
    assert jnp.isfinite(derivative)
    assert abs(float(derivative)) > 0.0


def test_physical_array_provider_names_missing_fields():
    provider = PhysicalArrayGeometryProvider(
        arrays={"B": np.ones(8)},
        provider_name="incomplete",
    )
    with pytest.raises(ValueError, match="missing required field 'b_dot_grad_z'"):
        provider.get_geometry(GeometryRequest(n_z=8))
