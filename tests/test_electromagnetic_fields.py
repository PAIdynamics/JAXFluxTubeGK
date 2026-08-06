import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stellarator_gk import (
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    build_fourier_grid,
    build_linear_residual_precompute,
    build_parallel_ampere_precompute,
    build_parallel_grid,
    build_s_alpha_geometry,
    build_velocity_grid,
    parallel_ampere_residual,
    solve_parallel_ampere,
)
from stellarator_gk.tem_validation import _build_tem_system, gyaradax_tem_case_spec


def _setup(beta=0.01):
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=4,
            vpar_max=3.0,
            mu_max=4.5,
            backend="finite_difference",
        )
    )
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=8, z_min=-0.5, z_max=0.5, topology="periodic")
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.4, ky_values=(0.0, 0.3))
    )
    geometry = build_s_alpha_geometry(
        parallel, GeometryScalarParams(q=1.4, shat=0.8, eps=0.18)
    )
    species = (
        SpeciesParams(1.0, 1.0, 1.0, 1.0, 2.2, 0.0),
        SpeciesParams(-1.0, 0.01, 1.0, 1.0, 2.2, 6.9),
    )
    linear = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species, field_model="kinetic"
    )
    ampere = build_parallel_ampere_precompute(
        velocity,
        geometry,
        fourier,
        species,
        linear.rhs.flr_factors,
        beta=beta,
    )
    return velocity, parallel, fourier, geometry, species, ampere


def test_parallel_ampere_solve_closes_discrete_field_residual():
    velocity, parallel, fourier, _geometry, _species, precompute = _setup()
    shape = (2, velocity.vpar.shape[0], velocity.mu.shape[0], parallel.z.shape[0], 3, 2)
    state = jax.random.normal(jax.random.key(2), shape).astype(jnp.complex128)

    apar = jax.jit(solve_parallel_ampere)(state, precompute)
    residual = parallel_ampere_residual(apar, state, precompute)

    assert apar.shape == (parallel.z.shape[0], fourier.kx.shape[0], fourier.ky.shape[0])
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-12, rtol=0.0)


def test_parallel_ampere_selects_odd_parallel_velocity_current():
    velocity, parallel, _fourier, _geometry, _species, precompute = _setup()
    envelope = jnp.ones((2, 8, 4, parallel.z.shape[0], 3, 2))
    even = (velocity.vpar**2)[None, :, None, None, None, None] * envelope
    odd = velocity.vpar[None, :, None, None, None, None] * envelope

    np.testing.assert_allclose(solve_parallel_ampere(even, precompute), 0.0, atol=1.0e-14)
    assert float(jnp.max(jnp.abs(solve_parallel_ampere(odd, precompute)))) > 0.0


def test_parallel_ampere_vanishes_in_electrostatic_limit():
    velocity, parallel, _fourier, _geometry, _species, precompute = _setup(beta=0.0)
    state = jnp.ones((2, 8, 4, parallel.z.shape[0], 3, 2))

    np.testing.assert_allclose(precompute.source_weight, 0.0, atol=0.0)
    np.testing.assert_allclose(solve_parallel_ampere(state, precompute), 0.0, atol=0.0)


def test_parallel_ampere_beta_response_is_differentiable():
    velocity, parallel, fourier, geometry, species, _precompute = _setup()
    state = velocity.vpar[None, :, None, None, None, None] * jnp.ones(
        (2, 8, 4, parallel.z.shape[0], 3, 2)
    )
    linear = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species, field_model="kinetic"
    )

    def objective(beta):
        precompute = build_parallel_ampere_precompute(
            velocity, geometry, fourier, species, linear.rhs.flr_factors, beta=beta
        )
        return jnp.sum(jnp.abs(solve_parallel_ampere(state, precompute)) ** 2)

    value = objective(0.01)
    derivative = jax.grad(objective)(0.01)
    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))


@pytest.mark.external
def test_parallel_ampere_precompute_matches_pinned_gyaradax(gyaradax_root):
    import sys

    if str(gyaradax_root) not in sys.path:
        sys.path.insert(0, str(gyaradax_root))
    from gyaradax.geometry import compute_geometry
    from gyaradax.integrals import precompute_apar
    from gyaradax.params import GKParams

    spec = dataclasses.replace(gyaradax_tem_case_spec(), n_z=8, n_vpar=8, n_mu=4)
    velocity, _parallel, fourier, geometry, species, linear = _build_tem_system(spec)
    observed = build_parallel_ampere_precompute(
        velocity,
        geometry,
        fourier,
        species,
        linear.rhs.flr_factors,
        beta=0.01,
    )
    kthnorm = spec.q / (2.0 * np.pi * spec.eps)
    reference_geometry = compute_geometry(
        q=spec.q,
        shat=spec.shat,
        eps=spec.eps,
        ns=8,
        nvpar=8,
        nmu=4,
        vpar_max=3.0,
        nkx=1,
        nky=1,
        nperiod=2,
        kxmax=spec.ky * kthnorm,
        krhomax=spec.ky * kthnorm,
        geom_type="s-alpha",
    )
    masses = jnp.asarray([1.0, spec.electron_mass])
    params = GKParams(
        adiabatic_electrons=False,
        beta=0.01,
        mas=masses,
        signz=jnp.asarray([1.0, -1.0]),
        tmp=jnp.ones(2),
        de=jnp.ones(2),
        vthrat=jnp.sqrt(1.0 / masses),
        dgrid=1.0,
    )
    reference_weight, reference_denominator, reference_kperp2 = precompute_apar(
        reference_geometry, params
    )

    # optimal-fusion uses a differentiable Cephes J0 approximation while
    # Gyaradax uses jax.scipy.special.bessel_jn.  Their field coefficients
    # agree to the approximation error rather than bitwise precision.
    np.testing.assert_allclose(observed.source_weight, reference_weight, rtol=3e-8, atol=2e-13)
    np.testing.assert_allclose(observed.denominator, reference_denominator, rtol=3e-8, atol=2e-13)
    np.testing.assert_allclose(
        observed.kperp_squared,
        np.asarray(reference_kperp2).reshape(observed.kperp_squared.shape),
        rtol=2e-13,
        atol=2e-13,
    )
