from pathlib import Path

import numpy as np

from jax_fluxtube_gk import (
    GxEikGeometryProvider,
    GeometryRequest,
    StellaGeometryProvider,
    has_duplicate_stella_endpoint,
    internal_geometry_from_result,
    load_gx_eik_data,
    load_stella_geometry_data,
    resolve_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
GX_EIK = ROOT / "fixtures/gx_desc_dshape_rho05_alpha0.eik.out"
STELLA_GEOMETRY = (
    ROOT / "fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.geometry"
)


def test_gx_block_eik_reader_lives_in_geometry_adapter():
    data = load_gx_eik_data(GX_EIK)

    assert data.theta.shape == (33,)
    assert data.header[:3] == (16.0, 1.0, 32.0)
    np.testing.assert_allclose(data.theta[[0, -1]], (-np.pi, np.pi), rtol=2.0e-12)
    assert np.all(np.isfinite(data.bmag))
    assert np.all(data.bmag > 0.0)


def test_gx_eik_provider_returns_physical_contract_and_canonical_drifts():
    request = GeometryRequest(
        configuration="desc-dshape-eik",
        radial_value=0.5,
        alpha=0.0,
        n_z=32,
    )
    provider = GxEikGeometryProvider(
        GX_EIK,
        iota=1.0 / 1.2012012012012012,
        shear=0.40240240240240244,
        provider_version="fixture",
        revision="fixture-contract",
    )

    result = resolve_geometry(provider, request)
    internal = internal_geometry_from_result(result)
    sampled = load_gx_eik_data(GX_EIK)

    assert result.metadata.provenance.provider == "gx-eik"
    assert result.metadata.differentiable is False
    assert result.physical.provider == "gx-eik"
    np.testing.assert_allclose(result.physical.alpha, request.alpha)
    np.testing.assert_allclose(
        internal.D_x,
        sampled.gbdrift0[:-1] + sampled.cvdrift0[:-1],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        internal.D_y,
        sampled.gbdrift[:-1] + sampled.cvdrift[:-1],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(internal.E_y, sampled.gbdrift[:-1], atol=1.0e-14)
    np.testing.assert_allclose(internal.g_xx, sampled.gds22[:-1], atol=1.0e-14)
    np.testing.assert_allclose(internal.g_xy, sampled.gds21[:-1], atol=1.0e-14)
    np.testing.assert_allclose(internal.g_yy, sampled.gds2[:-1], atol=1.0e-14)


def test_stella_reader_drops_duplicate_periodic_endpoint():
    retained = load_stella_geometry_data(STELLA_GEOMETRY, drop_periodic_endpoint=False)
    dropped = load_stella_geometry_data(STELLA_GEOMETRY)

    assert has_duplicate_stella_endpoint(retained.rows)
    assert retained.original_n_z == 257
    assert retained.rows.shape == (257, 12)
    assert dropped.dropped_periodic_endpoint is True
    assert dropped.rows.shape == (256, 12)
    np.testing.assert_allclose(dropped.global_value("rhoc"), 0.8)
    np.testing.assert_allclose(dropped.global_value("qinp"), 1.098)


def test_stella_provider_returns_same_public_physical_contract():
    data = load_stella_geometry_data(STELLA_GEOMETRY)
    request = GeometryRequest(
        configuration="w7x-stella",
        radial_value=0.8,
        alpha=0.0,
        parallel_coordinate="zed_over_2pi",
        parallel_coordinate_unit="turn",
        n_z=256,
        z_min=-0.5,
        z_max=0.5,
        field_periods=data.field_periods,
    )

    result = resolve_geometry(
        StellaGeometryProvider(
            STELLA_GEOMETRY,
            nfp=5,
            provider_version="fixture",
            revision="fixture-contract",
        ),
        request,
    )
    internal = internal_geometry_from_result(result)

    assert result.metadata.provenance.provider == "stella-geometry"
    assert result.metadata.nfp == 5
    assert result.metadata.endpoint_policy == "exclude"
    assert result.physical.B.shape == (256,)
    np.testing.assert_allclose(result.physical.iota, 1.0 / 1.098)
    np.testing.assert_allclose(result.physical.shear, -0.1051)
    np.testing.assert_allclose(result.physical.alpha, 0.0, atol=1.0e-14)
    assert np.all(np.isfinite(internal.G))
    assert np.max(np.abs(np.asarray(internal.D_x))) > 0.0
    assert np.max(np.abs(np.asarray(internal.D_y))) > 0.0


def test_stella_provider_rejects_mismatched_resolution_and_field_span():
    data = load_stella_geometry_data(STELLA_GEOMETRY)
    provider = StellaGeometryProvider(STELLA_GEOMETRY, nfp=5)

    with np.testing.assert_raises_regex(ValueError, "provides n_z=256"):
        provider.get_geometry(
            GeometryRequest(
                n_z=32,
                z_min=-0.5,
                z_max=0.5,
                parallel_coordinate="zed_over_2pi",
                parallel_coordinate_unit="turn",
                field_periods=data.field_periods,
            )
        )

    with np.testing.assert_raises_regex(ValueError, "field-line turns"):
        provider.get_geometry(
            GeometryRequest(
                n_z=256,
                z_min=-0.5,
                z_max=0.5,
                parallel_coordinate="zed_over_2pi",
                parallel_coordinate_unit="turn",
                field_periods=1.0,
            )
        )
