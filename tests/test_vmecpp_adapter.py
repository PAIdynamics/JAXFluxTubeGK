from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_fluxtube_gk import (
    GeometryRequest,
    VmecppGeometryProvider,
    internal_geometry_from_result,
    load_stella_geometry_data,
    load_eik_data,
    resolve_geometry,
    vmec_field_line_from_wout,
)


def _circular_wout(*, lasym=False):
    ns = 9
    s = np.linspace(0.0, 1.0, ns)
    radius = 0.5 * np.sqrt(s)
    return SimpleNamespace(
        lasym=lasym,
        signgs=1,
        nfp=5,
        ns=ns,
        xm=np.array([0, 1]),
        xn=np.array([0, 0]),
        xm_nyq=np.array([0]),
        xn_nyq=np.array([0]),
        rmnc=np.vstack((5.5 * np.ones(ns), radius)),
        zmns=np.vstack((np.zeros(ns), radius)),
        lmns_full=np.zeros((2, ns)),
        bmnc=1.2 * np.ones((1, ns)),
        iotaf=0.7 + 0.04 * s,
        iotas=0.7 + 0.04 * s,
        phi=-2.0 * np.pi * s,
        presf=2.0e3 * (1.0 - s),
        pres=2.0e3 * (1.0 - s),
        Aminor_p=0.5,
    )


def _request(configuration="w7x-standard"):
    return GeometryRequest(
        configuration=configuration,
        radial_value=0.7,
        alpha=0.2,
        n_z=24,
        z_min=-np.pi / 5.0,
        z_max=np.pi / 5.0,
    )


def test_vmec_wout_transformation_builds_complete_physical_contract():
    arrays = vmec_field_line_from_wout(_circular_wout(), _request())

    assert set(arrays) == {
        "theta",
        "phi",
        "alpha",
        "rho",
        "iota",
        "shear",
        "B",
        "b_dot_grad_z",
        "grad_psi_sq",
        "grad_alpha_sq",
        "grad_psi_dot_grad_alpha",
        "B_cross_gradB_dot_grad_psi",
        "B_cross_gradB_dot_grad_alpha",
        "b_cross_kappa_dot_grad_psi",
        "b_cross_kappa_dot_grad_alpha",
    }
    np.testing.assert_allclose(arrays["theta"] - arrays["iota"] * arrays["phi"], 0.2)
    np.testing.assert_allclose(arrays["B"], 1.2 / 8.0)
    assert np.all(np.asarray(arrays["grad_psi_sq"]) > 0.0)
    assert np.all(np.asarray(arrays["grad_alpha_sq"]) > 0.0)
    assert all(np.all(np.isfinite(np.asarray(value))) for value in arrays.values())


def test_vmecpp_provider_consumes_in_memory_output_without_file_io():
    result = resolve_geometry(
        VmecppGeometryProvider(
            output=SimpleNamespace(wout=_circular_wout()),
            provider_version="test",
            revision="fake-vmecpp-revision",
        ),
        _request("in-memory-test"),
    )
    geometry = internal_geometry_from_result(result)

    assert result.metadata.provenance.provider == "vmecpp"
    assert result.metadata.provenance.source == "in-memory VmecOutput"
    assert result.metadata.provenance.revision == "fake-vmecpp-revision"
    assert result.metadata.differentiable is False
    assert result.physical.nfp == 5
    assert geometry.B.shape == (24,)
    assert np.all(np.isfinite(np.asarray(geometry.D_y)))


def test_named_w7x_provider_loads_input_then_runs_in_memory():
    calls = []
    vmec_input = object()

    def named_loader(name):
        calls.append(("load", name))
        return vmec_input

    def runner(value, *, max_threads, verbose):
        calls.append(("run", value, max_threads, verbose))
        return SimpleNamespace(wout=_circular_wout())

    result = resolve_geometry(
        VmecppGeometryProvider(
            named_loader=named_loader,
            runner=runner,
            max_threads=2,
        ),
        _request("W7-X"),
    )

    assert calls == [("load", "w7x-standard"), ("run", vmec_input, 2, False)]
    assert "installed VMEC++ named configuration" in result.metadata.provenance.source
    assert result.metadata.provenance.command == "vmecpp.run(...)"


def test_vmecpp_provider_rejects_unsupported_requests_and_asymmetry():
    provider = VmecppGeometryProvider(output=_circular_wout())
    with pytest.raises(ValueError, match="radial_coordinate='rho'"):
        provider.get_geometry(
            GeometryRequest(configuration="test", radial_coordinate="psi", radial_value=0.5)
        )
    with pytest.raises(NotImplementedError, match="lasym"):
        vmec_field_line_from_wout(_circular_wout(lasym=True), _request())


def test_vmecpp_provider_rejects_ambiguous_equilibrium_sources():
    with pytest.raises(ValueError, match="at most one"):
        VmecppGeometryProvider(output=object(), vmec_input=object())


@pytest.mark.external
def test_installed_vmecpp_runs_named_w7x_without_repository_artifact(vmecpp_root):
    result = resolve_geometry(
        VmecppGeometryProvider(revision="065ba90a147853f86326f31510d1debbfa424f5a"),
        _request(),
    )

    assert vmecpp_root.name == "vmecpp"
    assert result.metadata.provenance.configuration == "w7x-standard"
    assert result.metadata.provenance.source == (
        "installed VMEC++ named configuration w7x-standard"
    )
    assert result.physical.nfp == 5
    assert np.all(np.asarray(result.physical.B) > 0.0)


@pytest.mark.external
def test_direct_vmec_w7x_matches_independent_stella_geometry_terms(gx_root):
    root = Path(__file__).resolve().parents[1]
    stella = load_stella_geometry_data(
        root / "fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.geometry"
    )
    zeta_reference = stella.column("zeta")
    zeta_min = float(zeta_reference[0])
    zeta_max = zeta_min + 2.0 * np.pi * stella.field_periods
    request = GeometryRequest(
        configuration="external-gx-w7x-wout",
        radial_value=stella.global_value("rhoc"),
        alpha=float(stella.column("alpha")[0]),
        n_z=4096,
        z_min=zeta_min,
        z_max=zeta_max,
        field_periods=stella.field_periods,
    )
    result = resolve_geometry(
        VmecppGeometryProvider(
            wout_path=gx_root / "benchmarks/linear/ITG_w7x/wout_w7x.nc",
            revision="bc2fe5523c23e3d0198181a3e3b7c8a482e25ba5",
        ),
        request,
    )
    physical = result.physical
    zeta = np.asarray(physical.phi)

    def sampled(values):
        return np.interp(zeta_reference, zeta, values)

    rho = request.radial_value
    bmag = np.asarray(physical.B)
    dxdpsit = -1.0 / rho
    direct = {
        "bmag": bmag,
        "g_xx": np.asarray(physical.grad_psi_sq) * dxdpsit**2,
        "g_xy": np.asarray(physical.grad_psi_dot_grad_alpha) * dxdpsit * rho,
        "g_yy": np.asarray(physical.grad_alpha_sq) * rho**2,
        "B_cross_gradB_dot_grad_alpha": (
            np.asarray(physical.B_cross_gradB_dot_grad_alpha) / bmag**3 * rho
        ),
        "b_cross_kappa_dot_grad_alpha": (
            np.asarray(physical.b_cross_kappa_dot_grad_alpha) / bmag * rho
        ),
        "B_cross_gradB_dot_grad_psi": (
            np.asarray(physical.B_cross_gradB_dot_grad_psi) / bmag**3 * dxdpsit
        ),
    }
    relative_errors = {
        name: float(
            np.linalg.norm(sampled(values) - stella.column(name))
            / np.linalg.norm(stella.column(name))
        )
        for name, values in direct.items()
    }
    scales = {
        name: float(
            np.dot(sampled(values), stella.column(name))
            / np.dot(stella.column(name), stella.column(name))
        )
        for name, values in direct.items()
    }
    assert relative_errors["bmag"] < 1.0e-2
    assert max(relative_errors.values()) < 3.0e-1, relative_errors
    assert all(0.85 < scale < 1.15 for scale in scales.values()), scales

    zed = stella.column("zed")
    dzed_dzeta = np.gradient(zed, zeta_reference)
    expected_parallel = stella.column("b_dot_grad_zed") / dzed_dzeta
    np.testing.assert_allclose(
        sampled(np.asarray(physical.b_dot_grad_z)),
        expected_parallel,
        rtol=3.0e-2,
        atol=3.0e-2,
    )
    assert result.metadata.nfp == 5
    assert result.metadata.endpoint_policy == "exclude"


@pytest.mark.external
def test_direct_vmec_w7x_matches_same_surface_gx_gist_terms(gx_root):
    eik = load_eik_data(
        gx_root
        / "geometry_modules/vmec/tests/"
        "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
    )
    stride = 10
    theta = eik.theta[:-1:stride]
    q = eik.header[4]
    iota = 1.0 / q
    rho = np.sqrt(eik.header[5])
    request = GeometryRequest(
        configuration="gx-w7x-standard-same-surface",
        radial_value=rho,
        alpha=0.0,
        n_z=theta.size,
        z_min=float(theta[0] / iota),
        z_max=float(eik.theta[-1] / iota),
        field_periods=float((eik.theta[-1] - eik.theta[0]) / (2.0 * np.pi * iota)),
    )
    result = resolve_geometry(
        VmecppGeometryProvider(
            wout_path=gx_root / "geometry_modules/vmec/tests/wout_w7x_standardConfig.nc",
            revision="bc2fe5523c23e3d0198181a3e3b7c8a482e25ba5",
        ),
        request,
    )
    physical = result.physical
    B = np.asarray(physical.B)
    s = rho**2
    sqrt_s = rho
    direct_shat = float(physical.shear)
    shat = eik.header[2]
    sign_psi = 1.0
    direct = {
        "bmag": B,
        "gradpar": -iota * np.asarray(physical.b_dot_grad_z),
        "gds2": s * np.asarray(physical.grad_alpha_sq),
        "gds21": -shat * np.asarray(physical.grad_psi_dot_grad_alpha),
        "gds22": shat**2 / s * np.asarray(physical.grad_psi_sq),
        "gbdrift": (
            -sign_psi
            * 2.0
            * sqrt_s
            * np.asarray(physical.B_cross_gradB_dot_grad_alpha)
            / B**3
        ),
        "gbdrift0": (
            -sign_psi
            * 2.0
            * shat
            / sqrt_s
            * np.asarray(physical.B_cross_gradB_dot_grad_psi)
            / B**3
        ),
        "cvdrift": (
            -sign_psi
            * 2.0
            * sqrt_s
            * np.asarray(physical.b_cross_kappa_dot_grad_alpha)
            / B
        ),
        "cvdrift0": (
            -sign_psi
            * 2.0
            * shat
            / sqrt_s
            * np.asarray(physical.b_cross_kappa_dot_grad_psi)
            / B
        ),
    }
    relative_errors = {
        name: float(
            np.linalg.norm(values - np.asarray(getattr(eik, name))[:-1:stride])
            / np.linalg.norm(np.asarray(getattr(eik, name))[:-1:stride])
        )
        for name, values in direct.items()
    }
    scales = {
        name: float(
            np.dot(values, np.asarray(getattr(eik, name))[:-1:stride])
            / np.dot(
                np.asarray(getattr(eik, name))[:-1:stride],
                np.asarray(getattr(eik, name))[:-1:stride],
            )
        )
        for name, values in direct.items()
    }
    assert relative_errors["bmag"] < 1.0e-2
    assert relative_errors["gradpar"] < 1.0e-1
    assert max(relative_errors.values()) < 2.0e-1, relative_errors
    assert all(0.6 < scale < 1.4 for scale in scales.values()), scales
    assert result.metadata.nfp == 5
    assert result.metadata.endpoint_policy == "exclude"
    assert np.isclose(result.metadata.field_periods, request.field_periods)
    assert np.isclose(direct_shat, shat, rtol=2.0e-1)
