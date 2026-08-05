from types import SimpleNamespace

import numpy as np
import pytest

from stellarator_gk import (
    GeometryRequest,
    VmecppGeometryProvider,
    internal_geometry_from_result,
    resolve_geometry,
    vmec_field_line_from_wout,
)


def _circular_wout(*, lasym=False):
    ns = 9
    s = np.linspace(0.0, 1.0, ns)
    radius = 0.5 * np.sqrt(s)
    return SimpleNamespace(
        lasym=lasym,
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
        phi=-2.0 * np.pi * s,
        presf=2.0e3 * (1.0 - s),
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
    np.testing.assert_allclose(arrays["B"], 1.2)
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
