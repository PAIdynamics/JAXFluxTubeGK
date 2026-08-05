import numpy as np
import pytest
import shutil

from stellarator_gk import (
    GVEC_GEOMETRY_COMPUTE_KEYS,
    GeometryRequest,
    GvecGeometryProvider,
    VmecppGeometryProvider,
    gvec_geometry_arrays_from_data,
    internal_geometry_from_result,
    resolve_geometry,
)


def _gvec_data(n_z=12):
    zeta = np.linspace(-0.5, 0.5, n_z, endpoint=False)
    B = 1.3 + 0.05 * np.cos(zeta)
    return {
        "iota": 0.82,
        "diota_dr": 0.12,
        "dPhi_dr": 1.7,
        "Phi_edge": 1.0,
        "chi_edge": 0.8,
        "r_minor": 1.0,
        "grad_rho": np.tile([1.0, 0.0, 0.0], (n_z, 1)),
        "grad_theta_P": np.tile([0.1, 1.0, 0.0], (n_z, 1)),
        "grad_zeta": np.tile([0.0, 0.0, 0.4], (n_z, 1)),
        "B": np.column_stack((np.zeros(n_z), np.zeros(n_z), B)),
        "mod_B": B,
        "grad_mod_B": np.column_stack(
            (0.03 * np.ones(n_z), 0.02 * np.sin(zeta), 0.01 * np.ones(n_z))
        ),
        "kappa_B": np.column_stack(
            (0.1 * np.cos(zeta), 0.05 * np.sin(zeta), np.zeros(n_z))
        ),
    }


class _FakeGvecState:
    nfp = 5

    def __init__(self, data):
        self.data = data
        self.compute_calls = []

    def evaluate(self, *names, **_coordinates):
        if names == ("iota",):
            return {"iota": self.data["iota"]}
        assert names == ("Phi", "chi")
        return {"Phi": self.data["Phi_edge"], "chi": self.data["chi_edge"]}

    def compute(self, evaluated, *names):
        self.compute_calls.append(names)
        evaluated.update(self.data)


def _request():
    return GeometryRequest(
        configuration="gvec-test",
        radial_value=0.6,
        alpha=0.15,
        n_z=12,
        z_min=-0.5,
        z_max=0.5,
    )


def test_gvec_data_mapping_produces_physical_arrays():
    request = _request()
    zeta = np.linspace(request.z_min, request.z_max, request.n_z, endpoint=False)
    arrays = gvec_geometry_arrays_from_data(
        _gvec_data(request.n_z),
        zeta=zeta,
        rho=request.radial_value,
        alpha=request.alpha,
    )

    np.testing.assert_allclose(arrays["theta"] - arrays["iota"] * arrays["phi"], 0.15)
    np.testing.assert_allclose(arrays["grad_psi_sq"], (1.7 / 2.0) ** 2)
    np.testing.assert_allclose(arrays["B"], _gvec_data(request.n_z)["mod_B"] / 2.0)
    assert arrays["B"].shape == (12,)
    assert all(np.all(np.isfinite(np.asarray(value))) for value in arrays.values())


def test_gvec_provider_uses_pest_evaluation_and_common_contract():
    state = _FakeGvecState(_gvec_data())
    calls = []

    def evaluations_pest(**kwargs):
        calls.append(kwargs)
        return {}

    result = resolve_geometry(
        GvecGeometryProvider(
            state=state,
            evaluations_pest=evaluations_pest,
            provider_version="test",
            revision="fake-gvec-revision",
        ),
        _request(),
    )
    geometry = internal_geometry_from_result(result)

    assert state.compute_calls == [GVEC_GEOMETRY_COMPUTE_KEYS]
    assert calls[0]["state"] is state
    assert calls[0]["theta_P"].shape == (1, 1, 12)
    assert result.metadata.provenance.provider == "gvec"
    assert result.metadata.provenance.revision == "fake-gvec-revision"
    assert result.metadata.normalization.length_reference.startswith("GVEC")
    assert "Phi_edge" in result.metadata.normalization.magnetic_field_reference
    assert result.physical.nfp == 5
    assert geometry.B.shape == (12,)
    assert np.all(np.isfinite(np.asarray(geometry.D_x)))


def test_gvec_provider_loads_explicit_parameter_and_state_files():
    state = _FakeGvecState(_gvec_data())
    calls = []

    def state_factory(parameter_file, state_file):
        calls.append((parameter_file, state_file))
        return state

    provider = GvecGeometryProvider(
        parameter_file="parameter.ini",
        state_file="equilibrium_state.dat",
        state_factory=state_factory,
        evaluations_pest=lambda **_kwargs: {},
    )
    result = resolve_geometry(provider, _request())

    assert calls == [("parameter.ini", "equilibrium_state.dat")]
    assert result.metadata.provenance.source == "GVEC state parameter.ini"


def test_gvec_provider_reports_missing_and_ambiguous_inputs():
    with pytest.raises(ValueError, match="exactly one"):
        GvecGeometryProvider()
    with pytest.raises(ValueError, match="exactly one"):
        GvecGeometryProvider(state=object(), parameter_file="parameter.ini")

    data = _gvec_data()
    data.pop("kappa_B")
    with pytest.raises(KeyError, match="kappa_B"):
        gvec_geometry_arrays_from_data(
            data,
            zeta=np.linspace(-0.5, 0.5, 12, endpoint=False),
            rho=0.6,
            alpha=0.0,
        )


@pytest.mark.external
def test_installed_gvec_evaluates_live_pest_field_line(gvec_root):
    parameter_file = gvec_root / "test-CI/examples/ellipstell_lowres/parameter.ini"
    result = resolve_geometry(
        GvecGeometryProvider(
            parameter_file=parameter_file,
            revision="08ea5bd54e08572770e560d9ca88fe594bc8fd01",
        ),
        _request(),
    )

    assert result.metadata.provenance.provider == "gvec"
    assert result.metadata.provenance.revision == "08ea5bd54e08572770e560d9ca88fe594bc8fd01"
    assert result.physical.nfp == 2
    assert np.all(np.asarray(result.physical.B) > 0.0)


@pytest.mark.external
def test_gvec_matches_vmec_on_common_w7x_equilibrium(
    gvec_root, vmecpp_root, tmp_path
):
    del vmecpp_root  # The fixture verifies the installed VMEC++ revision.
    case = gvec_root / "test-CI/examples/w7x_from_vmec_initLA_F"
    parameter_file = shutil.copy2(case / "parameter.ini", tmp_path)
    wout_path = shutil.copy2(case / "wout_d23p4_tm.nc", tmp_path)
    request = GeometryRequest(
        configuration="w7x-common-equilibrium",
        radial_value=0.6,
        alpha=0.0,
        n_z=64,
        z_min=-np.pi / 5.0,
        z_max=np.pi / 5.0,
    )
    vmec = resolve_geometry(
        VmecppGeometryProvider(wout_path=wout_path), request
    )
    gvec = resolve_geometry(
        GvecGeometryProvider(parameter_file=parameter_file), request
    )

    relative_errors = {}
    scale_factors = {}
    for name in (
        "B",
        "b_dot_grad_z",
        "grad_psi_sq",
        "grad_alpha_sq",
        "grad_psi_dot_grad_alpha",
        "B_cross_gradB_dot_grad_psi",
        "B_cross_gradB_dot_grad_alpha",
        "b_cross_kappa_dot_grad_psi",
        "b_cross_kappa_dot_grad_alpha",
    ):
        actual = np.asarray(getattr(gvec.physical, name))
        expected = np.asarray(getattr(vmec.physical, name))
        relative_errors[name] = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        scale_factors[name] = np.dot(actual, expected) / np.dot(expected, expected)

    # This GVEC case is the finite-element initialization projected from the
    # exact same wout, before a GVEC minimization. Low-order invariants should
    # agree tightly; the looser all-term bound records projection error while
    # still detecting coordinate/sign/normalization regressions.
    assert relative_errors["B"] < 0.08
    assert relative_errors["b_dot_grad_z"] < 0.05
    assert relative_errors["grad_psi_sq"] < 0.08
    assert max(relative_errors.values()) < 0.75
    assert all(0.2 < scale < 1.3 for scale in scale_factors.values())
    np.testing.assert_allclose(gvec.physical.iota, vmec.physical.iota, rtol=1.0e-10)
    np.testing.assert_allclose(gvec.physical.shear, vmec.physical.shear, rtol=5.0e-6)
