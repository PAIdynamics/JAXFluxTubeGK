from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "examples/vmecpp_w7x_design_loop.py"
    spec = importlib.util.spec_from_file_location("vmecpp_w7x_design_loop", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input():
    rbc = np.zeros((3, 3))
    zbs = np.zeros((3, 3))
    rbc[1, 1] = 0.5
    zbs[1, 1] = 0.5
    return SimpleNamespace(mpol=3, ntor=1, rbc=rbc, zbs=zbs)


def _wout(boundary_scale):
    ns = 9
    s = np.linspace(0.0, 1.0, ns)
    radius = (0.45 + 0.02 * boundary_scale) * np.sqrt(s)
    return SimpleNamespace(
        lasym=False,
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
        bmnc=(1.2 + 0.01 * boundary_scale) * np.ones((1, ns)),
        iotaf=0.7 + 0.04 * s,
        iotas=0.7 + 0.04 * s,
        phi=-2.0 * np.pi * s,
        presf=2.0e3 * (1.0 - s),
        pres=2.0e3 * (1.0 - s),
        Aminor_p=0.5,
    )


def test_reduced_vmecpp_design_loop_runs_fresh_provider_solves(tmp_path):
    module = _load_module()
    calls = []

    def named_loader(name):
        calls.append(("load", name))
        return _input()

    def runner(vmec_input, *, max_threads, verbose):
        scale = vmec_input.rbc[1, 1] / 0.5
        calls.append(("run", float(scale), max_threads, verbose))
        return SimpleNamespace(wout=_wout(scale))

    output = tmp_path / "design_loop.json"
    args = module._parse_args(
        [
            "--output",
            str(output),
            "--iterations",
            "1",
            "--n-z",
            "8",
            "--n-vpar",
            "3",
            "--n-mu",
            "3",
            "--n-steps",
            "1",
        ]
    )
    report = module.run_vmecpp_w7x_design_loop(
        args,
        named_loader=named_loader,
        runner=runner,
    )

    assert report["passed"]
    assert report["status"] == module.STATUS
    assert report["claims"]["real_mhd_provider_loop"]
    assert not report["claims"]["end_to_end_mhd_autodiff"]
    assert len([call for call in calls if call[0] == "run"]) == 3
    assert np.isfinite(report["iterations"][0]["finite_difference_gradient"])
    assert json.loads(output.read_text())["topology"]["provider_topology"][1] == 5


def test_vmecpp_design_loop_refuses_repository_artifacts():
    module = _load_module()
    args = module._parse_args(["--output", str(ROOT / "runs/design.json")])

    with pytest.raises(ValueError, match="outside the repository"):
        module.run_vmecpp_w7x_design_loop(
            args,
            named_loader=lambda _name: _input(),
            runner=lambda *_args, **_kwargs: None,
        )
