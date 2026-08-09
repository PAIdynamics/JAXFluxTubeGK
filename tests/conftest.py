from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tomllib

import jax
import pytest


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("external dependencies")
    for name in ("gyaradax", "gx", "desc", "vmecpp", "gvec", "stella", "gkw"):
        group.addoption(
            f"--{name}-root",
            type=Path,
            help=(
                f"path to the revision-pinned {name} checkout; alternatively set "
                f"JAX_FLUXTUBE_GK_{name.upper()}_ROOT"
            ),
        )


def _dependency_root(request: pytest.FixtureRequest, name: str) -> Path:
    option = request.config.getoption(f"--{name}-root")
    configured = option or os.environ.get(f"JAX_FLUXTUBE_GK_{name.upper()}_ROOT")
    if configured is None:
        pytest.fail(
            f"external {name} test requires --{name}-root or "
            f"JAX_FLUXTUBE_GK_{name.upper()}_ROOT"
        )
    root = Path(configured).expanduser().resolve()
    if not (root / ".git").exists():
        pytest.fail(f"configured {name} root is not a Git checkout: {root}")

    with (ROOT / "dependencies.toml").open("rb") as handle:
        dependency = tomllib.load(handle)["dependencies"][name]
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    expected = dependency["revision"]
    if revision != expected:
        pytest.fail(f"configured {name} root is at {revision}, expected {expected}: {root}")

    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"external {name}: {root} @ {revision}")
    return root


@pytest.fixture
def gyaradax_root(request: pytest.FixtureRequest) -> Path:
    return _dependency_root(request, "gyaradax")


@pytest.fixture
def gx_root(request: pytest.FixtureRequest) -> Path:
    return _dependency_root(request, "gx")


@pytest.fixture
def desc_root(request: pytest.FixtureRequest) -> Path:
    return _dependency_root(request, "desc")


@pytest.fixture
def vmecpp_root(request: pytest.FixtureRequest) -> Path:
    return _dependency_root(request, "vmecpp")


@pytest.fixture
def gvec_root(request: pytest.FixtureRequest) -> Path:
    return _dependency_root(request, "gvec")


@pytest.fixture
def stella_root(request: pytest.FixtureRequest) -> Path:
    return _dependency_root(request, "stella")


@pytest.fixture
def gkw_root(request: pytest.FixtureRequest) -> Path:
    return _dependency_root(request, "gkw")
