from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/bootstrap_dependencies.py"
    spec = importlib.util.spec_from_file_location("bootstrap_dependencies", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dependency_manifest_has_pinned_profiles():
    module = _module()
    manifest = module.load_manifest(ROOT / "dependencies.toml")

    assert manifest.schema_version == 1
    assert manifest.profiles["mhd"] == ("vmecpp", "desc", "gvec")
    assert manifest.profiles["validation-native"] == ("gx", "stella", "gkw")
    assert all(len(item.revision) == 40 for item in manifest.dependencies.values())
    assert manifest.dependencies["vmecpp"].install == "python"
    assert dict(manifest.dependencies["vmecpp"].macos_environment) == {
        "OpenMP_ROOT": "{brew_prefix}/opt/libomp"
    }
    assert dict(manifest.dependencies["desc"].environment) == {
        "PYTHONPATH": "{root}/scripts/dependency_compat/desc"
    }
    assert manifest.dependencies["gx"].install == "native"
    assert manifest.dependencies["stella"].build[0][-1] == ("{build_dir}/COMPILATION/build_cmake")


def test_profile_resolution_is_ordered_and_deduplicated():
    module = _module()
    manifest = module.load_manifest(ROOT / "dependencies.toml")

    selected = module.resolve_dependencies(
        manifest,
        ("mhd", "validation-python"),
        ("vmecpp",),
    )
    assert tuple(item.name for item in selected) == ("vmecpp", "desc", "gvec", "gyaradax")

    with pytest.raises(ValueError, match="unknown dependency profile"):
        module.resolve_dependencies(manifest, ("missing",), ())


def test_native_only_skip_project_does_not_require_python_validation():
    module = _module()
    manifest = module.load_manifest(ROOT / "dependencies.toml")

    assert not module.python_environment_changes(
        (manifest.dependencies["stella"],), skip_project=True, fetch_only=False
    )
    assert module.python_environment_changes(
        (manifest.dependencies["desc"],), skip_project=True, fetch_only=False
    )
    assert module.python_environment_changes(
        (manifest.dependencies["stella"],), skip_project=False, fetch_only=False
    )
    assert not module.python_environment_changes(
        (manifest.dependencies["desc"],), skip_project=False, fetch_only=True
    )


def test_dry_run_renders_native_build_without_writing(tmp_path, capsys):
    module = _module()
    manifest = module.load_manifest(ROOT / "dependencies.toml")
    dependency = manifest.dependencies["stella"]
    bootstrapper = module.Bootstrapper(
        deps_dir=tmp_path / "deps",
        local_root=None,
        python=Path("/usr/bin/python3"),
        jobs=3,
        dry_run=True,
        fetch_only=False,
        editable=False,
        skip_project=True,
    )

    bootstrapper.prepare(dependency)
    bootstrapper.finish()

    output = capsys.readouterr().out
    assert "git clone" in output
    assert "cmake --build" in output
    assert "-j 3" in output
    assert not (tmp_path / "deps").exists()


def test_python_environment_path_is_not_resolved_through_symlink(tmp_path):
    module = _module()
    base_python = tmp_path / "base-python"
    base_python.touch()
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(base_python)

    bootstrapper = module.Bootstrapper(
        deps_dir=tmp_path / "deps",
        local_root=None,
        python=venv_python,
        jobs=1,
        dry_run=True,
        fetch_only=False,
        editable=False,
        skip_project=True,
    )

    assert bootstrapper.python == venv_python.absolute()
    assert bootstrapper.python != base_python.resolve()


def test_project_sync_retains_prepared_external_profile(tmp_path, capsys):
    module = _module()
    bootstrapper = module.Bootstrapper(
        deps_dir=tmp_path / "deps",
        local_root=None,
        python=Path("/usr/bin/python3"),
        jobs=1,
        dry_run=True,
        fetch_only=False,
        editable=False,
        skip_project=False,
    )

    bootstrapper.prepare_project()
    bootstrapper.verify_environment()

    output = capsys.readouterr().out
    assert "uv sync --extra dev --inexact" in output
    assert "uv pip check --python /usr/bin/python3" in output
