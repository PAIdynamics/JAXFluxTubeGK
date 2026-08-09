def test_import_smoke():
    import jax_fluxtube_gk

    assert jax_fluxtube_gk.__version__
    assert hasattr(jax_fluxtube_gk, "build_velocity_grid")


def test_core_import_does_not_load_benchmark_implementation():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, jax_fluxtube_gk; "
            "assert 'jax_fluxtube_gk.benchmarks' not in sys.modules; "
            "assert 'CycloneTrace' not in jax_fluxtube_gk.__all__",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_validation_namespace_and_legacy_attribute_are_lazy_compatible():
    from jax_fluxtube_gk.validation.fixture_io import PerKyModeStructureFixture

    assert PerKyModeStructureFixture.__module__ == "jax_fluxtube_gk.benchmarks"

    from jax_fluxtube_gk import CycloneTrace

    assert CycloneTrace.__module__ == "jax_fluxtube_gk.benchmarks"


def test_validation_namespace_import_is_itself_lazy():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, jax_fluxtube_gk.validation; "
            "assert 'jax_fluxtube_gk.benchmarks' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
