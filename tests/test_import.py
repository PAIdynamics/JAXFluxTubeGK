def test_import_smoke():
    import stellarator_gk

    assert stellarator_gk.__version__
    assert hasattr(stellarator_gk, "build_velocity_grid")


def test_core_import_does_not_load_benchmark_implementation():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, stellarator_gk; "
            "assert 'stellarator_gk.benchmarks' not in sys.modules; "
            "assert 'CycloneTrace' not in stellarator_gk.__all__",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_validation_namespace_and_legacy_attribute_are_lazy_compatible():
    from stellarator_gk.validation.fixture_io import PerKyModeStructureFixture

    assert PerKyModeStructureFixture.__module__ == "stellarator_gk.benchmarks"

    from stellarator_gk import CycloneTrace

    assert CycloneTrace.__module__ == "stellarator_gk.benchmarks"
