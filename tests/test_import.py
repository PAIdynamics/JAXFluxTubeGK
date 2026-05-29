def test_import_smoke():
    import stellarator_gk

    assert stellarator_gk.__version__
    assert hasattr(stellarator_gk, "build_velocity_grid")

