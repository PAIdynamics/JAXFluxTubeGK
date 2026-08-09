from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def _gkw_contracts() -> list[Path]:
    return sorted(
        path
        for path in FIXTURES.rglob("*")
        if path.is_file()
        and path.relative_to(FIXTURES).parts[0].startswith("gkw_")
    )


def test_gkw_contracts_remain_compact_selected_slices():
    contracts = _gkw_contracts()
    assert contracts
    sizes = [path.stat().st_size for path in contracts]
    assert sum(sizes) <= 200_000
    assert max(sizes) <= 64 * 1024


def test_fixture_tree_contains_no_archival_native_run_directories():
    forbidden = {"raw", "archive", "archives"}
    offenders = [
        path.relative_to(FIXTURES)
        for path in FIXTURES.rglob("*")
        if forbidden.intersection(part.lower() for part in path.parts)
    ]
    assert offenders == []
