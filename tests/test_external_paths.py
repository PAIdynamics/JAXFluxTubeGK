from pathlib import Path

from stellarator_gk.external import announce_external_path, external_path_provenance


def test_external_path_provenance_reports_repository_revision(capsys):
    root = Path(__file__).resolve().parents[1]
    provenance = external_path_provenance(root / "dependencies.toml")

    assert provenance.path == (root / "dependencies.toml").resolve()
    assert provenance.git_root == root
    assert provenance.revision is not None
    assert len(provenance.revision) == 40

    announced = announce_external_path("manifest", root / "dependencies.toml")
    output = capsys.readouterr().out
    assert announced == provenance
    assert f"external manifest: {provenance.path} @ {provenance.revision}" in output


def test_external_path_provenance_marks_unversioned_path(tmp_path, capsys):
    path = tmp_path / "input.dat"
    path.write_text("synthetic")

    provenance = announce_external_path("input", path)

    assert provenance.git_root is None
    assert provenance.revision is None
    assert "@ unversioned" in capsys.readouterr().out
