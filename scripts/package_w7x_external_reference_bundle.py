"""Package the W7-X GX external-reference run bundle.

The bundle is meant to be transferred to a CUDA/NVIDIA-capable GX machine.  It
contains the patched GX input, workflow metadata, README, handoff scripts, and
the VMEC file when available.  A SHA-256 manifest is embedded in the archive so
the returned artifacts can be tied to the exact handoff inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "fixtures/gx_w7x_mode_structure_run/mode_structure_run_metadata.json"
DEFAULT_OUTPUT = ROOT / "fixtures/gx_w7x_mode_structure_run/w7x_external_reference_bundle.tar.gz"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = package_w7x_external_reference_bundle(
        metadata_path=args.metadata,
        output_path=args.output,
        include_vmec=not args.no_vmec,
    )
    print(f"wrote {args.output}")
    print(f"files={len(manifest['files'])}")
    return 0


def package_w7x_external_reference_bundle(
    *,
    metadata_path: Path = DEFAULT_METADATA,
    output_path: Path = DEFAULT_OUTPUT,
    include_vmec: bool = True,
) -> dict[str, object]:
    """Write a tarball and return its manifest."""

    metadata = _load_json(metadata_path)
    files = _bundle_files(metadata, metadata_path, include_vmec=include_vmec)
    manifest = {
        "benchmark_name": "w7x_itg_external_reference_bundle",
        "metadata": _display_path(metadata_path),
        "output": _display_path(output_path),
        "include_vmec": bool(include_vmec),
        "requires_repository_root": True,
        "unpack_command": (
            "tar -xzf /tmp/w7x_external_reference_bundle.tar.gz "
            "-C /path/to/new-plasma-code"
        ),
        "files": [_file_record(source, arcname) for source, arcname in files],
        "external_run_command": (
            "GX_EXECUTABLE=/path/to/gx bash "
            "fixtures/gx_w7x_mode_structure_run/run_external_reference.sh"
        ),
        "returned_outputs_ingest_command": (
            "bash fixtures/gx_w7x_mode_structure_run/ingest_returned_outputs.sh "
            "--copy-outputs --resample-reference-to-observed-z"
        ),
        "post_parity_timing_command": (
            "bash fixtures/gx_w7x_mode_structure_run/run_production_timing_after_parity.sh"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        for source, arcname in files:
            archive.add(source, arcname=arcname)
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo("BUNDLE_MANIFEST.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
    return manifest


def _bundle_files(
    metadata: dict[str, object],
    metadata_path: Path,
    *,
    include_vmec: bool,
) -> list[tuple[Path, str]]:
    run_dir = _resolve(metadata["prepared_input"]).parent
    candidates = [
        _resolve(metadata["prepared_input"]),
        metadata_path,
        run_dir / "README.md",
        run_dir / "run_external_reference.sh",
        run_dir / "ingest_returned_outputs.sh",
        run_dir / "run_production_timing_after_parity.sh",
    ]
    if include_vmec:
        candidates.append(_resolve(metadata["vmec_source"]))
    files: list[tuple[Path, str]] = []
    for source in candidates:
        if not source.exists():
            raise FileNotFoundError(source)
        arcname = _arcname_for(source, run_dir, metadata)
        files.append((source, arcname))
    return files


def _arcname_for(source: Path, run_dir: Path, metadata: dict[str, object]) -> str:
    vmec_source = _resolve(metadata["vmec_source"])
    vmec_destination = Path(str(metadata["vmec_destination"]))
    if source.resolve() == vmec_source.resolve():
        if vmec_destination.is_absolute():
            return str(Path("fixtures/gx_w7x_mode_structure_run") / vmec_destination.name)
        return str(vmec_destination)
    try:
        return str(source.resolve().relative_to(ROOT))
    except ValueError:
        try:
            return str(run_dir.relative_to(ROOT) / source.name)
        except ValueError:
            return str(Path("fixtures/gx_w7x_mode_structure_run") / source.name)


def _file_record(source: Path, arcname: str) -> dict[str, object]:
    data = source.read_bytes()
    return {
        "path": _display_path(source),
        "archive_path": arcname,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _resolve(path_like) -> Path:
    path = Path(str(path_like))
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-vmec", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
