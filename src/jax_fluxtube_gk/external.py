"""Explicit external-path provenance helpers for command-line workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class ExternalPathProvenance:
    path: Path
    git_root: Path | None
    revision: str | None


def external_path_provenance(path: str | Path) -> ExternalPathProvenance:
    resolved = Path(path).expanduser().resolve()
    anchor = resolved if resolved.is_dir() else resolved.parent
    result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel", "HEAD"),
        cwd=anchor,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) != 2:
        return ExternalPathProvenance(resolved, None, None)
    return ExternalPathProvenance(resolved, Path(lines[0]).resolve(), lines[1])


def announce_external_path(label: str, path: str | Path) -> ExternalPathProvenance:
    provenance = external_path_provenance(path)
    revision = provenance.revision or "unversioned"
    print(f"external {label}: {provenance.path} @ {revision}")
    return provenance
