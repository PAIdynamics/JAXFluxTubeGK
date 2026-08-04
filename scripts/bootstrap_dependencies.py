"""Fetch, build, and install pinned optional dependencies for optimal-fusion."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "dependencies.toml"
DEFAULT_DEPS_DIR = ROOT / ".dependencies"


@dataclass(frozen=True)
class Dependency:
    name: str
    role: str
    url: str
    revision: str
    source_dir: str
    install: str
    import_name: str | None = None
    build: tuple[tuple[str, ...], ...] = ()
    executables: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    macos_environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    profiles: dict[str, tuple[str, ...]]
    dependencies: dict[str, Dependency]


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Manifest:
    """Load and validate the dependency manifest."""

    with path.open("rb") as handle:
        data = tomllib.load(handle)
    schema_version = int(data.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError(f"unsupported dependency manifest schema {schema_version}")

    raw_dependencies = data.get("dependencies", {})
    dependencies: dict[str, Dependency] = {}
    for name, raw in raw_dependencies.items():
        revision = str(raw["revision"])
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError(f"dependency {name!r} must use a full lowercase Git commit")
        install = str(raw["install"])
        if install not in ("python", "native"):
            raise ValueError(f"dependency {name!r} has unsupported install mode {install!r}")
        dependencies[name] = Dependency(
            name=name,
            role=str(raw["role"]),
            url=str(raw["url"]),
            revision=revision,
            source_dir=str(raw.get("source_dir", name)),
            install=install,
            import_name=raw.get("import_name"),
            build=tuple(tuple(str(token) for token in command) for command in raw.get("build", ())),
            executables=tuple(str(value) for value in raw.get("executables", ())),
            environment=tuple(
                (str(key), str(value)) for key, value in raw.get("environment", {}).items()
            ),
            macos_environment=tuple(
                (str(key), str(value))
                for key, value in raw.get("macos_environment", {}).items()
            ),
        )

    profiles = {
        name: tuple(str(item) for item in members)
        for name, members in data.get("profiles", {}).items()
    }
    for profile, members in profiles.items():
        unknown = sorted(set(members) - dependencies.keys())
        if unknown:
            raise ValueError(f"profile {profile!r} contains unknown dependencies: {unknown}")
    return Manifest(schema_version, profiles, dependencies)


def resolve_dependencies(
    manifest: Manifest,
    profiles: Sequence[str],
    explicit: Sequence[str],
) -> tuple[Dependency, ...]:
    """Resolve profile and explicit dependency names in deterministic order."""

    names: list[str] = []
    for profile in profiles:
        if profile not in manifest.profiles:
            raise ValueError(f"unknown dependency profile {profile!r}")
        names.extend(manifest.profiles[profile])
    names.extend(explicit)
    unknown = sorted(set(names) - manifest.dependencies.keys())
    if unknown:
        raise ValueError(f"unknown dependencies: {unknown}")
    return tuple(manifest.dependencies[name] for name in dict.fromkeys(names))


class Bootstrapper:
    """Execute dependency preparation without mutating user-owned local clones."""

    def __init__(
        self,
        *,
        deps_dir: Path,
        local_root: Path | None,
        python: Path,
        jobs: int,
        dry_run: bool,
        fetch_only: bool,
        editable: bool,
        skip_project: bool,
    ) -> None:
        self.deps_dir = deps_dir.resolve()
        self.source_root = self.deps_dir / "src"
        self.build_root = self.deps_dir / "build"
        self.bin_dir = self.deps_dir / "bin"
        self.local_root = local_root.resolve() if local_root else None
        # Keep the environment path itself. Resolving a venv's Python symlink
        # would incorrectly target the base interpreter for `uv pip install`.
        self.python = python.absolute()
        self.jobs = jobs
        self.dry_run = dry_run
        self.fetch_only = fetch_only
        self.editable = editable
        self.skip_project = skip_project
        self.state: dict[str, object] = {"schema_version": 1, "dependencies": {}}

    def prepare_project(self) -> None:
        if self.skip_project:
            return
        # Keep an already prepared external profile on repeat runs. A default
        # exact sync would remove every provider before reinstalling it.
        self._run(("uv", "sync", "--extra", "dev", "--inexact"), cwd=ROOT)

    def prepare(self, dependency: Dependency) -> None:
        source, managed = self._resolve_source(dependency)
        if managed:
            self._prepare_managed_clone(dependency, source)
        else:
            self._verify_local_clone(dependency, source)

        commands: list[list[str]] = []
        dependency_environment = (
            {} if self.fetch_only else self._dependency_environment(dependency)
        )
        if not self.fetch_only:
            if dependency.install == "python":
                command = [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(self.python),
                ]
                if self.editable:
                    command.append("--editable")
                command.append(str(source))
                self._run(command, cwd=ROOT, extra_env=dependency_environment)
                commands.append(command)
                if dependency.import_name:
                    verify = [
                        str(self.python),
                        "-c",
                        (
                            "import importlib; "
                            f"importlib.import_module({dependency.import_name!r})"
                        ),
                    ]
                    self._run(verify, cwd=ROOT)
                    commands.append(verify)
            else:
                build_dir = self.build_root / dependency.name
                for template in dependency.build:
                    command = self._render(template, build_dir)
                    self._run(command, cwd=source, extra_env=dependency_environment)
                    commands.append(command)
                self._install_native_executable(dependency, source, build_dir)

        self.state["dependencies"][dependency.name] = {
            "role": dependency.role,
            "revision": dependency.revision,
            "source": str(source),
            "managed_clone": managed,
            "install": dependency.install,
            "commands": commands,
            "environment": dependency_environment,
        }

    def finish(self) -> None:
        if self.dry_run:
            return
        self.deps_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.deps_dir / "state.json"
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        temporary.replace(state_path)

    def verify_environment(self) -> None:
        """Reject an installation with incompatible Python package versions."""

        if self.fetch_only:
            return
        self._run(
            ("uv", "pip", "check", "--python", str(self.python)),
            cwd=ROOT,
        )

    def _resolve_source(self, dependency: Dependency) -> tuple[Path, bool]:
        if self.local_root is not None:
            candidate = self.local_root / dependency.source_dir
            if candidate.is_dir():
                return candidate.resolve(), False
        return (self.source_root / dependency.name).resolve(), True

    def _prepare_managed_clone(self, dependency: Dependency, source: Path) -> None:
        if not source.exists():
            if not self.dry_run:
                source.parent.mkdir(parents=True, exist_ok=True)
            self._run(
                ("git", "clone", "--filter=blob:none", "--no-checkout", dependency.url, str(source)),
                cwd=ROOT,
            )
        elif not (source / ".git").exists():
            raise RuntimeError(f"managed dependency path is not a Git clone: {source}")
        if source.exists() and (source / ".git").exists():
            dirty = self._capture(("git", "status", "--porcelain"), cwd=source)
            if dirty.strip():
                raise RuntimeError(f"managed dependency clone is dirty: {source}")
        self._run(("git", "fetch", "origin", dependency.revision), cwd=source)
        self._run(("git", "checkout", "--detach", dependency.revision), cwd=source)

    def _verify_local_clone(self, dependency: Dependency, source: Path) -> None:
        if not (source / ".git").exists():
            raise RuntimeError(f"local dependency is not a Git clone: {source}")
        revision = self._capture(("git", "rev-parse", "HEAD"), cwd=source).strip()
        if revision != dependency.revision:
            raise RuntimeError(
                f"local {dependency.name} is at {revision}, expected {dependency.revision}; "
                "the bootstrapper will not checkout a user-owned clone"
            )

    def _install_native_executable(
        self,
        dependency: Dependency,
        source: Path,
        build_dir: Path,
    ) -> None:
        if self.dry_run:
            for candidate in dependency.executables:
                print(f"[dry-run] locate {self._render_path(candidate, source, build_dir)}")
            return
        for candidate in dependency.executables:
            executable = self._render_path(candidate, source, build_dir)
            if executable.is_file() and os.access(executable, os.X_OK):
                self.bin_dir.mkdir(parents=True, exist_ok=True)
                link = self.bin_dir / dependency.name
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(executable)
                return
        raise RuntimeError(
            f"{dependency.name} built but no executable was found; checked "
            f"{dependency.executables}"
        )

    def _render(self, command: Sequence[str], build_dir: Path) -> list[str]:
        values = {"jobs": str(self.jobs), "build_dir": str(build_dir)}
        return [token.format_map(values) for token in command]

    def _dependency_environment(self, dependency: Dependency) -> dict[str, str]:
        values = {
            "root": str(ROOT),
            "deps_dir": str(self.deps_dir),
            "python": str(self.python),
        }
        environment = {
            key: template.format_map(values) for key, template in dependency.environment
        }
        if platform.system() != "Darwin" or not dependency.macos_environment:
            return environment
        brew = shutil.which("brew")
        if brew is None:
            raise RuntimeError(
                f"{dependency.name} requires Homebrew prerequisites on macOS; "
                "install Homebrew and the packages documented in docs/dependencies.md"
            )
        # Resolving an installed toolchain is a read-only host inspection, so it
        # must also run during --dry-run; only dependency mutations are skipped.
        brew_prefix = subprocess.run(
            (brew, "--prefix"),
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        if not brew_prefix:
            raise RuntimeError("could not determine the Homebrew prefix")
        values["brew_prefix"] = brew_prefix
        environment.update({
            key: template.format_map(values)
            for key, template in dependency.macos_environment
        })
        if not self.dry_run:
            for key, value in environment.items():
                if key.endswith("_ROOT") and not Path(value).exists():
                    raise RuntimeError(
                        f"{dependency.name} prerequisite {key}={value} does not exist"
                    )
        return environment

    @staticmethod
    def _render_path(template: str, source: Path, build_dir: Path) -> Path:
        rendered = template.format_map({"build_dir": str(build_dir)})
        path = Path(rendered)
        return path if path.is_absolute() else source / path

    def _run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        extra_env = extra_env or {}
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in extra_env.items())
        rendered = f"{prefix} {shlex.join(command)}".strip()
        print(f"[{cwd}] {rendered}")
        if not self.dry_run:
            environment = os.environ.copy()
            environment.update(extra_env)
            subprocess.run(command, cwd=cwd, check=True, env=environment)

    def _capture(self, command: Sequence[str], *, cwd: Path) -> str:
        if self.dry_run:
            if tuple(command[:2]) == ("git", "rev-parse"):
                return self._expected_revision_for_local(cwd)
            return ""
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        return result.stdout

    def _expected_revision_for_local(self, cwd: Path) -> str:
        # Dry-run local verification still uses Git when a real clone exists.
        if (cwd / ".git").exists():
            result = subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=cwd,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            return result.stdout
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--deps-dir", type=Path, default=DEFAULT_DEPS_DIR)
    parser.add_argument(
        "--local-root",
        type=Path,
        help="use matching existing clones below this directory without modifying them",
    )
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--editable", action="store_true")
    parser.add_argument("--skip-project", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    profiles = args.profile or ([] if args.dependency else ["mhd"])
    manifest = load_manifest(args.manifest)
    dependencies = resolve_dependencies(manifest, profiles, args.dependency)
    bootstrapper = Bootstrapper(
        deps_dir=args.deps_dir,
        local_root=args.local_root,
        python=args.python,
        jobs=args.jobs,
        dry_run=args.dry_run,
        fetch_only=args.fetch_only,
        editable=args.editable,
        skip_project=args.skip_project,
    )
    bootstrapper.prepare_project()
    for dependency in dependencies:
        bootstrapper.prepare(dependency)
    bootstrapper.verify_environment()
    bootstrapper.finish()
    print("Dependency preparation complete.")
    if any(item.install == "native" for item in dependencies):
        print(f"Add {bootstrapper.bin_dir} to PATH for native validation executables.")
    if not args.fetch_only and any(item.install == "python" for item in dependencies):
        print("For provider workflows, use .venv/bin/python or uv run --no-sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
