"""Build and import the jax-fluxtube-gk wheel outside the source checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="3.13", help="Python version for the smoke venv")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="jax-fluxtube-gk-package-") as temporary:
        workspace = Path(temporary)
        dist = workspace / "dist"
        environment = workspace / "venv"
        subprocess.run(("uv", "build", "--out-dir", str(dist)), cwd=ROOT, check=True)
        wheels = tuple(dist.glob("*.whl"))
        sdists = tuple(dist.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError(f"expected one wheel and one sdist, found {wheels=} {sdists=}")
        subprocess.run(("uv", "venv", "--python", args.python, str(environment)), check=True)
        python = environment / "bin/python"
        subprocess.run(
            ("uv", "pip", "install", "--python", str(python), str(wheels[0])),
            check=True,
        )
        subprocess.run(
            (
                str(python),
                "-c",
                "import jax_fluxtube_gk; "
                "assert 'site-packages' in jax_fluxtube_gk.__file__; "
                "print(jax_fluxtube_gk.__file__)",
            ),
            cwd=workspace,
            check=True,
        )
    print("Package smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
