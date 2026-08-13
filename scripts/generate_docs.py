"""Build the Doxygen API reference from the repository root Doxyfile.

Run from the repository root:

    uv run python scripts/generate_docs.py

Output is written to ``docs/api/html`` (generated, not tracked in git); open
``docs/api/html/index.html`` in a browser.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOXYFILE = ROOT / "Doxyfile"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if shutil.which("doxygen") is None:
        raise SystemExit(
            "doxygen executable not found on PATH; install it (e.g. `brew install doxygen`) "
            "before running this script."
        )
    result = subprocess.run(["doxygen", str(DOXYFILE)], cwd=ROOT, check=False)
    if result.returncode != 0:
        return result.returncode
    index = ROOT / "docs/api/html/index.html"
    print(f"generated {index}")
    if args.open and shutil.which("open") is not None:
        subprocess.run(["open", str(index)], check=False)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated index.html in the default browser (macOS `open`).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
