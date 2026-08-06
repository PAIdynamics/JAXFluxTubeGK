"""Run the algebraic kinetic-electron TEM preflight and print JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from stellarator_gk.tem_validation import (
    run_reduced_tem_linear_smoke,
    run_tem_physics_preflight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-smoke", action="store_true")
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--n-windows", type=int, default=12)
    args = parser.parse_args(argv)
    report = (
        run_reduced_tem_linear_smoke(
            steps_per_window=args.steps_per_window,
            n_windows=args.n_windows,
        )
        if args.linear_smoke
        else run_tem_physics_preflight()
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    successful = report.finite if args.linear_smoke else report.passed
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
