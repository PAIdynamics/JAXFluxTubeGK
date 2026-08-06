"""Run the algebraic kinetic-electron TEM preflight and print JSON."""

from __future__ import annotations

from dataclasses import asdict
import json

from stellarator_gk.tem_validation import run_tem_physics_preflight


def main() -> int:
    report = run_tem_physics_preflight()
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
