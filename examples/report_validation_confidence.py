"""Print or write the machine-readable Priority 5 confidence ledger."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from jax_fluxtube_gk.validation_confidence import (
    VALIDATION_CONFIDENCE_SCHEMA_VERSION,
    priority5_confidence_gaps,
    write_validation_confidence_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    gaps = priority5_confidence_gaps()
    if args.output is not None:
        write_validation_confidence_report(args.output, gaps, overwrite=args.overwrite)
        print(args.output)
    else:
        print(
            json.dumps(
                {
                    "schema_version": VALIDATION_CONFIDENCE_SCHEMA_VERSION,
                    "gaps": [asdict(gap) for gap in gaps],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
