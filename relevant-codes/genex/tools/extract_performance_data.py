#!/usr/bin/env python3
import pandas as pd
import argparse
import json
from pathlib import Path
import sys
from typing import Any
import profiler_tools
import tempfile

def write_json(tree: dict, path: str) -> None:
    """
    Normalize a performance-tree-like structure to JSON-serializable
    primitives.
    """
    def normalize(obj):
        # Primitives
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        # Mappings
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        # Sequences / iterables (but not strings)
        if isinstance(obj, (list, tuple, set)):
            return [normalize(v) for v in obj]
        # Objects with __dict__ (dataclasses, simple objects)
        if hasattr(obj, "__dict__"):
            return normalize(vars(obj))
        # Mapping/Iterable fallbacks
        try:
            import collections.abc as cabc
            if isinstance(obj, cabc.Mapping):
                return {k: normalize(v) for k, v in obj.items()}
            if isinstance(obj, cabc.Iterable):
                return [normalize(v) for v in obj]
        except Exception:
            pass
        # Fallback to string representation
        return str(obj)

    normalized = normalize(tree)

    # Ensure target directory exists
    dirpath = Path(path).parent
    if dirpath and not Path(dirpath).exists():
        Path(dirpath).mkdir(exist_ok=True)

    with open(path,mode='w') as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
        f.write("\n")
        print("json file written to ",path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract performance data and write as JSON")
    parser.add_argument("input", help="Path to input profiler file")
    parser.add_argument("-o", "--output", help="Path to output JSON file",
                        default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    inp = args.input
    out = args.output or (Path(inp).stem + ".json")

    if not Path(inp).exists():
        print(f"Input file does not exist: {inp}", file=sys.stderr)
        sys.exit(2)

    try:
        pt = profiler_tools.load_profiler_tree(inp)
    except Exception as e:
        print(f"Failed to load performance data: {e}", file=sys.stderr)
        sys.exit(3)

    try:
        write_json(pt, out)
    except Exception as e:
        print(f"Failed to write JSON: {e}", file=sys.stderr)
        sys.exit(4)

    print(out)


if __name__ == "__main__":
    main()
