#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

uv run python scripts/run_w7x_production_readiness_gate.py
uv run python scripts/run_w7x_production_cpu_timing.py "$@"
uv run python scripts/run_w7x_production_readiness_gate.py --require-pass
