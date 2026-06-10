#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${GX_EXECUTABLE:?Set GX_EXECUTABLE to the GX executable path}"

uv run python scripts/run_w7x_external_reference_workflow.py \
  --copy-vmec \
  --run-gx \
  --gx-executable "${GX_EXECUTABLE}" \
  --require-pass

uv run python scripts/run_w7x_production_readiness_gate.py
