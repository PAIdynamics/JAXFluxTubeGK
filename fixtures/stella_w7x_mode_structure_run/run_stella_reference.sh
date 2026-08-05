#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STELLA_EXECUTABLE="${STELLA_EXECUTABLE:?Set STELLA_EXECUTABLE to the pinned stella binary}"
cd "${SCRIPT_DIR}"
"${STELLA_EXECUTABLE}" stella_w7x_adiabatic_electrons.in
