#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
uv run python examples/export_stella_mode_structure_fixture.py --stella-output fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.out.nc --ky-values 0.1,0.2,0.3 --average-fraction 0.5 --stella-z-coordinate zed_over_2pi --output fixtures/w7x_itg_external_mode_structure_fixture.csv
JAX_ENABLE_X64=1 uv run python examples/compare_mode_structure_fixtures.py --observed fixtures/w7x_itg_reduced_benchmark/mode_structures.csv --reference fixtures/w7x_itg_external_mode_structure_fixture.csv --ky-values 0.1,0.2,0.3 --require-profile --resample-reference-to-observed-z --output figures/w7x_itg_external_mode_structure_comparison.csv
uv run python scripts/run_w7x_production_readiness_gate.py
