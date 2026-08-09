"""Write the selected-ky CBC source-term trace used for GKW history debugging.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_source_term_trace.py
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jax_fluxtube_gk_matplotlib")

import jax

jax.config.update("jax_enable_x64", True)

from jax_fluxtube_gk import (
    run_cyclone_base_case_source_term_trace,
    write_cyclone_source_term_trace_csv,
)


def main() -> None:
    args = _parse_args()
    trace = run_cyclone_base_case_source_term_trace(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        output_windows=tuple(args.output_windows),
        initial_profile=args.initial_profile,
        normalization_model=args.normalization_model,
        parallel_derivative_model=args.parallel_derivative_model,
    )
    write_cyclone_source_term_trace_csv(args.output, trace)
    print(
        f"max_reconstruction_error={float(trace.reconstruction_error.max()):.8e}, "
        f"final_rhs_norm={float(trace.rhs_norm[-1]):.8e}, "
        f"final_log_normalization={float(trace.log_normalization[-1]):.8e}"
    )
    print(args.output)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cyclone_source_term_trace.csv"),
    )
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--output-windows", type=int, nargs="+", default=(0, 1, 40, 80))
    parser.add_argument("--initial-profile", choices=("cosine2", "cosine"), default="cosine2")
    parser.add_argument(
        "--normalization-model",
        choices=("weighted", "gkw_unweighted"),
        default="gkw_unweighted",
    )
    parser.add_argument(
        "--parallel-derivative-model",
        choices=("gkw_upwind", "gkw_igh", "matrix"),
        default="gkw_igh",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
