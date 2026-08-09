# W7-X ky=0.3 stella RHS Trace Summary

This fixture summarizes the targeted stella diagnostic trace generated with:

```bash
uv run python scripts/prepare_stella_w7x_rhs_trace_run.py --stella-source ../stella --vmec-file ../gx/benchmarks/linear/ITG_w7x/wout_w7x.nc --overwrite --output-root /tmp/jax_fluxtube_gk_stella_w7x_rhs_trace
bash /tmp/jax_fluxtube_gk_stella_w7x_rhs_trace/build_stella_rhs_trace.sh
bash /tmp/jax_fluxtube_gk_stella_w7x_rhs_trace/run_stella_rhs_trace.sh
uv run python scripts/summarize_stella_w7x_rhs_trace.py /tmp/jax_fluxtube_gk_stella_w7x_rhs_trace/run/jax_fluxtube_gk_w7x_ky03_rhs_trace.dat --stella-source ../stella --stella-executable /tmp/jax_fluxtube_gk_stella_w7x_rhs_trace/stella/COMPILATION/build_cmake/COMPILATION/stella --output fixtures/w7x_ky03_stella_rhs_trace_summary/rhs_trace_summary.json
```

The raw v3 trace is about 332 MiB and stays outside git. The JSON summary records
the source/executable provenance, velocity weights, row counts, selected
indices, units, grid extents, and complex norms needed to verify that the
external stella trace contains the requested `pdf_g`, `phi`, mirror, drift,
drive, parallel-streaming, total-RHS, quasineutrality numerator/denominator,
native-state-scale, and explicit RHS-call records.
