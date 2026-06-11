# W7-X ky=0.3 stella RHS Trace Summary

This fixture summarizes the targeted stella diagnostic trace generated with:

```bash
uv run python scripts/prepare_stella_w7x_rhs_trace_run.py --overwrite --output-root /tmp/stellarator_gk_stella_w7x_rhs_trace
bash /tmp/stellarator_gk_stella_w7x_rhs_trace/build_stella_rhs_trace.sh
bash /tmp/stellarator_gk_stella_w7x_rhs_trace/run_stella_rhs_trace.sh
uv run python scripts/summarize_stella_w7x_rhs_trace.py /tmp/stellarator_gk_stella_w7x_rhs_trace/run/stellarator_gk_w7x_ky03_rhs_trace.dat --output fixtures/w7x_ky03_stella_rhs_trace_summary/rhs_trace_summary.json
```

The raw trace is about 263 MB and stays outside git. The JSON summary records
the row counts, selected indices, units, grid extents, and complex norms needed
to verify that the external stella trace contains the requested `pdf_g`, `phi`,
mirror, drift, drive, parallel-streaming, and total-RHS records.
