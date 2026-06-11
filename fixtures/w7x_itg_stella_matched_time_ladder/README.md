# W7-X stella-Matched Time Ladder

Regenerate from the repository root with:

```bash
uv run python scripts/run_w7x_stella_matched_time_ladder.py
```

The ladder holds the stella-imported geometry, selected ky set,
and kx=0/n_kx=1 controls fixed. It only extends the solver time
horizon so the ordered stella parity audit can move beyond
`growth_window_time_normalization` before any RHS or velocity
space terms are changed.
