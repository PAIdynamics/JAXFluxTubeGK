# W7-X Reduced Convergence and Timing Study

Regenerate from the repository root with:

```bash
uv run python scripts/run_w7x_reduced_convergence_study.py
```

This is a reduced solver-regression convergence matrix using the
real GX/GIST W7-X eik table. It is not an external-code parity
claim. The production W7-X comparison remains pending until
`fixtures/w7x_itg_external_mode_structure_fixture.csv` is exported
from a matched GX/GKW/GS2/stella run.

The current production-readiness ledger can be regenerated with:

```bash
uv run python scripts/run_w7x_production_readiness_gate.py
```

It audits these reduced convergence/timing artifacts, runs the W7-X external
mode-structure gate, and keeps DESC optimization labeled reduced until the
external parity target and true production CPU timing artifact exist.
