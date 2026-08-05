# stella W7-X Mode-Structure Reference Run

This directory contains the matched stella W7-X linear ITG input
for the external continuum-code mode-structure reference.

Matched controls:

- VMEC source: `dependency://gx/benchmarks/linear/ITG_w7x/wout_w7x.nc`
- torflux: `0.64`
- alpha0: `0.0`
- nfield_periods: `34.752359999999996`
- electron model: `adiabatic`
- ky grid: `[0.0, 0.1, 0.2, 0.3]`
- nzed/nmu/nvgrid: `256` / `8` / `16`
- tend/delt/growth window: `200.0` / `0.1` / `0.5`

Run stella:

```bash
bash fixtures/stella_w7x_mode_structure_run/run_stella_reference.sh
```

Export the portable reference fixture:

```bash
uv run python examples/export_stella_mode_structure_fixture.py --stella-output fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.out.nc --ky-values 0.1,0.2,0.3 --average-fraction 0.5 --stella-z-coordinate zed_over_2pi --output fixtures/w7x_itg_external_mode_structure_fixture.csv
```

Compare against the current solver fixture:

```bash
JAX_ENABLE_X64=1 uv run python examples/compare_mode_structure_fixtures.py --observed fixtures/w7x_itg_reduced_benchmark/mode_structures.csv --reference fixtures/w7x_itg_external_mode_structure_fixture.csv --ky-values 0.1,0.2,0.3 --require-profile --resample-reference-to-observed-z --output figures/w7x_itg_external_mode_structure_comparison.csv
```

The stella run and exported fixture now exist. The production parity claim
remains open until the solver observed fixture is rebuilt on matched stella
geometry/field-line-length, kx/linking, and late-window controls and then
passes the W7-X mode-structure gate.
