# GX W7-X Mode-Structure Reference Run

This directory contains the patched GX W7-X linear ITG input for
producing the external complex mode-structure reference required
to upgrade the reduced W7-X fixture into a full code-to-code gate.

1. Copy the VMEC file into the run directory:

   `cp dependency://gx/benchmarks/linear/ITG_w7x/wout_w7x.nc fixtures/gx_w7x_mode_structure_run/wout_w7x.nc`

2. Run GX externally:

   `cd fixtures/gx_w7x_mode_structure_run && path/to/gx itg_w7x_adiabatic_electrons.in`

3. Export the retained GX field diagnostic to a portable fixture:

   `uv run python examples/export_gx_mode_structure_fixture.py --gx-big-output fixtures/gx_w7x_mode_structure_run/itg_w7x_adiabatic_electrons.big.nc --gx-growth-output fixtures/gx_w7x_mode_structure_run/itg_w7x_adiabatic_electrons.out.nc --ky-values 0.1,0.2,0.3 --gx-z-coordinate theta_over_2pi --output fixtures/w7x_itg_external_mode_structure_fixture.csv`

4. Compare the current solver fixture against the external one:

   `JAX_ENABLE_X64=1 uv run python examples/compare_mode_structure_fixtures.py --observed fixtures/w7x_itg_reduced_benchmark/mode_structures.csv --reference fixtures/w7x_itg_external_mode_structure_fixture.csv --ky-values 0.1,0.2,0.3 --require-profile --output figures/w7x_itg_external_mode_structure_comparison.csv`

The compact `.out.nc` file alone is not enough for profile parity;
the complex profiles are read from `Diagnostics/Phi` in `.big.nc`.

The workflow can be audited or run through:

```bash
uv run python scripts/run_w7x_external_reference_workflow.py
```

If the run will happen on another machine, package the inputs and handoff
scripts first. Unpack the tarball over a full `new-plasma-code` checkout on
the GX/CUDA machine:

```bash
uv run python scripts/package_w7x_external_reference_bundle.py --output fixtures/gx_w7x_mode_structure_run/w7x_external_reference_bundle.tar.gz
```

The current committed status is written to
`fixtures/gx_w7x_mode_structure_run/external_reference_status.json` and is
`blocked_missing_gx_executable`: the prepared input exists, the VMEC file has
been copied into this run directory, and the local transfer bundle
`w7x_external_reference_bundle.tar.gz` exists, but no local
CUDA/NVIDIA-capable GX executable or retained W7-X `.big.nc`/`.out.nc` outputs
are present. On a GX-capable machine, rerun with
`--copy-vmec --run-gx --gx-executable /path/to/gx`.

Equivalently, set `GX_EXECUTABLE` and use the checked-in handoff:

```bash
GX_EXECUTABLE=/path/to/gx bash fixtures/gx_w7x_mode_structure_run/run_external_reference.sh
```

After returned `.big.nc`/`.out.nc` files are available in this checkout, ingest
them, export the CSV fixture, run the W7-X parity gate, and refresh readiness:

```bash
bash fixtures/gx_w7x_mode_structure_run/ingest_returned_outputs.sh --copy-outputs --resample-reference-to-observed-z
```

After the external parity fixture passes, run the production CPU timing/readiness
handoff:

```bash
bash fixtures/gx_w7x_mode_structure_run/run_production_timing_after_parity.sh
```
