# GX W7-X Mode-Structure Reference Run

This directory contains the patched GX W7-X linear ITG input for
producing the external complex mode-structure reference required
to upgrade the reduced W7-X fixture into a full code-to-code gate.

1. Copy the VMEC file into the run directory:

   `cp relevant-codes/gx/benchmarks/linear/ITG_w7x/wout_w7x.nc fixtures/gx_w7x_mode_structure_run/wout_w7x.nc`

2. Run GX externally:

   `cd fixtures/gx_w7x_mode_structure_run && path/to/gx itg_w7x_adiabatic_electrons.in`

3. Export the retained GX field diagnostic to a portable fixture:

   `uv run python examples/export_gx_mode_structure_fixture.py --gx-big-output fixtures/gx_w7x_mode_structure_run/itg_w7x_adiabatic_electrons.big.nc --gx-growth-output fixtures/gx_w7x_mode_structure_run/itg_w7x_adiabatic_electrons.out.nc --ky-values 0.1,0.2,0.3 --gx-z-coordinate theta_over_2pi --output fixtures/w7x_itg_external_mode_structure_fixture.csv`

4. Compare the current solver fixture against the external one:

   `JAX_ENABLE_X64=1 uv run python examples/compare_mode_structure_fixtures.py --observed fixtures/w7x_itg_reduced_benchmark/mode_structures.csv --reference fixtures/w7x_itg_external_mode_structure_fixture.csv --ky-values 0.1,0.2,0.3 --require-profile --output figures/w7x_itg_external_mode_structure_comparison.csv`

The compact `.out.nc` file alone is not enough for profile parity;
the complex profiles are read from `Diagnostics/Phi` in `.big.nc`.
