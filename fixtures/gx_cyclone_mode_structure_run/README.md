# GX Mode-Structure Fixture Run

This directory contains a patched GX input that retains the full-field
`.big.nc` diagnostics needed for the multi-ky mode-structure gate.

1. Run GX externally:

   `cd fixtures/gx_cyclone_mode_structure_run && path/to/gx itg_salpha_adiabatic_electrons.in`

2. Export the retained GX field diagnostic to the portable fixture:

   `uv run python examples/export_gx_mode_structure_fixture.py --gx-big-output fixtures/gx_cyclone_mode_structure_run/itg_salpha_adiabatic_electrons.big.nc --gx-growth-output fixtures/gx_cyclone_mode_structure_run/itg_salpha_adiabatic_electrons.out.nc --ky-values 0.3,0.5 --gx-z-coordinate theta_over_2pi --output fixtures/gx_cyclone_mode_structure_fixture.csv`

3. Compare the current solver against that fixture:

   `JAX_ENABLE_X64=1 uv run python examples/run_cyclone_mode_structure_gate.py --reference-fixture fixtures/gx_cyclone_mode_structure_fixture.csv --profile gx-salpha-input --target-convention gx-salpha --ky-input-convention internal_krho --require-profile --resample-reference-to-solver-z --periodic-z`

The compact `.out.nc` file alone is not enough for this gate; the
required complex profiles are stored in `Diagnostics/Phi` in `.big.nc`.
