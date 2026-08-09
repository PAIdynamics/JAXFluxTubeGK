# W7-X ITG Reduced Benchmark Fixture

This fixture uses the local GX/GIST W7-X eik geometry reference and the GX
`ITG_w7x` adiabatic-electron input deck as provenance:

- geometry: `dependency://gx/geometry_modules/vmec/tests/gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000`
- GX input: `dependency://gx/benchmarks/linear/ITG_w7x/itg_w7x_adiabatic_electrons.in`

Regenerate the fixture from the repository root with:

```bash
uv run python scripts/generate_w7x_itg_reduced_benchmark.py
```

The geometry is a real W7-X external eik table. The committed growth rates,
real frequencies, and complex mode structures are reduced `jax_fluxtube_gk`
regression artifacts, not an external-code parity claim. Replace those
diagnostics once a matching GX, GKW, GS2, or stella W7-X time-history or
mode-structure fixture is available.

The GX path for producing that external fixture is prepared in
`fixtures/gx_w7x_mode_structure_run/`. Its metadata gives the VMEC copy
command, GX run command, `.big.nc` export command, and selected-`ky`
comparison command.
