# Numerical fixture policy

Fixtures are compact executable contracts, not an archive of external-solver
runs. Native outputs, full distribution histories, NetCDF files, build trees,
and restart states belong in caller-owned scratch or archival storage outside
the repository. A fixture may be committed only when a test consumes it and
its source revision, input, command, normalization, and transformation are
recorded beside it or in the corresponding status/README record.

## GKW decision

The retained GKW contracts are deliberately selected slices:

- selected-`ky` parallel-potential profiles for normalization and profile
  alignment;
- four selected velocity-distribution slices for sign, phase, and term audits;
- three selected times of those four slices for temporal consistency;
- compact selected-state/RHS/matrix inputs and the matching time/input files.

Together, the root `fixtures/gkw_*` contracts occupy less than 200 kB; no raw
GKW run directory or full phase-space history is tracked. These slices are
essential because they exercise different conventions and are directly loaded
by regression tests. Further downsampling would remove velocity, time, or
parallel-position structure used by those tests, so the current selected
slices are the compact baseline.

Raw native output should be regenerated from the pinned dependency workflow
and written outside the repository. `fixtures/raw`, `fixtures/archive`, and
nested directories with those names are ignored and excluded from source
manifests. The regression test enforces a 200 kB aggregate GKW budget, a 64 KiB
per-file ceiling, and forbidden archival naming.
