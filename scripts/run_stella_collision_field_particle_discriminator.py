"""Run a paired native-stella field-particle collision discriminator.

Outputs are written only to a caller-selected scratch directory.  This is a
sensitive native reference observable, not yet a coefficient/action parity
test for the local collision operator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def stella_collision_input(*, field_particle: bool) -> str:
    """Return the compact collision-only stella input used by the discriminator."""

    fieldpart = ".true." if field_particle else ".false."
    return f"""&geometry_options
  geometry_option = 'miller'
/
&geometry_miller
  nzed_local = 128
  rhoc = 0.5
  shat = 0.796
  qinp = 1.4
  rmaj = 2.77778
  rgeo = 2.77778
/
&gyrokinetic_terms
  include_parallel_streaming = .false.
  include_mirror = .false.
  include_xdrift = .false.
  include_ydrift = .false.
  include_drive = .false.
  include_nonlinear = .false.
/
&species_options
  nspec = 2
/
&species_parameters_1
  dens = 1.0
  mass = 1.0
  temp = 1.0
  type = 'ion'
  z = 1.0
/
&species_parameters_2
  dens = 1.0
  mass = 0.0005446
  temp = 1.0
  type = 'electron'
  z = -1.0
/
&kxky_grid_option
  grid_option = 'box'
/
&kxky_grid_box
  nx = 6
  ny = 9
  y0 = 15
/
&z_grid
  nzed = 12
/
&z_boundary_condition
  boundary_option = 'linked'
/
&velocity_grids
  nmu = 2
  nvgrid = 3
/
&diagnostics
  nsave = 1
  nwrite = 1
  write_all_distribution = .true.
/
&initialise_distribution
  initialise_distribution_option = 'default'
  phiinit = 0.01
/
&initialise_distribution_maxwellian
  width0 = 1.0
/
&time_trace_options
  nstep = 1
/
&time_step
  delt = 0.01
/
&numerical_algorithms
  explicit_algorithm = 'rk2'
  stream_implicit = .true.
  mirror_implicit = .true.
  drifts_implicit = .false.
/
&dissipation_and_collisions_options
  include_collisions = .true.
  collisions_implicit = .true.
  collision_model = 'fokker-planck'
  vnew_ref = 0.01
/
&collisions_fokker_planck
  testpart = .true.
  fieldpart = {fieldpart}
  lmax = 1
  jmax = 1
  nvel_local = 64
  interspec = .true.
  intraspec = .true.
  advfield_coll = .false.
  density_conservation = .false.
  density_conservation_field = .false.
  density_conservation_tp = .false.
  exact_conservation = .false.
  exact_conservation_tp = .false.
  vpa_operator = .true.
  mu_operator = .true.
/
"""


def source_revision(source: Path) -> dict[str, object]:
    """Read provenance from the source checkout, not stella's unreliable NetCDF tag."""

    source = Path(source).resolve()
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"source": str(source), "revision": revision, "dirty": dirty}


def summarize_outputs(
    field_particle_output: Path,
    test_particle_output: Path,
    *,
    provenance: dict[str, object],
) -> dict[str, object]:
    """Validate matched initial conditions and quantify the native field term."""

    import netCDF4

    variable_names = ("phi2", "g2_vs_vpamus", "h2_vs_vpamus", "f2_vs_vpamus")
    metrics: dict[str, dict[str, float]] = {}
    with netCDF4.Dataset(field_particle_output) as enabled, netCDF4.Dataset(
        test_particle_output
    ) as disabled:
        if enabled.dimensions["t"].size != 2 or disabled.dimensions["t"].size != 2:
            raise ValueError("collision discriminator requires exactly two output times")
        for name in variable_names:
            if name not in enabled.variables or name not in disabled.variables:
                raise ValueError(f"missing required stella diagnostic {name!r}")
            on = np.asarray(enabled.variables[name][:])
            off = np.asarray(disabled.variables[name][:])
            if on.shape != off.shape:
                raise ValueError(f"shape mismatch for {name!r}: {on.shape} != {off.shape}")
            initial_max = float(np.max(np.abs(on[0] - off[0])))
            final_delta = on[-1] - off[-1]
            denominator = max(float(np.linalg.norm(off[-1].ravel())), np.finfo(float).tiny)
            metrics[name] = {
                "initial_max_abs_difference": initial_max,
                "final_max_abs_difference": float(np.max(np.abs(final_delta))),
                "final_relative_l2_difference": float(
                    np.linalg.norm(final_delta.ravel()) / denominator
                ),
            }
        native_tag = str(getattr(enabled, "software_version", "unknown"))

    max_initial = max(item["initial_max_abs_difference"] for item in metrics.values())
    h2_effect = metrics["h2_vs_vpamus"]["final_relative_l2_difference"]
    if max_initial > 1.0e-13:
        raise ValueError(f"paired stella initial states differ by {max_initial:.3e}")
    if h2_effect <= 1.0e-8:
        raise ValueError(f"field-particle discriminator is insensitive: h2 effect={h2_effect:.3e}")
    return {
        "schema_version": 1,
        "benchmark": "stella_collision_field_particle_discriminator",
        "status": "native_discriminator_passed",
        "scope": "sensitive native observable; not local coefficient/action parity",
        "source_provenance": provenance,
        "native_netcdf_software_version_informational": native_tag,
        "metrics": metrics,
    }


def run_discriminator(
    output_dir: Path,
    stella_executable: Path,
    stella_source: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write, execute, and summarize the paired native runs."""

    output_dir = Path(output_dir).resolve()
    executable = Path(stella_executable).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "field_particle": True,
        "test_particle": False,
    }
    for name, enabled in cases.items():
        input_path = output_dir / f"{name}.in"
        output_path = output_dir / f"{name}.out.nc"
        if (input_path.exists() or output_path.exists()) and not overwrite:
            raise FileExistsError(f"{name} output exists; pass --overwrite")
        input_path.write_text(stella_collision_input(field_particle=enabled))
        subprocess.run([str(executable), input_path.name], cwd=output_dir, check=True)

    report = summarize_outputs(
        output_dir / "field_particle.out.nc",
        output_dir / "test_particle.out.nc",
        provenance=source_revision(stella_source),
    )
    report["stella_executable"] = str(executable)
    report_path = output_dir / "collision_field_particle_discriminator.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stella-executable", type=Path, required=True)
    parser.add_argument("--stella-source", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = run_discriminator(
        args.output_dir,
        args.stella_executable,
        args.stella_source,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
