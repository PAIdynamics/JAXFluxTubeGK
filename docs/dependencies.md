# External Dependencies

`optimal-fusion` keeps external source trees outside its repository while
providing a reproducible preparation step in `scripts/bootstrap_dependencies.py`.
All sources and exact fork commits are declared in `dependencies.toml`.

Provider forks remain manifest-installed rather than PEP 508 extras. Their
exact revisions, native prerequisites, build commands, and executable discovery
cannot be represented reliably by a normal Python extra. Pure reader support
needed by the standalone tests, including NetCDF4, is declared in the `dev`
extra instead.

## Profiles

- `core`: install only `optimal-fusion` and its development tools.
- `mhd`: fetch/build/install VMEC++, DESC, and GVEC as geometry providers.
- `validation-python`: install Gyaradax for Python-level comparisons.
- `validation-native`: fetch and compile GX, stella, and GKW.
- `validation`: prepare all Python and native validation codes.
- `all`: prepare every declared dependency.

The default profile is `mhd`:

```bash
python3 scripts/bootstrap_dependencies.py
```

Managed clones, native build trees, executable links, and the resolved state
ledger are written below `.dependencies/`, which is ignored by Git. Python
providers are installed into `.venv`. Use the exact recorded commits unless the
manifest is deliberately reviewed and updated. Project synchronization is
inexact so rerunning the command retains an already installed external profile.
Preparation ends with `uv pip check` and fails if provider requirements leave
the Python environment inconsistent.

On macOS, install the native libraries required by the MHD providers first:

```bash
brew install gcc lapack netcdf hdf5 libomp
```

The bootstrapper passes the Homebrew `OpenMP_ROOT` to the VMEC++ wheel build.
It also supplies a build-only compatibility shim for the old Versioneer copy in
the pinned DESC fork when `SafeConfigParser` is unavailable. Neither adjustment
modifies the dependency source trees.
Linux systems need the corresponding compiler, OpenMP, LAPACK, NetCDF, and HDF5
development packages.

The pinned DESC fork currently requires JAX below 0.10, so the MHD profile
selects a compatible JAX release. A later exact `uv sync` deliberately restores
the core lock; rerun the MHD preparation before using the providers again. Run
provider-dependent commands with `.venv/bin/python` or `uv run --no-sync` so
`uv` does not replace the prepared environment first.

To reuse sibling clones without modifying or checking them out:

```bash
python3 scripts/bootstrap_dependencies.py \
  --profile mhd \
  --local-root ..
```

The command verifies each local checkout is at the manifest revision. A
mismatch fails with an actionable message; user-owned clones are never
automatically checked out. Use `--editable` only when actively developing the
provider fork.

Prepare native validation codes separately:

```bash
python3 scripts/bootstrap_dependencies.py --profile validation
export PATH="$PWD/.dependencies/bin:$PATH"
```

GX still requires a supported CUDA environment and `GK_SYSTEM`. stella may
need site-specific CMake compiler/MPI/NetCDF flags, and GKW may need a machine
configuration. The bootstrapper intentionally exposes their native build
output rather than hiding platform errors.

Useful inspection modes:

```bash
python3 scripts/bootstrap_dependencies.py --profile all --dry-run
python3 scripts/bootstrap_dependencies.py --profile validation --fetch-only
python3 scripts/bootstrap_dependencies.py --dependency vmecpp
```

After a successful run, `.dependencies/state.json` records the resolved source
paths, revisions, roles, and commands. This ledger is local run metadata, not a
tracked fixture.
