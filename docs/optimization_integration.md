# Optimization Integration

The optimization layer provides a fixed-topology design boundary for reduced
single-surface experiments and real-provider geometry arrays.

## Stable Design Objective

`DesignObjectiveSpec` composes named physical terms without encoding the loss
in an opaque objective string:

```python
from jax_fluxtube_gk import DesignObjectiveSpec, design_objective

spec = DesignObjectiveSpec(
    selected_ky=3,
    growth_weight=1.0,
    frequency_weight=0.1,
    frequency_target=0.0,
    mode_structure_weight=0.2,
    quasilinear_weight=0.01,
)
result = design_objective(
    knobs,
    velocity_grid,
    parallel_grid,
    fourier_grid,
    initial_state,
    spec,
    geometry=provider_geometry,
    target_mode_structure=target_phi,
)
```

The result exposes the scalar loss, selected and maximum growth, selected real
frequency, frequency penalty, phase-aligned complex mode-shape penalty, and
quasilinear proxy. Frequency targeting requires an explicit `selected_ky`;
the API refuses to attach frequency to an ambiguous maximum-growth branch.
Each `ky` target is aligned by an independent unit complex phase before the
shape penalty, while its amplitude remains physical.

### Branch and gradient policy

`growth_aggregation="max"` is a hard maximum and is nonsmooth when two modes
exchange dominance. `growth_aggregation="softmax"` supplies a differentiable
log-sum-exp envelope controlled by `softmax_temperature`. Every result reports
the top-two `growth_branch_gap` and `near_degenerate_branch`; these diagnostics
must be retained with optimization records.

`audit_design_gradient` compares reverse-mode AD with a central difference for
one scalar equilibrium/provider parameter. Near a small branch gap, use the
soft maximum or select a physical `ky` branch explicitly, then require the
audit to pass. A hard-maximum gradient across a branch change must not be
reported as a validated derivative.

### Multiple surfaces and field lines

`scan_single_surface_objective` evaluates a fixed Cartesian sample set in
`rho`, `alpha`, and `ky`. Pass its result to `robust_scan_objective` to reduce
all samples with a weighted mean, hard worst case, or smooth worst-case
(`softmax`) policy:

```python
from jax_fluxtube_gk import RobustAggregationSpec, robust_scan_objective

robust = robust_scan_objective(
    scan,
    RobustAggregationSpec(method="softmax", softmax_temperature=0.05),
    sample_weights=weights,
)
loss = robust.scalar_objective
```

The scan tensor and optional weights have shape `(rho, alpha, ky)` and are
flattened in row-major order. Weights are normalized, fixed controls; gradients
flow through every sample objective. The result always reports the weighted
mean, hard worst case, and smooth worst case so optimization records can be
audited independently of the selected reduction.

## Remeshing and topology changes

Create an `OptimizationTopologyContract` before compiling an objective. It
hashes velocity nodes, parallel nodes, Fourier modes, mode connectivity, and
the provider's field-line/linking metadata. Before reusing a compiled objective,
precompute, checkpoint, or gradient, build the candidate contract and call
`assert_fixed_optimization_topology(reference, candidate)`.

A changed resolution, node placement, derivative backend, parallel boundary,
`kx`/`ky` set, `ikxspace`, twist-and-shift map, field-line period count, or MHD
provider topology raises `TopologyChangeError`. The caller must stop the current
AD trace and rebuild all discrete objects. Continuous geometry values may
change without rebuilding when this contract remains identical.

## Fixed And Differentiable Inputs

Static during a traced objective:

- grid sizes and derivative backends,
- perpendicular Fourier mode topology and mode connectivity,
- selected geometry model,
- selected `ky` index and time-step count.

Differentiable:

- profile knobs: density, temperature, `R/L_n`, `R/L_T`,
- local geometry knobs: `q`, `shat`, `eps`, `rho`, `alpha`,
- placeholder beta and pressure-gradient controls,
- low-amplitude equilibrium coefficients used as a toy stand-in for future
  DESC/Boozer geometry coefficients.

## Toy Objective

Run the repository example with:

```bash
uv run --extra dev python examples/optimization_loop.py --iterations 5
```

Each row prints the scalar objective, selected/max growth rates, and the
current optimization knobs before applying the next gradient step.

```python
import jax
from jax_fluxtube_gk import single_surface_objective, toy_gradient_descent_step

def loss(knobs):
    return single_surface_objective(
        knobs,
        velocity_grid,
        parallel_grid,
        fourier_grid,
        initial_state,
        electron_params=electrons,
        connectivity=connectivity,
        config=config,
    ).scalar_objective

value, gradient = jax.value_and_grad(loss)(knobs)
step = toy_gradient_descent_step(loss, knobs, learning_rate=1e-3)
```

The toy equilibrium-coefficient path is only for plumbing and gradient tests.
Production stellarator optimization should replace it with precomputed Boozer,
GX/eik, or DESC-derived geometry arrays on the same fixed grid topology.

## Reduced W7-X Loop Through VMEC++

After preparing the MHD dependency profile, run a one-iteration smoke case with
the installed VMEC++ `w7x-standard` configuration:

```bash
JAX_ENABLE_X64=1 uv run --no-sync \
  python examples/vmecpp_w7x_design_loop.py \
  --iterations 1 --n-z 16 --n-vpar 2 --n-mu 2 --n-steps 1 \
  --output /tmp/jax-fluxtube-gk-vmecpp-w7x-design/smoke.json
```

The example loads W7-X from VMEC++ rather than a repository artifact, perturbs
one boundary harmonic, performs fresh center/plus/minus VMEC++ solves, converts
each in-memory result through `VmecppGeometryProvider`, checks the fixed-topology
contract, and takes an outer central-finite-difference step. Its JSON record is
required to live outside the repository.

This demonstrates the real MHD-provider integration boundary. VMEC++ is not
currently differentiated through, and the example is neither an end-to-end AD
claim nor a production full-boundary shape optimizer.

## Reproducible Checkpoints

Use `build_optimization_checkpoint` and `write_optimization_checkpoint` at an
accepted iteration. Schema version 1 records the complete `rho/alpha/ky`
sample axes and objectives, objective and aggregation policies, design
parameters, iteration history, provider provenance, code and dependency
revisions, invocation, random seed, and every fixed-topology fingerprint.

On restart, load the record with `load_optimization_checkpoint` and call
`assert_checkpoint_topology` with the newly constructed contracts before
reusing compiled functions or optimizer state. A mesh, mode, linking, or
provider-topology change raises `TopologyChangeError` and requires rebuilding
the differentiated solve. Checkpoint files contain optimization records, not
embedded MHD equilibria or W7-X geometry arrays.
