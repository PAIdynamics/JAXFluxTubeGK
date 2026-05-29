# Optimization Integration

Phase 12 introduces a small fixed-topology optimization layer. It is intended
for reduced single-surface experiments before coupling full DESC equilibria.

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
from stellarator_gk import single_surface_objective, toy_gradient_descent_step

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
