"""Schema-versioned checkpoints for reproducible fixed-topology design runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .design_topology import (
    OptimizationTopologyContract,
    assert_fixed_optimization_topology,
)
from .optimization import DesignObjectiveSpec, RobustAggregationSpec


OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OptimizationCheckpoint:
    """Complete restart/audit record for one robust design iteration."""

    schema_version: int
    iteration: int
    objective_spec: DesignObjectiveSpec
    aggregation_spec: RobustAggregationSpec
    rho_values: tuple[float, ...]
    alpha_values: tuple[float, ...]
    ky_indices: tuple[int, ...]
    topology_contracts: tuple[OptimizationTopologyContract, ...]
    provider_provenance: tuple[dict[str, object], ...]
    parameters: dict[str, object]
    sample_objectives: tuple[float, ...]
    aggregate_objective: float
    history: tuple[dict[str, object], ...]
    code_revision: str
    dependency_revisions: tuple[tuple[str, str], ...]
    command: tuple[str, ...]
    random_seed: int | None


def build_optimization_checkpoint(
    *,
    iteration: int,
    objective_spec: DesignObjectiveSpec,
    aggregation_spec: RobustAggregationSpec,
    rho_values,
    alpha_values,
    ky_indices,
    topology_contracts: Sequence[OptimizationTopologyContract],
    provider_provenance: Sequence[Mapping[str, object]],
    parameters: Mapping[str, object],
    sample_objectives,
    aggregate_objective,
    code_revision: str,
    dependency_revisions: Mapping[str, str],
    command: Sequence[str],
    history: Sequence[Mapping[str, object]] = (),
    random_seed: int | None = None,
) -> OptimizationCheckpoint:
    """Normalize and validate metadata needed to reproduce an iteration."""

    rho = _float_tuple(rho_values, "rho_values")
    alpha = _float_tuple(alpha_values, "alpha_values")
    ky = tuple(int(value) for value in np.ravel(np.asarray(ky_indices)))
    objectives = _float_tuple(sample_objectives, "sample_objectives")
    if iteration < 0:
        raise ValueError("checkpoint iteration must be nonnegative")
    if not rho or not alpha or not ky:
        raise ValueError("rho_values, alpha_values, and ky_indices must be nonempty")
    expected_objectives = len(rho) * len(alpha) * len(ky)
    if len(objectives) != expected_objectives:
        raise ValueError(
            "sample_objectives must contain one value per rho/alpha/ky sample"
        )
    geometry_samples = len(rho) * len(alpha)
    topologies = tuple(topology_contracts)
    provenance = tuple(dict(_json_mapping(value)) for value in provider_provenance)
    if len(topologies) not in (1, geometry_samples):
        raise ValueError(
            "topology_contracts must contain one shared contract or one per rho/alpha sample"
        )
    if len(provenance) not in (1, geometry_samples):
        raise ValueError(
            "provider_provenance must contain one shared record or one per rho/alpha sample"
        )
    aggregate = float(aggregate_objective)
    if not math.isfinite(aggregate):
        raise ValueError("aggregate_objective must be finite")
    if not code_revision.strip():
        raise ValueError("code_revision must be recorded")
    dependencies = tuple(
        sorted((str(name), str(revision)) for name, revision in dependency_revisions.items())
    )
    if any(not name or not revision for name, revision in dependencies):
        raise ValueError("dependency revision names and values must be nonempty")
    return OptimizationCheckpoint(
        schema_version=OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION,
        iteration=int(iteration),
        objective_spec=objective_spec,
        aggregation_spec=aggregation_spec,
        rho_values=rho,
        alpha_values=alpha,
        ky_indices=ky,
        topology_contracts=topologies,
        provider_provenance=provenance,
        parameters=dict(_json_mapping(parameters)),
        sample_objectives=objectives,
        aggregate_objective=aggregate,
        history=tuple(dict(_json_mapping(row)) for row in history),
        code_revision=code_revision,
        dependency_revisions=dependencies,
        command=tuple(str(part) for part in command),
        random_seed=None if random_seed is None else int(random_seed),
    )


def write_optimization_checkpoint(
    path: str | Path,
    checkpoint: OptimizationCheckpoint,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a checkpoint atomically enough to avoid a partial target file."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"optimization checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    payload = _checkpoint_payload(checkpoint)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def load_optimization_checkpoint(path: str | Path) -> OptimizationCheckpoint:
    """Load and validate an optimization checkpoint from its public schema."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported optimization checkpoint schema version")
    return build_optimization_checkpoint(
        iteration=payload["iteration"],
        objective_spec=DesignObjectiveSpec(**payload["objective_spec"]),
        aggregation_spec=RobustAggregationSpec(**payload["aggregation_spec"]),
        rho_values=payload["samples"]["rho_values"],
        alpha_values=payload["samples"]["alpha_values"],
        ky_indices=payload["samples"]["ky_indices"],
        topology_contracts=tuple(
            _topology_contract_from_payload(item) for item in payload["topology_contracts"]
        ),
        provider_provenance=payload["provider_provenance"],
        parameters=payload["parameters"],
        sample_objectives=payload["sample_objectives"],
        aggregate_objective=payload["aggregate_objective"],
        history=payload["history"],
        code_revision=payload["code_revision"],
        dependency_revisions=dict(payload["dependency_revisions"]),
        command=payload["command"],
        random_seed=payload["random_seed"],
    )


def assert_checkpoint_topology(
    checkpoint: OptimizationCheckpoint,
    candidates: Sequence[OptimizationTopologyContract],
) -> None:
    """Reject a restart whose current grids/linking differ from the checkpoint."""

    candidates = tuple(candidates)
    references = checkpoint.topology_contracts
    if len(references) == 1 and len(candidates) > 1:
        references = references * len(candidates)
    if len(candidates) == 1 and len(references) > 1:
        candidates = candidates * len(references)
    if len(references) != len(candidates):
        raise ValueError("checkpoint and candidate topology counts differ")
    for reference, candidate in zip(references, candidates, strict=True):
        assert_fixed_optimization_topology(reference, candidate)


def _checkpoint_payload(checkpoint: OptimizationCheckpoint) -> dict[str, object]:
    return {
        "schema_version": checkpoint.schema_version,
        "iteration": checkpoint.iteration,
        "objective_spec": asdict(checkpoint.objective_spec),
        "aggregation_spec": asdict(checkpoint.aggregation_spec),
        "samples": {
            "rho_values": checkpoint.rho_values,
            "alpha_values": checkpoint.alpha_values,
            "ky_indices": checkpoint.ky_indices,
        },
        "topology_contracts": [asdict(value) for value in checkpoint.topology_contracts],
        "provider_provenance": checkpoint.provider_provenance,
        "parameters": checkpoint.parameters,
        "sample_objectives": checkpoint.sample_objectives,
        "aggregate_objective": checkpoint.aggregate_objective,
        "history": checkpoint.history,
        "code_revision": checkpoint.code_revision,
        "dependency_revisions": checkpoint.dependency_revisions,
        "command": checkpoint.command,
        "random_seed": checkpoint.random_seed,
    }


def _topology_contract_from_payload(payload) -> OptimizationTopologyContract:
    values = dict(payload)
    values["velocity_shape"] = tuple(values["velocity_shape"])
    values["fourier_shape"] = tuple(values["fourier_shape"])
    provider_topology = list(values["provider_topology"])
    if provider_topology:
        provider_topology[-1] = tuple(provider_topology[-1])
    values["provider_topology"] = tuple(provider_topology)
    return OptimizationTopologyContract(**values)


def _float_tuple(values, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in np.ravel(np.asarray(values)))
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    converted = _json_value(dict(value))
    if not isinstance(converted, dict):  # pragma: no cover - defensive type narrowing
        raise TypeError("expected a mapping")
    return converted


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("checkpoint metadata must not contain non-finite values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"checkpoint metadata value {type(value).__name__} is not JSON serializable")


__all__ = [
    "OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION",
    "OptimizationCheckpoint",
    "assert_checkpoint_topology",
    "build_optimization_checkpoint",
    "load_optimization_checkpoint",
    "write_optimization_checkpoint",
]
