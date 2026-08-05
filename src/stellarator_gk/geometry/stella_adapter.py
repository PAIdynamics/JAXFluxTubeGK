"""stella ``.geometry`` reader and provider adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..grids import build_parallel_grid
from ..types import ParallelGridSpec
from .flux_tube import build_physical_flux_tube_geometry_from_coordinate_arrays
from .provider import (
    GeometryProvenance,
    GeometryRequest,
    GeometryResult,
    build_geometry_metadata,
)


STELLA_GEOMETRY_COLUMNS = (
    "alpha",
    "zed",
    "zeta",
    "bmag",
    "b_dot_grad_zed",
    "g_yy",
    "g_xy",
    "g_xx",
    "B_cross_gradB_dot_grad_alpha",
    "b_cross_kappa_dot_grad_alpha",
    "B_cross_gradB_dot_grad_psi",
    "bmag_psi0",
)
STELLA_GLOBAL_COLUMNS = (
    "rhoc",
    "qinp",
    "shat",
    "rhotor",
    "aref",
    "bref",
    "dxdpsi",
    "dydalpha",
    "exb_nonlin",
    "flux_fac",
    "one_over_Grho",
)


@dataclass(frozen=True)
class StellaGeometryData:
    """Parsed stella geometry table before conversion to the public schema."""

    rows: np.ndarray
    global_header: tuple[float, ...]
    source: str
    original_n_z: int
    dropped_periodic_endpoint: bool
    field_periods: float

    def column(self, name: str) -> np.ndarray:
        return self.rows[:, STELLA_GEOMETRY_COLUMNS.index(name)]

    def global_value(self, name: str) -> float:
        return float(self.global_header[STELLA_GLOBAL_COLUMNS.index(name)])


@dataclass(frozen=True)
class StellaGeometryProvider:
    """Read an explicit stella geometry table through the common provider API."""

    path: str | Path
    nfp: int = 1
    provider_version: str = "unknown"
    revision: str = "unknown"

    def get_geometry(self, request: GeometryRequest) -> GeometryResult:
        if request.topology != "periodic" or request.endpoint_policy != "exclude":
            raise ValueError("stella geometry provider requires periodic endpoint-excluded grids")
        if (
            request.parallel_coordinate != "zed_over_2pi"
            or request.parallel_coordinate_unit != "turn"
        ):
            raise ValueError(
                "stella geometry provider requires parallel_coordinate='zed_over_2pi' "
                "with unit='turn'"
            )
        data = load_stella_geometry_data(self.path, drop_periodic_endpoint=True)
        if data.rows.shape[0] != request.n_z:
            raise ValueError(
                f"stella geometry provides n_z={data.rows.shape[0]} after endpoint handling; "
                f"request requires n_z={request.n_z}"
            )
        if not np.isclose(data.field_periods, request.field_periods, rtol=5.0e-4, atol=5.0e-4):
            raise ValueError(
                f"stella geometry spans {data.field_periods:.12g} field-line turns; "
                f"request declares {request.field_periods:.12g}"
            )
        parallel = build_parallel_grid(
            ParallelGridSpec(
                n_z=request.n_z,
                z_min=request.z_min,
                z_max=request.z_max,
                topology=request.topology,
                dtype=request.dtype,
            )
        )
        sampled = _resample_stella_rows(data, 2.0 * np.pi * np.asarray(parallel.z))
        B = sampled[:, STELLA_GEOMETRY_COLUMNS.index("bmag")]
        gb_alpha = sampled[
            :, STELLA_GEOMETRY_COLUMNS.index("B_cross_gradB_dot_grad_alpha")
        ]
        kappa_alpha = sampled[
            :, STELLA_GEOMETRY_COLUMNS.index("b_cross_kappa_dot_grad_alpha")
        ]
        gb_psi = sampled[:, STELLA_GEOMETRY_COLUMNS.index("B_cross_gradB_dot_grad_psi")]
        qinp = data.global_value("qinp")
        if qinp == 0.0:
            raise ValueError("stella global geometry header qinp must be nonzero")
        iota = 1.0 / qinp
        physical = build_physical_flux_tube_geometry_from_coordinate_arrays(
            parallel,
            theta=2.0 * np.pi * parallel.z,
            phi=sampled[:, STELLA_GEOMETRY_COLUMNS.index("zeta")],
            alpha=sampled[:, STELLA_GEOMETRY_COLUMNS.index("alpha")],
            rho=data.global_value("rhoc"),
            iota=iota,
            shear=data.global_value("shat"),
            B=B,
            b_dot_grad_z=(
                sampled[:, STELLA_GEOMETRY_COLUMNS.index("b_dot_grad_zed")]
                / (2.0 * np.pi)
            ),
            grad_psi_sq=sampled[:, STELLA_GEOMETRY_COLUMNS.index("g_xx")],
            grad_alpha_sq=sampled[:, STELLA_GEOMETRY_COLUMNS.index("g_yy")],
            grad_psi_dot_grad_alpha=sampled[:, STELLA_GEOMETRY_COLUMNS.index("g_xy")],
            B_cross_gradB_dot_grad_psi=B**2 * gb_psi,
            B_cross_gradB_dot_grad_alpha=B**2 * gb_alpha,
            b_cross_kappa_dot_grad_psi=np.zeros_like(B),
            b_cross_kappa_dot_grad_alpha=B * kappa_alpha,
            nfp=self.nfp,
            field_periods=request.field_periods,
            endpoint_policy=request.endpoint_policy,
            radial_coordinate=request.radial_coordinate,
            source="stella-geometry",
            provider="stella-geometry",
        )
        metadata = build_geometry_metadata(
            request,
            provenance=GeometryProvenance(
                provider="stella-geometry",
                provider_version=self.provider_version,
                revision=self.revision,
                source=str(Path(self.path)),
                configuration=request.configuration,
            ),
            nfp=self.nfp,
            differentiable=False,
        )
        return GeometryResult(parallel_grid=parallel, physical=physical, metadata=metadata)


def load_stella_geometry_data(
    path: str | Path,
    *,
    drop_periodic_endpoint: bool = True,
) -> StellaGeometryData:
    """Parse a stella geometry table and apply its explicit endpoint policy."""

    path = Path(path)
    global_header = None
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                stripped = stripped[1:].strip()
                if not stripped:
                    continue
            try:
                values = [float(item) for item in stripped.split()]
            except ValueError:
                continue
            if len(values) == len(STELLA_GLOBAL_COLUMNS) and global_header is None:
                global_header = tuple(values)
            elif len(values) >= len(STELLA_GEOMETRY_COLUMNS):
                rows.append(values[: len(STELLA_GEOMETRY_COLUMNS)])
    if global_header is None:
        raise ValueError(f"{path} is missing the stella global geometry header")
    if not rows:
        raise ValueError(f"{path} contains no stella geometry rows")
    array = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path} contains non-finite stella geometry values")
    original_n_z = int(array.shape[0])
    full_zeta = array[:, STELLA_GEOMETRY_COLUMNS.index("zeta")]
    field_periods = float((full_zeta[-1] - full_zeta[0]) / (2.0 * np.pi))
    dropped = bool(drop_periodic_endpoint and has_duplicate_stella_endpoint(array))
    if dropped:
        array = array[:-1]
    return StellaGeometryData(
        rows=array,
        global_header=tuple(global_header),
        source=str(path),
        original_n_z=original_n_z,
        dropped_periodic_endpoint=dropped,
        field_periods=field_periods,
    )


def has_duplicate_stella_endpoint(rows) -> bool:
    """Return whether a stella table repeats its periodic upper endpoint."""

    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] < 3:
        return False
    first = rows[0]
    last = rows[-1]
    zed_index = STELLA_GEOMETRY_COLUMNS.index("zed")
    zed_span = abs((last[zed_index] - first[zed_index]) - 2.0 * np.pi)
    periodic_columns = (
        "bmag",
        "b_dot_grad_zed",
        "g_yy",
        "g_xx",
        "B_cross_gradB_dot_grad_alpha",
        "b_cross_kappa_dot_grad_alpha",
        "bmag_psi0",
    )
    periodic_match = all(
        np.isclose(
            first[STELLA_GEOMETRY_COLUMNS.index(name)],
            last[STELLA_GEOMETRY_COLUMNS.index(name)],
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        for name in periodic_columns
    )
    return bool(zed_span <= 1.0e-3 and periodic_match)


def _resample_stella_rows(data: StellaGeometryData, target_zed: np.ndarray) -> np.ndarray:
    source_zed = data.column("zed")
    if not np.all(np.diff(source_zed) > 0.0):
        raise ValueError("stella zed grid must be strictly increasing")
    if target_zed[0] < source_zed[0] - 1.0e-3 or target_zed[-1] > source_zed[-1] + 1.0e-3:
        raise ValueError("requested z interval lies outside the stella geometry table")
    return np.column_stack(
        [np.interp(target_zed, source_zed, data.rows[:, index]) for index in range(data.rows.shape[1])]
    )
