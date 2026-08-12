"""Provider-neutral reader and geometry adapter for ``eik`` tables."""

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


@dataclass(frozen=True)
class EikData:
    """Provider-neutral representation of the fields stored in an eik table."""

    theta: np.ndarray
    bmag: np.ndarray
    gradpar: np.ndarray
    gds2: np.ndarray
    gds21: np.ndarray
    gds22: np.ndarray
    gbdrift: np.ndarray
    gbdrift0: np.ndarray
    cvdrift: np.ndarray
    cvdrift0: np.ndarray
    source: str
    header: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float)
        if theta.ndim != 1 or theta.size < 2:
            raise ValueError("eik theta must be one-dimensional with at least two nodes")
        if not np.all(np.isfinite(theta)):
            raise ValueError("eik theta contains non-finite values")
        object.__setattr__(self, "theta", theta)
        for name in (
            "bmag",
            "gradpar",
            "gds2",
            "gds21",
            "gds22",
            "gbdrift",
            "gbdrift0",
            "cvdrift",
            "cvdrift0",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != theta.shape:
                raise ValueError(f"eik field {name!r} must match theta shape {theta.shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"eik field {name!r} contains non-finite values")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "header", tuple(float(value) for value in self.header))


@dataclass(frozen=True)
class EikGeometryProvider:
    """Read an explicit eik table through the common geometry-provider API."""

    path: str | Path
    iota: float = 1.0
    shear: float = 0.0
    nfp: int = 1
    provider_version: str = "unknown"
    revision: str = "unknown"

    def __post_init__(self) -> None:
        if not np.isfinite(self.iota) or self.iota == 0.0:
            raise ValueError("eik provider iota must be finite and non-zero")
        if not np.isfinite(self.shear):
            raise ValueError("eik provider shear must be finite")
        if isinstance(self.nfp, bool) or not isinstance(self.nfp, int) or self.nfp < 1:
            raise ValueError("eik provider nfp must be a positive integer")
        if not str(self.provider_version).strip() or not str(self.revision).strip():
            raise ValueError("eik provider version and revision must be non-empty")

    def get_geometry(self, request: GeometryRequest) -> GeometryResult:
        if request.topology != "periodic":
            raise ValueError("eik provider currently requires periodic topology")
        if request.parallel_coordinate_unit != "radian":
            raise ValueError("eik provider requires a radian parallel coordinate")
        data = resample_eik_data(load_eik_data(self.path), _request_z(request))
        parallel = build_parallel_grid(
            ParallelGridSpec(
                n_z=request.n_z,
                z_min=request.z_min,
                z_max=request.z_max,
                topology=request.topology,
                dtype=request.dtype,
            )
        )
        B = data.bmag
        phi = (data.theta - request.alpha) / self.iota
        physical = build_physical_flux_tube_geometry_from_coordinate_arrays(
            parallel,
            theta=data.theta,
            phi=phi,
            alpha=request.alpha,
            rho=request.radial_value,
            iota=self.iota,
            shear=self.shear,
            B=B,
            b_dot_grad_z=data.gradpar,
            grad_psi_sq=data.gds22,
            grad_alpha_sq=data.gds2,
            grad_psi_dot_grad_alpha=data.gds21,
            B_cross_gradB_dot_grad_psi=B**2 * data.gbdrift0,
            B_cross_gradB_dot_grad_alpha=B**2 * data.gbdrift,
            b_cross_kappa_dot_grad_psi=B * data.cvdrift0,
            b_cross_kappa_dot_grad_alpha=B * data.cvdrift,
            nfp=self.nfp,
            field_periods=request.field_periods,
            endpoint_policy=request.endpoint_policy,
            radial_coordinate=request.radial_coordinate,
            source=str(Path(self.path)),
            provider="eik-table",
        )
        metadata = build_geometry_metadata(
            request,
            provenance=GeometryProvenance(
                provider="eik-table",
                provider_version=self.provider_version,
                revision=self.revision,
                source=str(Path(self.path)),
                configuration=request.configuration,
            ),
            nfp=self.nfp,
            differentiable=False,
        )
        return GeometryResult(parallel_grid=parallel, physical=physical, metadata=metadata)


def load_eik_data(path: str | Path) -> EikData:
    """Load either the numeric ten-column or labelled multi-block eik layout."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if "gbdrift gradpar grho tgrid" in text:
        return _load_block_eik(path, text)
    rows = _numeric_rows(path)
    if len(rows) < 2:
        raise ValueError("eik file must contain a numeric header and data rows")
    data = np.asarray([row for row in rows[1:] if len(row) >= 10], dtype=float)
    if data.ndim != 2 or data.shape[1] < 10:
        raise ValueError("eik data rows must have at least 10 numeric columns")
    return EikData(
        theta=data[:, 0],
        bmag=data[:, 1],
        gradpar=data[:, 2],
        gds2=data[:, 3],
        gds21=data[:, 4],
        gds22=data[:, 5],
        cvdrift=data[:, 6],
        cvdrift0=data[:, 7],
        gbdrift=data[:, 8],
        gbdrift0=data[:, 9],
        source=str(path),
        header=tuple(rows[0]),
    )


def resample_eik_data(data: EikData, theta) -> EikData:
    """Interpolate an eik table onto a strictly increasing requested grid."""

    target = np.asarray(theta, dtype=float)
    source = np.asarray(data.theta, dtype=float)
    if target.ndim != 1 or target.size < 2:
        raise ValueError("requested eik theta must be one-dimensional")
    if not np.all(np.diff(source) > 0.0):
        raise ValueError("eik source theta must be strictly increasing")
    if target[0] < source[0] - 1.0e-12 or target[-1] > source[-1] + 1.0e-12:
        raise ValueError(
            "requested parallel interval lies outside the eik theta interval; "
            "generate a table covering the requested field line"
        )
    fields = {
        name: np.interp(target, source, np.asarray(getattr(data, name)))
        for name in (
            "bmag",
            "gradpar",
            "gds2",
            "gds21",
            "gds22",
            "gbdrift",
            "gbdrift0",
            "cvdrift",
            "cvdrift0",
        )
    }
    return EikData(
        theta=target,
        **fields,
        source=data.source,
        header=data.header,
    )


def _request_z(request: GeometryRequest) -> np.ndarray:
    return np.linspace(request.z_min, request.z_max, request.n_z, endpoint=False, dtype=float)


def _load_block_eik(path: Path, text: str) -> EikData:
    lines = text.splitlines()
    header = _first_numeric_row(lines)
    if len(header) < 8:
        raise ValueError("eik block header must contain at least 8 values")
    expected_rows = int(round(header[2])) + 1
    gb = _parse_block(lines, "gbdrift gradpar grho tgrid", expected_rows, 4)
    cv = _parse_block(lines, "cvdrift gds2 bmag tgrid", expected_rows, 4)
    gds = _parse_block(lines, "gds21 gds22 tgrid", expected_rows, 3)
    drift0 = _parse_block(lines, "cvdrift0 gbdrift0 tgrid", expected_rows, 3)
    theta = gb[:, 3]
    for name, values in (
        ("cvdrift", cv[:, 3]),
        ("gds21", gds[:, 2]),
        ("cvdrift0", drift0[:, 2]),
    ):
        if not np.allclose(theta, values, rtol=2.0e-12, atol=2.0e-12):
            raise ValueError(f"eik block {name!r} uses an inconsistent theta grid")
    return EikData(
        theta=theta,
        bmag=cv[:, 2],
        gradpar=gb[:, 1],
        gds2=cv[:, 1],
        gds21=gds[:, 0],
        gds22=gds[:, 1],
        gbdrift=gb[:, 0],
        gbdrift0=drift0[:, 1],
        cvdrift=cv[:, 0],
        cvdrift0=drift0[:, 0],
        source=str(path),
        header=tuple(header),
    )


def _numeric_rows(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            rows.append([float(value) for value in stripped.split()])
        except ValueError:
            continue
    return rows


def _first_numeric_row(lines: list[str]) -> list[float]:
    for line in lines:
        try:
            row = [float(value) for value in line.split()]
        except ValueError:
            continue
        if row:
            return row
    raise ValueError("eik file contains no numeric header row")


def _parse_block(
    lines: list[str],
    label: str,
    expected_rows: int,
    expected_columns: int,
) -> np.ndarray:
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == label)
    except StopIteration as exc:
        raise ValueError(f"eik file is missing block {label!r}") from exc
    rows = []
    for line in lines[start + 1 :]:
        try:
            row = [float(value) for value in line.split()]
        except ValueError:
            break
        if len(row) != expected_columns:
            break
        rows.append(row)
        if len(rows) == expected_rows:
            break
    if len(rows) != expected_rows:
        raise ValueError(
            f"eik block {label!r} expected {expected_rows} rows; got {len(rows)}"
        )
    return np.asarray(rows, dtype=float)
