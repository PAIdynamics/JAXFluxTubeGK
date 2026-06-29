"""Summarize a patched stella W7-X ``ky=0.3`` RHS trace.

The raw trace produced by ``prepare_stella_w7x_rhs_trace_run.py`` is too large
to version directly.  This script records the reproducibility-critical contract:
which records were exported, their grid extents, and simple complex norms.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from math import hypot, sqrt
from pathlib import Path
from typing import Iterable, TypeVar


T = TypeVar("T", int, float)


TRACE_COLUMNS_V1 = (
    "record",
    "step",
    "term",
    "iky",
    "ikx",
    "iz",
    "it",
    "ivmu",
    "iv",
    "imu",
    "is",
    "vpa",
    "mu",
    "code_time",
    "code_dt",
    "real",
    "imag",
)
TRACE_COLUMNS_V2 = (
    "record",
    "step",
    "term",
    "iky",
    "ikx",
    "iz",
    "it",
    "ivmu",
    "iv",
    "imu",
    "is",
    "vpa",
    "mu",
    "wgts_vpa",
    "wgts_mu",
    "code_time",
    "code_dt",
    "real",
    "imag",
)

REQUIRED_RECORD_TERMS = (
    ("pdf_g", "input_pdf"),
    ("phi", "field_phi"),
    ("rhs_delta", "mirror_force"),
    ("rhs_delta", "magnetic_drift_y"),
    ("rhs_delta", "magnetic_drift_x"),
    ("rhs_delta", "equilibrium_drive_wstar"),
    ("rhs_delta", "parallel_streaming"),
    ("rhs_total", "total"),
)


@dataclass
class _Accumulator:
    rows: int = 0
    sum_real: float = 0.0
    sum_imag: float = 0.0
    l2_square: float = 0.0
    max_abs: float = 0.0
    iz_min: int | None = None
    iz_max: int | None = None
    it_min: int | None = None
    it_max: int | None = None
    ivmu_min: int | None = None
    ivmu_max: int | None = None
    iv_min: int | None = None
    iv_max: int | None = None
    imu_min: int | None = None
    imu_max: int | None = None
    is_min: int | None = None
    is_max: int | None = None
    vpa_min: float | None = None
    vpa_max: float | None = None
    mu_min: float | None = None
    mu_max: float | None = None
    wgts_vpa_min: float | None = None
    wgts_vpa_max: float | None = None
    wgts_mu_min: float | None = None
    wgts_mu_max: float | None = None
    weighted_velocity_l2_square: float = 0.0

    def add(
        self,
        *,
        iz: int,
        it: int,
        ivmu: int,
        iv: int,
        imu: int,
        species: int,
        vpa: float,
        mu: float,
        wgts_vpa: float | None,
        wgts_mu: float | None,
        real: float,
        imag: float,
    ) -> None:
        self.rows += 1
        self.sum_real += real
        self.sum_imag += imag
        value_abs = hypot(real, imag)
        self.l2_square += value_abs * value_abs
        self.max_abs = max(self.max_abs, value_abs)
        self.iz_min, self.iz_max = _minmax(self.iz_min, self.iz_max, iz)
        self.it_min, self.it_max = _minmax(self.it_min, self.it_max, it)
        self.ivmu_min, self.ivmu_max = _minmax(self.ivmu_min, self.ivmu_max, ivmu)
        self.iv_min, self.iv_max = _minmax(self.iv_min, self.iv_max, iv)
        self.imu_min, self.imu_max = _minmax(self.imu_min, self.imu_max, imu)
        self.is_min, self.is_max = _minmax(self.is_min, self.is_max, species)
        self.vpa_min, self.vpa_max = _minmax(self.vpa_min, self.vpa_max, vpa)
        self.mu_min, self.mu_max = _minmax(self.mu_min, self.mu_max, mu)
        if wgts_vpa is not None and wgts_mu is not None:
            self.wgts_vpa_min, self.wgts_vpa_max = _minmax(
                self.wgts_vpa_min,
                self.wgts_vpa_max,
                wgts_vpa,
            )
            self.wgts_mu_min, self.wgts_mu_max = _minmax(
                self.wgts_mu_min,
                self.wgts_mu_max,
                wgts_mu,
            )
            self.weighted_velocity_l2_square += wgts_vpa * wgts_mu * value_abs * value_abs

    def as_dict(self, record: str, term: str) -> dict[str, object]:
        return {
            "record": record,
            "term": term,
            "rows": self.rows,
            "l2_norm": sqrt(self.l2_square),
            "max_abs": self.max_abs,
            "sum_real": self.sum_real,
            "sum_imag": self.sum_imag,
            "iz_range": [self.iz_min, self.iz_max],
            "it_range": [self.it_min, self.it_max],
            "ivmu_range": [self.ivmu_min, self.ivmu_max],
            "iv_range": [self.iv_min, self.iv_max],
            "imu_range": [self.imu_min, self.imu_max],
            "species_range": [self.is_min, self.is_max],
            "vpa_range": [self.vpa_min, self.vpa_max],
            "mu_range": [self.mu_min, self.mu_max],
            "velocity_weight_columns_present": self.wgts_vpa_min is not None,
            "wgts_vpa_range": [self.wgts_vpa_min, self.wgts_vpa_max],
            "wgts_mu_range": [self.wgts_mu_min, self.wgts_mu_max],
            "weighted_velocity_l2_norm": sqrt(self.weighted_velocity_l2_square),
        }


def summarize_trace(
    trace_path: Path,
    *,
    required_record_terms: Iterable[tuple[str, str]] = REQUIRED_RECORD_TERMS,
) -> dict[str, object]:
    """Return a compact JSON-ready summary of a stella RHS trace file."""

    trace_path = Path(trace_path)
    accumulators: dict[tuple[str, str], _Accumulator] = defaultdict(_Accumulator)
    steps: set[int] = set()
    iky_values: set[int] = set()
    ikx_values: set[int] = set()
    code_times: set[float] = set()
    code_dts: set[float] = set()
    rows = 0
    with trace_path.open(encoding="utf-8") as handle:
        header = tuple(handle.readline().strip().split())
        if header == TRACE_COLUMNS_V1:
            trace_format = "stellarator_gk_stella_rhs_trace_v1"
            has_velocity_weights = False
        elif header == TRACE_COLUMNS_V2:
            trace_format = "stellarator_gk_stella_rhs_trace_v2"
            has_velocity_weights = True
        else:
            raise ValueError(f"unexpected trace header in {trace_path}: {header}")
        for line_number, line in enumerate(handle, start=2):
            parts = line.split()
            if len(parts) != len(header):
                raise ValueError(f"malformed trace row {line_number} in {trace_path}")
            record = parts[0]
            term = parts[2]
            step = int(parts[1])
            iky = int(parts[3])
            ikx = int(parts[4])
            iz = int(parts[5])
            it = int(parts[6])
            ivmu = int(parts[7])
            iv = int(parts[8])
            imu = int(parts[9])
            species = int(parts[10])
            vpa = float(parts[11])
            mu = float(parts[12])
            if has_velocity_weights:
                wgts_vpa = float(parts[13])
                wgts_mu = float(parts[14])
                code_time = float(parts[15])
                code_dt = float(parts[16])
                real = float(parts[17])
                imag = float(parts[18])
            else:
                wgts_vpa = None
                wgts_mu = None
                code_time = float(parts[13])
                code_dt = float(parts[14])
                real = float(parts[15])
                imag = float(parts[16])

            rows += 1
            steps.add(step)
            iky_values.add(iky)
            ikx_values.add(ikx)
            code_times.add(code_time)
            code_dts.add(code_dt)
            accumulators[(record, term)].add(
                iz=iz,
                it=it,
                ivmu=ivmu,
                iv=iv,
                imu=imu,
                species=species,
                vpa=vpa,
                mu=mu,
                wgts_vpa=wgts_vpa,
                wgts_mu=wgts_mu,
                real=real,
                imag=imag,
            )

    required = tuple(required_record_terms)
    present = set(accumulators)
    missing = [
        {"record": record, "term": term}
        for record, term in sorted(required)
        if (record, term) not in present
    ]
    return {
        "trace_path": str(trace_path),
        "trace_format": trace_format,
        "rhs_units": "stella_native_rhs_times_code_dt",
        "velocity_weight_columns_present": has_velocity_weights,
        "total_rows": rows,
        "steps": sorted(steps),
        "iky_values": sorted(iky_values),
        "ikx_values": sorted(ikx_values),
        "code_times": sorted(code_times),
        "code_dts": sorted(code_dts),
        "required_record_terms_present": not missing,
        "missing_record_terms": missing,
        "term_summaries": [
            accumulators[key].as_dict(*key)
            for key in sorted(accumulators)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = summarize_trace(args.trace)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    return 0


def _minmax(current_min: T | None, current_max: T | None, value: T) -> tuple[T, T]:
    if current_min is None or current_max is None:
        return value, value
    return min(current_min, value), max(current_max, value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
