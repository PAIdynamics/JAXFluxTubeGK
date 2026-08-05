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
import subprocess
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
TRACE_COLUMNS_V3 = (
    "record",
    "step",
    "rhs_call",
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
V3_REQUIRED_RECORD_TERMS = REQUIRED_RECORD_TERMS + (
    ("quasineutrality", "numerator"),
    ("quasineutrality", "denominator"),
    ("normalization", "native_state_scale"),
)
V4_COEFFICIENT_RECORD_TERMS = (
    ("coefficient", "mirror_force"),
    ("coefficient", "magnetic_drift_g_y"),
    ("coefficient", "magnetic_drift_phi_y"),
    ("coefficient", "equilibrium_drive"),
    ("coefficient", "parallel_streaming"),
)
V4_REQUIRED_RECORD_TERMS = V3_REQUIRED_RECORD_TERMS + V4_COEFFICIENT_RECORD_TERMS
V5_REQUIRED_RECORD_TERMS = V4_REQUIRED_RECORD_TERMS + (("coefficient", "gyroaverage_j0"),)


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
    rhs_call_min: int | None = None
    rhs_call_max: int | None = None
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
        rhs_call: int,
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
        self.rhs_call_min, self.rhs_call_max = _minmax(
            self.rhs_call_min, self.rhs_call_max, rhs_call
        )
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
            "rhs_call_range": [self.rhs_call_min, self.rhs_call_max],
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
    provenance: dict[str, str] | None = None,
) -> dict[str, object]:
    """Return a compact JSON-ready summary of a stella RHS trace file."""

    trace_path = Path(trace_path)
    accumulators: dict[tuple[str, str], _Accumulator] = defaultdict(_Accumulator)
    steps: set[int] = set()
    iky_values: set[int] = set()
    ikx_values: set[int] = set()
    code_times: set[float] = set()
    code_dts: set[float] = set()
    rhs_calls: set[int] = set()
    rows = 0
    with trace_path.open(encoding="utf-8") as handle:
        header = tuple(handle.readline().strip().split())
        if header == TRACE_COLUMNS_V1:
            trace_format = "stellarator_gk_stella_rhs_trace_v1"
            has_velocity_weights = False
        elif header == TRACE_COLUMNS_V2:
            trace_format = "stellarator_gk_stella_rhs_trace_v2"
            has_velocity_weights = True
        elif header == TRACE_COLUMNS_V3:
            trace_format = "stellarator_gk_stella_rhs_trace_v3"
            has_velocity_weights = True
        else:
            raise ValueError(f"unexpected trace header in {trace_path}: {header}")
        columns = {name: index for index, name in enumerate(header)}
        for line_number, line in enumerate(handle, start=2):
            parts = line.split()
            if len(parts) != len(header):
                raise ValueError(f"malformed trace row {line_number} in {trace_path}")
            record = parts[columns["record"]]
            term = parts[columns["term"]]
            step = int(parts[columns["step"]])
            rhs_call = int(parts[columns["rhs_call"]]) if "rhs_call" in columns else 0
            iky = int(parts[columns["iky"]])
            ikx = int(parts[columns["ikx"]])
            iz = int(parts[columns["iz"]])
            it = int(parts[columns["it"]])
            ivmu = int(parts[columns["ivmu"]])
            iv = int(parts[columns["iv"]])
            imu = int(parts[columns["imu"]])
            species = int(parts[columns["is"]])
            vpa = float(parts[columns["vpa"]])
            mu = float(parts[columns["mu"]])
            if has_velocity_weights:
                wgts_vpa = float(parts[columns["wgts_vpa"]])
                wgts_mu = float(parts[columns["wgts_mu"]])
            else:
                wgts_vpa = None
                wgts_mu = None
            code_time = float(parts[columns["code_time"]])
            code_dt = float(parts[columns["code_dt"]])
            real = float(parts[columns["real"]])
            imag = float(parts[columns["imag"]])

            rows += 1
            steps.add(step)
            iky_values.add(iky)
            ikx_values.add(ikx)
            code_times.add(code_time)
            code_dts.add(code_dt)
            rhs_calls.add(rhs_call)
            accumulators[(record, term)].add(
                iz=iz,
                it=it,
                ivmu=ivmu,
                iv=iv,
                imu=imu,
                species=species,
                rhs_call=rhs_call,
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
    v3_missing = [
        {"record": record, "term": term}
        for record, term in sorted(V3_REQUIRED_RECORD_TERMS)
        if (record, term) not in present
    ]
    v4_missing = [
        {"record": record, "term": term}
        for record, term in sorted(V4_REQUIRED_RECORD_TERMS)
        if (record, term) not in present
    ]
    if trace_format == "stellarator_gk_stella_rhs_trace_v3" and not v4_missing:
        trace_format = "stellarator_gk_stella_rhs_trace_v4"
    v5_missing = [
        {"record": record, "term": term}
        for record, term in sorted(V5_REQUIRED_RECORD_TERMS)
        if (record, term) not in present
    ]
    if trace_format == "stellarator_gk_stella_rhs_trace_v4" and not v5_missing:
        trace_format = "stellarator_gk_stella_rhs_trace_v5"
    summary = {
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
        "rhs_calls": sorted(rhs_calls),
        "required_record_terms_present": not missing,
        "missing_record_terms": missing,
        "v3_required_record_terms_present": not v3_missing,
        "v3_missing_record_terms": v3_missing,
        "v4_required_record_terms_present": not v4_missing,
        "v4_missing_record_terms": v4_missing,
        "v5_required_record_terms_present": not v5_missing,
        "v5_missing_record_terms": v5_missing,
        "term_summaries": [
            accumulators[key].as_dict(*key)
            for key in sorted(accumulators)
        ],
    }
    if provenance is not None:
        summary["provenance"] = dict(provenance)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = summarize_trace(
        args.trace,
        provenance=_stella_provenance(args.stella_source, args.stella_executable),
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    return 0


def _stella_provenance(source: Path, executable: Path) -> dict[str, str]:
    source = source.resolve()
    executable = executable.resolve()
    if not (source / ".git").exists():
        raise ValueError(f"stella source is not a Git checkout: {source}")
    if not executable.is_file():
        raise FileNotFoundError(executable)
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=source,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    return {
        "stella_source": str(source),
        "stella_executable": str(executable),
        "stella_revision": revision,
    }


def _minmax(current_min: T | None, current_max: T | None, value: T) -> tuple[T, T]:
    if current_min is None or current_max is None:
        return value, value
    return min(current_min, value), max(current_max, value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--stella-source", type=Path, required=True)
    parser.add_argument("--stella-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
