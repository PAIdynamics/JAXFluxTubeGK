"""Visualize a W7-X VMEC stellarator surface used by the GX/stella fixtures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import netcdf_file


os.environ.setdefault("MPLCONFIGDIR", "/tmp/jax_fluxtube_gk_matplotlib")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "fixtures/stella_w7x_mode_structure_run/mode_structure_run_metadata.json"
DEFAULT_STELLA_GEOMETRY = (
    ROOT / "fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.geometry"
)
DEFAULT_OUTPUT = ROOT / "figures/w7x_vmec_geometry.png"
DEFAULT_PDF_OUTPUT = ROOT / "figures/w7x_vmec_geometry.pdf"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from jax_fluxtube_gk.external import announce_external_path

    announce_external_path("VMEC equilibrium", args.vmec)
    metadata = _read_json(args.metadata) if args.metadata.is_file() else {}
    torflux = float(args.torflux if args.torflux is not None else _metadata_torflux(metadata))

    vmec = _load_vmec(args.vmec)
    surface_index = _surface_index(torflux, vmec["rmnc"].shape[0])
    surface = _evaluate_surface(
        vmec,
        surface_index=surface_index,
        n_theta=args.n_theta,
        n_zeta=args.n_zeta,
    )
    stella_geometry = _load_stella_geometry(args.stella_geometry)
    _plot_geometry(
        vmec=vmec,
        surface=surface,
        stella_geometry=stella_geometry,
        surface_index=surface_index,
        torflux=torflux,
        metadata=metadata,
        output=args.output,
        pdf_output=args.pdf_output,
    )
    print(args.output)
    if args.pdf_output is not None:
        print(args.pdf_output)
    return 0


def _load_vmec(path: Path) -> dict[str, Any]:
    handle = netcdf_file(path, "r", mmap=False)
    try:
        variables = handle.variables

        def array(name: str) -> np.ndarray:
            return np.asarray(variables[name].data).copy()

        def scalar(name: str) -> float:
            return float(np.asarray(variables[name].data).reshape(()))

        return {
            "path": path,
            "rmnc": array("rmnc"),
            "zmns": array("zmns"),
            "xm": array("xm"),
            "xn": array("xn"),
            "bmnc": array("bmnc"),
            "xm_nyq": array("xm_nyq"),
            "xn_nyq": array("xn_nyq"),
            "nfp": int(round(scalar("nfp"))),
            "rmajor": scalar("Rmajor_p"),
            "aminor": scalar("Aminor_p"),
        }
    finally:
        handle.close()


def _evaluate_surface(
    vmec: dict[str, Any],
    *,
    surface_index: int,
    n_theta: int,
    n_zeta: int,
) -> dict[str, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=True)
    zeta = np.linspace(0.0, 2.0 * np.pi, n_zeta, endpoint=True)
    r, z = _rz_from_coefficients(
        rmnc=vmec["rmnc"][surface_index],
        zmns=vmec["zmns"][surface_index],
        xm=vmec["xm"],
        xn=vmec["xn"],
        theta=theta,
        zeta=zeta,
    )
    b = _cosine_series(
        coefficients=vmec["bmnc"][surface_index],
        xm=vmec["xm_nyq"],
        xn=vmec["xn_nyq"],
        theta=theta,
        zeta=zeta,
    )
    x = r * np.cos(zeta)[None, :]
    y = r * np.sin(zeta)[None, :]
    return {"theta": theta, "zeta": zeta, "R": r, "Z": z, "B": b, "X": x, "Y": y}


def _rz_from_coefficients(
    *,
    rmnc: np.ndarray,
    zmns: np.ndarray,
    xm: np.ndarray,
    xn: np.ndarray,
    theta: np.ndarray,
    zeta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phase = _phase(theta, zeta, xm, xn)
    r = np.tensordot(np.cos(phase), rmnc, axes=([2], [0]))
    z = np.tensordot(np.sin(phase), zmns, axes=([2], [0]))
    return r, z


def _cosine_series(
    *,
    coefficients: np.ndarray,
    xm: np.ndarray,
    xn: np.ndarray,
    theta: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
    phase = _phase(theta, zeta, xm, xn)
    return np.tensordot(np.cos(phase), coefficients, axes=([2], [0]))


def _phase(theta: np.ndarray, zeta: np.ndarray, xm: np.ndarray, xn: np.ndarray) -> np.ndarray:
    return theta[:, None, None] * xm[None, None, :] - zeta[None, :, None] * xn[None, None, :]


def _load_stella_geometry(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    rows = np.loadtxt(path, comments="#")
    return {
        "path": path,
        "alpha": rows[:, 0],
        "zed": rows[:, 1],
        "zeta": rows[:, 2],
        "bmag": rows[:, 3],
        "gyy": rows[:, 5],
        "gxy": rows[:, 6],
        "gxx": rows[:, 7],
        "gradb_drift": rows[:, 8],
        "curvature_drift": rows[:, 9],
    }


def _plot_geometry(
    *,
    vmec: dict[str, Any],
    surface: dict[str, np.ndarray],
    stella_geometry: dict[str, np.ndarray] | None,
    surface_index: int,
    torflux: float,
    metadata: dict[str, Any],
    output: Path,
    pdf_output: Path | None,
) -> None:
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.cm import ScalarMappable

    output.parent.mkdir(parents=True, exist_ok=True)
    if pdf_output is not None:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.2, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.28, 1.28, 1.0), height_ratios=(1.0, 1.0))

    ax_surface = fig.add_subplot(grid[:, :2], projection="3d")
    ax_cross = fig.add_subplot(grid[0, 2])
    ax_tube = fig.add_subplot(grid[1, 2])

    b = surface["B"]
    norm = colors.Normalize(vmin=float(np.nanmin(b)), vmax=float(np.nanmax(b)))
    cmap = plt.get_cmap("viridis")
    ax_surface.plot_surface(
        surface["X"],
        surface["Y"],
        surface["Z"],
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        facecolors=cmap(norm(b)),
        alpha=0.96,
    )
    ax_surface.set_title(
        f"W7-X VMEC flux surface, torflux={torflux:.2f}\n"
        f"GX benchmark equilibrium, nfp={vmec['nfp']}",
        fontsize=11,
    )
    ax_surface.set_xlabel("X [m]")
    ax_surface.set_ylabel("Y [m]")
    ax_surface.set_zlabel("Z [m]")
    ax_surface.set_proj_type("ortho")
    ax_surface.xaxis.set_major_locator(MaxNLocator(5))
    ax_surface.yaxis.set_major_locator(MaxNLocator(5))
    ax_surface.zaxis.set_major_locator(MaxNLocator(5))
    ax_surface.tick_params(labelsize=8, pad=1)
    ax_surface.view_init(elev=24.0, azim=42.0)
    _set_3d_box_aspect(ax_surface, surface["X"], surface["Y"], surface["Z"])

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_surface,
        shrink=0.62,
        pad=0.04,
        fraction=0.05,
    )
    colorbar.set_label("|B| [VMEC units]")

    theta = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=True)
    cut_angles = np.array([0.0, np.pi / (2 * vmec["nfp"]), np.pi / vmec["nfp"]])
    cut_labels = ("0", r"\pi/(2N_{\rm fp})", r"\pi/N_{\rm fp}")
    cut_colors = ("#3255a4", "#d14f2a", "#2c7f5b")
    for zeta, label, color in zip(cut_angles, cut_labels, cut_colors, strict=True):
        r, z = _rz_from_coefficients(
            rmnc=vmec["rmnc"][surface_index],
            zmns=vmec["zmns"][surface_index],
            xm=vmec["xm"],
            xn=vmec["xn"],
            theta=theta,
            zeta=np.array([zeta]),
        )
        ax_cross.plot(r[:, 0], z[:, 0], color=color, lw=1.8, label=rf"$\zeta={label}$")
    ax_cross.set_title("Poloidal cuts")
    ax_cross.set_xlabel("R [m]")
    ax_cross.set_ylabel("Z [m]")
    ax_cross.set_aspect("equal", adjustable="box")
    ax_cross.grid(True, lw=0.4, alpha=0.35)
    ax_cross.legend(frameon=False, fontsize=8, loc="best")

    if stella_geometry is not None:
        zed_over_2pi = stella_geometry["zed"] / (2.0 * np.pi)
        ax_tube.plot(zed_over_2pi, stella_geometry["bmag"], color="#202020", lw=1.7)
        ax_tube.fill_between(
            zed_over_2pi,
            stella_geometry["bmag"],
            float(np.min(stella_geometry["bmag"])),
            color="#6fa8dc",
            alpha=0.22,
            lw=0.0,
        )
        ax_tube.set_title("stella flux-tube |B| sample")
        ax_tube.set_xlabel(r"$z/(2\pi)$")
        ax_tube.set_ylabel(r"$B/B_{\rm ref}$")
        ax_tube.grid(True, lw=0.4, alpha=0.35)
    else:
        ax_tube.text(0.5, 0.5, "No stella geometry table found", ha="center", va="center")
        ax_tube.set_axis_off()

    fig.suptitle(_figure_title(vmec, metadata), fontsize=12)
    fig.savefig(output, dpi=220)
    if pdf_output is not None:
        fig.savefig(pdf_output)
    plt.close(fig)


def _set_3d_box_aspect(ax: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    spans = np.array([np.ptp(x), np.ptp(y), np.ptp(z)], dtype=float)
    spans[spans == 0.0] = 1.0
    ax.set_box_aspect(spans)


def _figure_title(vmec: dict[str, Any], metadata: dict[str, Any]) -> str:
    source = Path(str(metadata.get("vmec_source") or vmec["path"])).name
    return f"Stellarator geometry from VMEC: {source}"


def _surface_index(torflux: float, n_surfaces: int) -> int:
    bounded = min(max(torflux, 0.0), 1.0)
    return int(round(bounded * (n_surfaces - 1)))


def _metadata_torflux(metadata: dict[str, Any]) -> float:
    geometry = metadata.get("geometry", {})
    return float(geometry.get("torflux", 0.64))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmec", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--stella-geometry", type=Path, default=DEFAULT_STELLA_GEOMETRY)
    parser.add_argument("--torflux", type=float, default=None)
    parser.add_argument("--n-theta", type=int, default=96)
    parser.add_argument("--n-zeta", type=int, default=160)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
