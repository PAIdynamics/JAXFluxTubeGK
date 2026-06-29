import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
import numpy as np
import argparse
from misc import load_RZ, load_phi, get_mesh_extents, create_field_2d, \
                 get_param, nml_to_bool
from plot_style import set_mpl_rcParams
from part import load_from_parts, files_exist
from dependencies import check_dependencies

check_dependencies()
set_mpl_rcParams()

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--record", type=int,
                    help="display given record")
parser.add_argument("-p", "--plane", type=int,
                    help="display given poloidal plane")
parser.add_argument('path', type=str, nargs="?", default="./",
                    help='path to the output directory')
args = parser.parse_args()

grid_file      = args.path + "mesh.nc"
params_file    = args.path + "params_out.txt"
params_in_file = args.path + "params_in.txt"

if not files_exist(args.path, "mom_2d.nc"):
    raise Exception("Error: moments output files not found!")

if files_exist(args.path, "mom_2d_tpc.nc"):
    tpc_available = True
else:
    tpc_available = False

time = load_from_parts(args.path, "mom_2d.nc").dim_time.data
n_t  = len(time)

if args.record is None:
    t = n_t - 1
else:
    t = (args.record - 1) % n_t

n_ref = float(get_param(params_file, "N_REF"))
m_ref = float(get_param(params_file, "M_REF"))
T_ref = float(get_param(params_file, "T_REF"))
L_ref = float(get_param(params_file, "L_REF"))

# SI constants
echarge  = 1.60217646e-19
m_proton = 1.67262158e-27
c_ref = np.sqrt(echarge * T_ref * 1.0e3 / (m_proton * m_ref))
t_ref = L_ref / c_ref

masses  = m_ref * np.array(get_param(params_file, "masses"), dtype=np.float32)
species = get_param(params_file, "names")
if(len(species) != 2):
    print("Error: At the moment the diagnostics only supports two species")
    sys.exit()

x, y, xind, yind = load_RZ(grid_file)
phi = load_phi(grid_file)

n_planes = x.shape[0]
if args.plane is None:
    plane = n_planes // 2
else:
    plane = (args.plane - 1) % n_planes

print( "Welcome to the GENE-X moment analysis tool!")
print( "")
print(f"    We received the following species: {species}")
print(f"    Displaying plane {plane + 1} of {n_planes}, " \
      + f"timestep {t + 1} of {n_t}")
print( "")

width  = 16
height = 9
fig, axes = plt.subplots(2, 4, sharex=True, sharey=True,
                         subplot_kw={'aspect':'equal'},
                         figsize=(width, height),
                         layout="tight")
caxes = np.empty_like(axes, dtype=object)

fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

# Flip y-axis (=Z) if axis is flipped in equi. This namelist only exist in the
# params_in.txt file because it is a parallax namelist
flip_Z = nml_to_bool(get_param(params_in_file, "flip_Z"))
if flip_Z:
    origin = "upper"
else:
    origin = "lower"
extents = get_mesh_extents(x, y, flip_Z)

plot_containers = {}

fluctuations    = False
show_normalized = True
plot_tpc        = False

field_name_dict = {
    "n":      {"title":           "density",
               "symbol":          r"$n$",
               "unit":            r"($10^{19}$ m$^{-3}$)",
               "unit_normalized": r"$n_\mathrm{ref}$",
               "normalization":   n_ref},
    "u_par":  {"title":           "|| flow",
               "symbol":          r"$u_{||}$",
               "unit":            "(m s$^{-1}$)",
               "unit_normalized": r"$v_\mathrm{{th,{}}}$",
               "normalization":   c_ref},
    "E_par":  {"title":           "|| energy",
               "symbol":          r"$W_{||}$",
               "unit":            r"(keV $10^{-19}$ m$^{-3}$)",
               "unit_normalized": r"$n_\mathrm{ref} T_\mathrm{ref}$",
               "normalization":   T_ref * n_ref},
    "E_perp": {"title":           r"$\perp$ energy",
               "symbol":          r"$W_{\bot}$",
               "unit":            r"(keV $10^{-19}$ m$^{-3}$)",
               "unit_normalized": r"$n_\mathrm{ref} T_\mathrm{ref}$",
               "normalization":   T_ref * n_ref},
    }

def plot_diagnostic(t, k, n, key, pos):
    mom_2d_file = "mom_2d_tpc.nc" if plot_tpc else "mom_2d.nc"
    spec_data = load_from_parts(args.path,
                    mom_2d_file, group=species[n])
    field_data = getattr(spec_data, key).data
    record = field_data[t][k][:]
    title  = field_name_dict[key]["title"]
    symbol = field_name_dict[key]["symbol"]
    unit   = field_name_dict[key]["unit_normalized"] \
                if show_normalized else field_name_dict[key]["unit"]
    normalization  = field_name_dict[key]["normalization"]

    if key == "u_par" and show_normalized:
        normalization = np.sqrt(masses[n] / 2.0)
        unit = unit.format(species[n])

    if show_normalized:
        tval  = time[t]
        tunit = r"$\tau$"
        Lunit = r"$L_\mathrm{ref}$"

        def fmt_axis(x, pos):
            return f'{x:.1f}'
    else:
        tval  = 1e6 * t_ref * time[t]
        tunit = r"$\mu$s"
        Lunit = "m"

        def fmt_axis(x, pos):
            return f'{x*L_ref:.1f}'

    if(fluctuations):
        record = (record - field_data[0,k,:]) * normalization

        cmap = "RdBu_r"
        norm = colors.CenteredNorm()
        fig.suptitle(r"Moment $\bf{fluctuations}$ at " + \
                     fr"$\phi$ = {phi[plane]:.3f}$\pi$, t = {tval:.5f} {tunit}")
        axtitle = f"{title} ({species[n]})"
        caxtitle = f"{symbol} - {symbol}$_0$ / {unit}"
    else:
        record *= normalization

        if key == "u_par":
            cmap = "RdBu_r"
            norm = colors.CenteredNorm()
        else:
            cmap = "plasma"
            norm = None
        fig.suptitle(r"Moments at " + \
                     fr"$\phi$ = {phi[plane]:.3f}$\pi$, t = {tval:.5f} {tunit}")
        axtitle = f"{title} ({species[n]})"
        caxtitle = f"{symbol} / {unit}"

    record_2d = create_field_2d(xind[k, :], yind[k, :], record)

    ax = axes[n, pos]
    cax = caxes[n, pos]
    if not is_initialized:
        ax.set_title(axtitle, loc="left")

        c1 = ax.imshow(record_2d, extent=extents[k], \
                       cmap=cmap, norm=norm, origin=origin)
        # Create colorbar axis only once, otherwise use the already-created one
        if not cax:
            clb = fig.colorbar(c1, ax=ax)
            clb.ax.set_title(caxtitle, ha="center", fontsize=8)
            caxes[n, pos] = clb.ax
        else:
            clb = fig.colorbar(c1, cax=cax)
        plot_containers[key + str(n)] = c1
    else:
        ax.set_title(axtitle, loc="left")
        caxes[n, pos].set_title(caxtitle, ha="center", fontsize=8)
        c1 = plot_containers[key + str(n)]
        c1.set_array(record_2d)
        c1.set(cmap=cmap, norm=norm)
        c1.autoscale()

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_axis))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_axis))

    # Set labels only for the outermost plots
    if n == 1:
        ax.set_xlabel(f"$R$ / {Lunit}")
    if pos == 0:
        ax.set_ylabel(f"$Z$ / {Lunit}")

def plot_all(t, k):
    plot_diagnostic(t, k, 0, "n",      0)
    plot_diagnostic(t, k, 0, "u_par",  1)
    plot_diagnostic(t, k, 0, "E_par",  2)
    plot_diagnostic(t, k, 0, "E_perp", 3)
    plot_diagnostic(t, k, 1, "n",      0)
    plot_diagnostic(t, k, 1, "u_par",  1)
    plot_diagnostic(t, k, 1, "E_par",  2)
    plot_diagnostic(t, k, 1, "E_perp", 3)

def press(event):
    global t, plane, fluctuations, show_normalized, plot_tpc
    sys.stdout.flush()

    supported_keys = {"down":  "previous timestep",
                      "up"  :  "next timestep",
                      "left":  "previous plane",
                      "right": "next plane",
                      "d":     "toggle fluctuations",
                      "n":     "toggle normalization",
                      "t":     "toggle trapped particle diagnostics"}

    if event == "help":
        print("Supported key press events in this diagnostics:")
        [print(f"  {k:8s}: {supported_keys[k]}") for k in supported_keys]
        return

    if event.key not in supported_keys.keys():
        print(f"Key {event.key} not supported by this diagnostics!")
        return

    if event.key == 'down':
        if t >= 1:
            t = t - 1
            plot_all(t, plane)
        print("    Displaying timestep", t + 1)

    elif event.key == 'up':
        if t < n_t - 1:
            t = t + 1
            plot_all(t, plane)
        print("    Displaying timestep", t + 1)

    elif event.key == 'right':
        plane = (plane + 1) % n_planes
        plot_all(t, plane)
        print("    Displaying plane", plane + 1)

    elif event.key == 'left':
        plane = (plane - 1) % n_planes
        plot_all(t, plane)
        print("    Displaying plane", plane + 1)

    elif event.key == 'd':
        fluctuations = not fluctuations
        plot_all(t, plane)
        print("    Switching to fluctuation mode", fluctuations)

    elif event.key == 'n':
        show_normalized = not show_normalized
        plot_all(t, plane)
        print("    Showing normalized quantities:", show_normalized)

    elif event.key == 't':
        if tpc_available:
            plot_tpc = not plot_tpc
            plot_all(t, plane)
            print("    Displaying TPC particles:")
        else:
            print("Error: TPC output files not found!")

    fig.canvas.draw()

press("help")
fig.canvas.mpl_connect('key_press_event', press)
is_initialized = False
plot_all(t, plane)
is_initialized = True
plt.show()
