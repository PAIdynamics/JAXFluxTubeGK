import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import timeit
from misc import load_RZ, load_magfield, get_param
from plot_style import set_mpl_rcParams
from part import load_from_parts
from dependencies import check_dependencies

check_dependencies()
set_mpl_rcParams()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--plane", type=int,
                    help="display given poloidal plane")
parser.add_argument('path', type=str, nargs="?", default="./",
                    help='path to the output directory')
args = parser.parse_args()

grid_file      = args.path + "mesh.nc"
params_file    = args.path + "params_out.txt"
params_in_file = args.path + "params_in.txt"

time = load_from_parts(args.path, "mom_0d.nc").dim_time.data
n_t = len(time)

x, y, _, _ = load_RZ(grid_file)

n_planes = x.shape[0]
if args.plane is None:
    plane = n_planes // 2
else:
    plane = (args.plane - 1) % n_planes

mf_grp = load_magfield(grid_file)
rho = np.array(mf_grp["rho"][()])

species = get_param(params_file, "names")
spec = 0

print( "Welcome to the GENE-X timetrace analysis tool!")
print( "")
print(f"    We received the following species: {species}")
print(f"    Displaying plane {plane + 1} of {n_planes}")
print( "")

width  = 16
height = 9
fig, ax = plt.subplots(1, 1, figsize=(width, height))

fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

field_name_dict = {
    "n":      {"symbol":          r"$n$",
               "unit_normalized": r"$n_\mathrm{ref}$"},
    "u_par":  {"symbol":          r"$u_{||}$",
               "unit_normalized": r"$v_\mathrm{{th,{}}}$"},
    "E_par":  {"symbol":          r"$W_{||}$",
               "unit_normalized": r"$n_\mathrm{ref} T_\mathrm{ref}$"},
    "E_perp": {"symbol":          r"$W_{\bot}$",
               "unit_normalized": r"$n_\mathrm{ref} T_\mathrm{ref}$"},
    }

def plot_timetrace(k, n):
    spec_data = load_from_parts(args.path, "mom_2d.nc", group=species[n])

    # Currently we simply select a point at the outboard side x=R > 1.0
    # (axis location) which is close to the Z=y=0 line which is on the
    # 0.99 rho flux surface.
    # TODO: Extend diagnostic to allow point selection
    LL = np.logical_and(np.abs(y[k, :]) < 0.1, x[k, :] > 1.0)
    if np.size(rho[k, LL]) > 0:
        point = np.argmin(np.abs(rho[k, LL] - 0.99))
        point = np.where(rho[k, :] == rho[k, LL][point])[0][0]
    else:
        point = 0
    print(f"    Displaying point {point} at R={x[plane, point]}, " + \
          f"Z={y[plane, point]}, rho={rho[plane, point]}\n")

    data = dict()
    start = timeit.default_timer()
    print(f"  Loading {n_t} timestamps for {species[n]}. Might take a while..")
    field_data = spec_data.isel(dim_phi=k, dim_RZ=point)
    data["n"]      = field_data["n"].values
    data["E_par"]  = field_data["E_par"].values
    data["E_perp"] = field_data["E_perp"].values
    end = timeit.default_timer()
    print(f"  Done. Elapsed time is {end - start} s\n")

    for key in data.keys():

        symbol = field_name_dict[key]["symbol"]
        unit   = field_name_dict[key]["unit_normalized"]
        label  = f"{symbol} / {unit}"

        ax.plot(field_data.dim_time, data[key], label=label)

    ax.legend()
    ax.set_title(species[n], fontweight="bold")
    ax.set_xlabel(r"t / $\tau$")

def press(event):
    global plane, spec
    sys.stdout.flush()
    change = False

    supported_keys = {"left":  "previous plane",
                      "right": "next plane",
                      "s":     "cycle through species"}

    if event == "help":
        print("Supported key press events in this diagnostics:")
        [print(f"  {k:8s}: {supported_keys[k]}") for k in supported_keys]
        print("")
        return

    if event.key not in supported_keys.keys():
        print(f"Key {event.key} not supported by this diagnostics!")
        return

    if event.key == 'right':
        plane = (plane + 1) % n_planes
        print("    Displaying plane", plane + 1)
        change = True

    elif event.key == 'left':
        plane = (plane - 1) % n_planes
        print("    Displaying plane", plane + 1)
        change = True

    elif event.key == 's':
        spec = (spec + 1) % len(species)
        print("    Switching to species", species[spec])
        change = True

    if change:
        ax.cla()
        plot_timetrace(plane, spec)

    fig.canvas.draw()

press("help")
fig.canvas.mpl_connect('key_press_event', press)
plot_timetrace(plane, spec)
plt.show()
