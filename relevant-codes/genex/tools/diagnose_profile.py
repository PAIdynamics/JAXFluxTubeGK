import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
from misc import load_RZ, create_mesh_2d, create_field_2d, get_param, nml_to_float
from part import load_from_parts
from dependencies import check_dependencies

check_dependencies()

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--record", type=int,
                    help="display given record")
parser.add_argument("-p", "--plane", type=int,
                    help="display given poloidal plane")
parser.add_argument('path', type=str, nargs=1,
                    help='path to the output directory')
args = parser.parse_args()

grid_file = args.path[0]  + "mesh.nc"
params_file = args.path[0] + "params_out.txt"

time = load_from_parts(args.path[0], "mom_0d.nc").dim_time.data
n_t = len(time)

if args.record is None:
    t = n_t - 1
else:
    t = (args.record - 1) % n_t

# Get grid
x, y, xind, yind = load_RZ(grid_file, mask_ghosts=False)

# Get plane number
n_planes = x.shape[0]
if args.plane is None:
    plane = n_planes // 2
else:
    plane = (args.plane - 1) % n_planes
print("    Displaying plane {} of {}, timestep {} of {}".format(plane + 1, n_planes, t + 1, n_t))

def plot_profiles(plane, t):
    (x_2d, y_2d) = create_mesh_2d(x[plane,:], y[plane,:])

    species = get_param(params_file, "names")
    mid_plane = np.argmin(np.abs(y_2d[:,0]))

    fig.suptitle("Profiles at Z = {:.3f}".format(y_2d[mid_plane,0]))

    for sp in range(len(species)):
        moments = load_from_parts(args.path[0], \
                        "mom_2d.nc", group=species[sp])

        n_raw = moments["n"]
        n = n_raw[t][plane][:]
        n_2d = create_field_2d(xind[plane,:], yind[plane,:], n)
        x_line = x_2d[mid_plane, :]
        n_line = n_2d[mid_plane, :]

        Epar_raw = moments["E_par"]
        Eperp_raw = moments["E_perp"]
        T_ref = nml_to_float(get_param(params_file, "T_ref"))
        T = (2.0 / 3.0) * T_ref \
                        * (Epar_raw[t][plane][:] + Eperp_raw[t][plane][:])
        T_2d = create_field_2d(xind[plane,:], yind[plane,:], T)
        T_line = T_2d[mid_plane, :]
        T_line = T_line / n_line

        sub1.set(title="density", xlabel="R / m",
            ylabel="n / $10^{19}$ $\mathregular{m^{-3}}$")
        sub1.plot(x_line, n_line, label=species[sp])
        sub1.legend()

        sub2.set(title="temperature", xlabel="R / m",
            ylabel="T / keV")
        sub2.plot(x_line, T_line, label=species[sp])
        sub2.legend()

def press(event):
    global t, plane, fig, sub1, sub2
    sys.stdout.flush()
    sub1.cla()
    sub2.cla()

    if event.key == 'down':
        if t >= 1:
            t = t - 1
            plot_profiles(plane, t)
        print("     Displaying timestep =", t + 1)
    elif event.key == 'up':
        if t < n_t - 1:
            t = t + 1
            plot_profiles(plane, t)
        print("     Displaying timestep =", t + 1)
    elif event.key == 'left':
        plane = (plane - 1) % n_planes
        plot_profiles(plane, t)
        print("     Displaying plane =", plane + 1)
    elif event.key == 'right':
        plane = (plane + 1) % n_planes
        plot_profiles(plane, t)
        print("     Displaying plane =", plane + 1)

    fig.canvas.draw()

#plot
fig, (sub1, sub2) = plt.subplots(1, 2, figsize=(12,6))
plot_profiles(plane, t)

fig.canvas.mpl_connect('key_press_event', press)

plt.show()
