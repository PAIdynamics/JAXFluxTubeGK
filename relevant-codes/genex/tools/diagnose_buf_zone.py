import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import argparse
from misc import load_RZ, load_phi, load_bufzone, get_param, nml_to_bool
from plot_style import set_mpl_rcParams, get_boxlims
from dependencies import check_dependencies

check_dependencies()
set_mpl_rcParams()

def plot_buf_zone():
    # Plot buffer zone array in RZ space
    global plot_container

    # Save plot collection on init
    if plot_container[0] == 0:
        # Use the plot function with blank linestyle since scatter doesn't have
        # the routines `set_x / ydata` for updating the plane number
        plot_container[0] = ax.plot(R[plane, not_buf_zone[plane, :]], \
                                    Z[plane, not_buf_zone[plane, :]], \
                                    linestyle='', marker='+', c='0.8', \
                                    label='Not buffer zone')
        plot_container[1] = ax.plot(R[plane, zone_bndry[plane, :]], \
                                    Z[plane, zone_bndry[plane, :]], \
                                    linestyle='', marker='+', c='b', \
                                    label='Boundary buffer')
        plot_container[2] = ax.plot(R[plane, zone_axis[plane, :]], \
                                    Z[plane, zone_axis[plane, :]], \
                                    linestyle='', marker='+', c='r', \
                                    label='Axis buffer')
        plot_container[3] = ax.plot(R[plane, zone_parcon[plane, :]], \
                                    Z[plane, zone_parcon[plane, :]], \
                                    linestyle='', marker='+', c='g', \
                                    label='Parallel connection buffer')
        plot_container[4] = ax.plot(R[plane, ~not_ghost[plane, :]], \
                                    Z[plane, ~not_ghost[plane, :]], \
                                    linestyle='', marker='+', c='m', \
                                    alpha=0.4, label='Ghost points')
        plot_container[5] = ax.plot(R[plane, ~not_in_target[plane, :]], \
                                    Z[plane, ~not_in_target[plane, :]], \
                                    linestyle='', marker='+', c='y', \
                                    alpha=0.6, label='Target points')
        ax.legend()
        ax.set_xlabel('$R$')
        ax.set_ylabel('$Z$')
        ax.set(**get_boxlims(R, Z))

    # Otherwise update collection data when plane is updated
    else:
        plot_container[0][0].set_xdata(R[plane, not_buf_zone[plane, :]])
        plot_container[0][0].set_ydata(Z[plane, not_buf_zone[plane, :]])
        plot_container[1][0].set_xdata(R[plane, zone_bndry[plane, :]])
        plot_container[1][0].set_ydata(Z[plane, zone_bndry[plane, :]])
        plot_container[2][0].set_xdata(R[plane, zone_axis[plane, :]])
        plot_container[2][0].set_ydata(Z[plane, zone_axis[plane, :]])
        plot_container[3][0].set_xdata(R[plane, zone_parcon[plane, :]])
        plot_container[3][0].set_ydata(Z[plane, zone_parcon[plane, :]])
        plot_container[4][0].set_xdata(R[plane, ~not_ghost[plane, :]])
        plot_container[4][0].set_ydata(Z[plane, ~not_ghost[plane, :]])
        plot_container[5][0].set_xdata(R[plane, ~not_in_target[plane, :]])
        plot_container[5][0].set_ydata(Z[plane, ~not_in_target[plane, :]])
    ax.set_title(r'Buffer zone at $\varphi={:.03f}\pi$'.format(phi[plane]))

def press(event):
    # Update plane number with right and left arrow keys
    global plane
    sys.stdout.flush()

    # Increase or decrease plane
    if event.key == 'd' or event.key == 'right':
        plane = (plane + 1) % n_planes
    elif event.key == 'a' or event.key == 'left':
        plane = (plane - 1) % n_planes

    print("     Displaying poloidal plane", plane + 1)
    plot_buf_zone()
    fig.canvas.draw()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--plane", type=int,
                    help="display given poloidal plane")
parser.add_argument('path', type=str, nargs=1,
                    help='path to the output directory')
args = parser.parse_args()

grid_file = args.path[0] + "mesh.nc"
R, Z, xind, yind, _, not_ghost, not_in_target = \
    load_RZ(grid_file, mask_ghosts=False, mask_target=False, return_masks=True)
phi = load_phi(grid_file)

# Load buffer zone and prepare masks
buf_zone = load_bufzone(grid_file)
not_buf_zone = np.logical_and(buf_zone == 0, np.logical_and(not_ghost, not_in_target))
zone_bndry = buf_zone == 1
zone_axis = buf_zone == 2
zone_parcon = buf_zone == 3

# Container for plot collections which change with phi
plot_container = [0,0,0,0,0,0]

# Get plane number
n_planes = R.shape[0]
if args.plane is None:
    plane = n_planes // 2
else:
    plane = (args.plane - 1) % n_planes
print("     Displaying poloidal plane", plane + 1, "of", n_planes)

# Create figure and axes
fig, ax = plt.subplots(figsize=(9,9), subplot_kw={'aspect':'equal'})
fig.canvas.mpl_connect('key_press_event', press)

# Flip y-axis (=Z) if axis is flipped in equi. This namelist only exist in the
# params_in.txt file because it is a parallax namelist
params_in_file = args.path[0] + "params_in.txt"
flip_Z = nml_to_bool(get_param(params_in_file, "flip_Z"))
if flip_Z:
    axes['mesh'].invert_yaxis()

plot_buf_zone()
plt.show()
