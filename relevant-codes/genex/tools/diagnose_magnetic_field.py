import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import argparse
from misc import load_RZ, load_magfield, create_mesh_2d, create_field_2d, \
                 get_param, nml_to_bool
from plot_style import set_mpl_rcParams, get_boxlims
from dependencies import check_dependencies

check_dependencies()
set_mpl_rcParams()

def axes_init():
    global axes, cbar_axes
    axes = []
    # Colorbar axes, so that they can be re-used when switching planes
    cbar_axes = []
    # Add 1 big 2x2 axes
    axes.append(plt.subplot2grid((4,4), (0,0), rowspan=2, colspan=2))
    # Add 4 smaller ones right to the big
    for i in range(2):
        for j in range(2):
            axes.append(plt.subplot2grid((4,4), (i,2+j)))
    # Fill the lower rows of the 4x4 field with smaller axes
    for i in range(2):
        for j in range(4):
            axes.append(plt.subplot2grid((4,4), (2+i,j)))

    if not mode_midline:
        boxlims = get_boxlims(R, Z)
        for ax in axes:
            # Set data limits to comfortably include mesh on all planes
            ax.set(**boxlims)
            # Create colorbar axes, using make_axes_locatable so that the height
            # of the colorbar matches the height of the figure
            ax_divider = make_axes_locatable(ax)
            cbar_axes.append(ax_divider.append_axes("right", size="7%", pad="2%"))

def plot_field(num, title, data, description):
    cmap = "plasma"
    record_2d = create_field_2d(xind[plane, :], yind[plane, :], data[plane, :])
    ax = axes[num]
    if(mode_midline is True):
        # Plot midplane graphs
        mid_plane = np.shape(record_2d)[0] // 2
        ax.set_title(title + " (midplane, Z = {:.2f})".format(y_2d[mid_plane, 0]))
        record_line = record_2d[mid_plane, :]
        x_line = x_2d[mid_plane, :]
        ax.grid()
        ax.plot(x_line, record_line)
        if(num == 0):
            ax.set(xlabel = "R / m", ylabel = description)
    else:
        # Plot 2D colormeshes
        ax.set_title(title)
        c = ax.pcolormesh(x_2d, y_2d, record_2d, cmap=cmap)
        col = fig.colorbar(c, cax=cbar_axes[num])
        ax.set_aspect('equal')

        if(num == 0):
            ax.set(xlabel = "R / m", ylabel = "Z / m")
            col.ax.set_title(description)

def plot_all():
    global x_2d, y_2d
    x_2d, y_2d = create_mesh_2d(R[plane, :], Z[plane, :])
    for i in range(len(quants)):
        plot_field(i, titles[i], quants[i], names[i] + " / " + units[i])

def onclick(event):
    global quants, titles, names, units
    for i in range(len(axes)):
        if (event.inaxes == axes[i]):
            highlighted = i

            print("     Highlight = ", titles[highlighted])

            # Swap elements of 0th index (big plot) with highlighted one
            quants[0], quants[highlighted] = quants[highlighted], quants[0]
            titles[0], titles[highlighted] = titles[highlighted], titles[0]
            names[0], names[highlighted] = names[highlighted], names[0]
            units[0], units[highlighted] = units[highlighted], units[0]

            plot_all()

    fig.canvas.draw()

def press(event):
    global mode_midline, plane
    sys.stdout.flush()
    # Increase or decrease plane
    if event.key == 'd' or event.key == 'right':
        plane = (plane + 1) % n_planes
        print("     Displaying poloidal plane", plane + 1)
    elif event.key == 'a' or event.key == 'left':
        plane = (plane - 1) % n_planes
        print("     Displaying poloidal plane", plane + 1)
    # Switch to change mode
    if event.key == 'm':
        mode_midline = not mode_midline
        print("     Switching to midline mode = ", mode_midline)

    axes_init()
    plot_all()
    fig.canvas.draw()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--field", type=int, default = 0,
                    help="field to plot")
parser.add_argument("-p", "--plane", type=int,
                    help="display given poloidal plane")
parser.add_argument('path', type=str, nargs=1,
                    help='path to the output directory')
args = parser.parse_args()

grid_file = args.path[0] + "mesh.nc"
mf_grp = load_magfield(grid_file)

# Get grid
R, Z, xind, yind = load_RZ(grid_file)

# Get plane number
n_planes = R.shape[0]
if args.plane is None:
    plane = n_planes // 2
else:
    plane = (args.plane - 1) % n_planes
print("     Displaying poloidal plane", plane + 1, "of", n_planes)

# State variables of the diagnostics
mode_midline = False

# Information for plots
titles = ["absB", "psi", "normb_R", "normb_Z", "normb_tor", "curl_normb_y",
          "dgyxdy_over_g", "dgyzdy_over_g", "dgyxdz_over_g","dgyzdx_over_g",
          "dabsBdx", "dabsBdz", "dabsBdy"]
names = ["|B|", "$\psi$", "$b_{R}$", "$b_{Z}$", "$b_{tor}$",
          r"$\left(\nabla \times b \right)_y$",
          r"$\frac{1}{g}\,\dfrac{dg_{yx}}{dy}$",
          r"$\frac{1}{g}\,\dfrac{dg_{yz}}{dy}$",
          r"$\frac{1}{g}\,\dfrac{dg_{yx}}{dz}$",
          r"$\frac{1}{g}\,\dfrac{dg_{yz}}{dx}$",
          r"$\dfrac{d|B|}{dx}$",
          r"$\dfrac{d|B|}{dz}$",
          r"$\dfrac{d|B|}{dy}$"]
units  = ["T", "T$\mathregular{m^{-2}}$", "1", "1", "1", "$m^{-1}$",
          "$m^{-1}$", "$m^{-1}$", "$m^{-1}$", "$m^{-1}$", "$Tm^{-1}$",
          "$Tm^{-1}$", "$Tm^{-1}$"]

# Get magnetic field data
quants = []
for title in titles:
    quants.append(np.array(mf_grp[title][()]))

# Plot
width = 16
height = 12
fig = plt.figure(figsize=(width, height))
axes_init()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', press)

# Flip y-axis (=Z) if axis is flipped in equi. This namelist only exist in the
# params_in.txt file because it is a parallax namelist
params_in_file = args.path[0] + "params_in.txt"
flip_Z = nml_to_bool(get_param(params_in_file, "flip_Z"))
if flip_Z:
    for ax in axes:
        ax.invert_yaxis()

plot_all()
plt.show()
