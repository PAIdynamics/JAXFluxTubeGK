import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import argparse
from misc import load_RZ, load_parcon, get_mesh_extents, create_field_2d, \
                 get_param, nml_to_bool
from plot_style import set_mpl_rcParams, get_boxlims
from dependencies import check_dependencies

check_dependencies()
set_mpl_rcParams()

def axes_prep(init=False):
    """
    Prepare plot and colorbar axes. On initialization, create colorbar axes
    by reserving space from each corresponding axis (created using
    plt.subplots). Otherwise, clear all axes to prepare for new data
    """
    global axes, cbar_axes
    boxlims = get_boxlims(R, Z)
    if init:
        cbar_axes = np.empty((nrows,ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i,j]
                ax.set(**boxlims)
                if i == 1 and j != 2:
                    # Only make colorbar axes for field-line length plots
                    ax_divider = make_axes_locatable(ax)
                    cbar_axes[i,j] = ax_divider.append_axes("right",
                                                            size="7%", pad="2%")
    else:
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                ax = axes[i,j]
                ax.cla()
                ax.set(**boxlims)
                if i == 1 and j != 2:
                    cbar_axes[i,j].cla()

def plot_all():
    """
    Plot all parallel connection information from the mesh file:
      (1) parcon_negative2 - whether a gridpoint connects target and compute
                             region in the previous (k - 2) plane
      (2) parcon_negative1 - whether a gridpoint connects target and compute
                             region in the previous (k - 1) plane
      (3) not_in_target    - whether a gridpoint does not lie within the target
      (4) parcon_positive1 - whether a gridpoint connects target and compute
                             region in the next (k + 1) plane
      (5) parcon_positive2 - whether a gridpoint connects target and compute
                             region in the next (k + 2) plane
      (6) fll_negative2    - field line lengths from each grid point to the
                             previous (k - 2) plane
      (7) fll_negative1    - field line lengths from each grid point to the
                             previous (k - 1) plane
      (8) maps_on_mesh     - whether a grid point maps onto all neighboring
                             meshes [k - 2, k + 2]
      (9) fll_positive1    - field line lengths from each grid point to the
                             next (k + 1) plane
      (10) fll_positive2   - field line lengths from each grid point to the
                             next (k + 2) plane
    """

    if not include_ghosts:
        ghost_mask = not_ghost[plane, :]
    else:
        # Show all points, including ghosts: mask is completely True
        ghost_mask = np.ones_like(not_ghost[plane, :], dtype=bool)

    xind_plane = xind[plane, ghost_mask]
    yind_plane = yind[plane, ghost_mask]

    # not_in_target mask
    data_2d = create_field_2d(xind_plane, yind_plane,
                              quants[0,2][plane, ghost_mask])
    c = axes[0,2].imshow(data_2d, extent=extents[plane],
                         norm=norm_true_false, cmap=cmap_true_false,
                         interpolation='none', origin='lower')
    axes[0,2].legend(legend_handles[:2], legend_labels_true_false)
    axes[0,2].set(title=titles[0,2])

    # Plot the maps_on_mesh mask
    data_2d = create_field_2d(xind_plane, yind_plane,
                              quants[1,2][plane, ghost_mask])
    c = axes[1,2].imshow(data_2d, extent=extents[plane],
                         norm=norm_true_false, cmap=cmap_true_false,
                         interpolation='none', origin='lower')
    axes[1,2].legend(legend_handles[:2], legend_labels_true_false)
    axes[1,2].set(xlabel="$R$", title=titles[1,2])

    for i in [0, 1, 3, 4]:
        # Parcon mask
        data_2d = create_field_2d(xind_plane, yind_plane,
                                  quants[0,i][plane, ghost_mask])
        c = axes[0,i].imshow(data_2d, extent=extents[plane],
                             norm=norm_parcon, cmap=cmap_parcon,
                             interpolation='none', origin='lower')
        if i == 0:
            axes[0,i].legend(legend_handles, legend_labels_parcon, fontsize=7)
        axes[0,i].set(title = titles[0,i])

        # Field line lengths
        data_2d = create_field_2d(xind_plane, yind_plane,
                                  quants[1,i][plane, ghost_mask])
        c = axes[1,i].imshow(data_2d, extent=extents[plane],
                             interpolation='none', origin='lower')
        cbar = fig.colorbar(c, cax=cbar_axes[1,i])
        axes[1,i].set(xlabel="$R$", title=titles[1,i])

    axes[0,0].set(ylabel="$Z$")
    axes[1,0].set(ylabel="$Z$")

def press(event):
    """
    Update plane number with arrow keys or switch whether to display ghost points
    """

    global plane, include_ghosts
    sys.stdout.flush()

    # Switch to display ghosts
    if event.key == 'g':
        include_ghosts = not include_ghosts
        print("     Switching to display ghosts = ", include_ghosts)
    # Increase or decrease plane
    elif event.key == 'right':
        plane = (plane + 1) % n_planes
        print("     Displaying poloidal plane", plane + 1)

    elif event.key == 'left':
        plane = (plane - 1) % n_planes
        print("     Displaying poloidal plane", plane + 1)

    axes_prep()
    plot_all()
    fig.canvas.draw()


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--plane", type=int,
                    help="display given poloidal plane")
parser.add_argument('path', type=str, nargs=1,
                    help='path to the output directory')
args = parser.parse_args()

grid_file = args.path[0] + "mesh.nc"
parcon_grp = load_parcon(grid_file)

# Get grid
R, Z, xind, yind, _, not_ghost, _ = load_RZ(
    grid_file, mask_ghosts=False, mask_target=False, return_masks=True
)
extents = get_mesh_extents(R, Z)

# Get parallel connection data
nrows, ncols = (2, 5)
quants = np.empty((nrows, ncols), dtype=object)
titles = np.empty((nrows, ncols), dtype=object)
titles[:,:] = [["parcon_negative2", "parcon_negative1", "not_in_target", \
                "parcon_positive1", "parcon_positive2"],
               ["fll_negative2",    "fll_negative1",    "maps_on_mesh",
                "fll_positive1",    "fll_positive2"]]
for i in range(nrows):
    for j in range(ncols):
        quants[i,j] = parcon_grp[titles[i,j]][()]

# Get plane number
n_planes = R.shape[0]
if args.plane is None:
    plane = n_planes // 2
else:
    plane = (args.plane - 1) % n_planes
print("     Displaying poloidal plane", plane + 1, "of", n_planes)

# Set default plot settings
include_ghosts = True

fig, axes = plt.subplots(nrows, ncols, figsize=(18,8), \
                         subplot_kw=dict(aspect='equal'), \
                         sharex=True, sharey=True)
fig.canvas.mpl_connect('key_press_event', press)

# Make colormaps for bool arrays
color_array = np.array([
    [1.0, 0.7, 0.0, 1.0], # Orange for False / NO_PARCON
    [0.0, 0.0, 1.0, 1.0], # Blue for True / PARCON_COMPUTE_TO_TARGET
    [0.0, 1.0, 1.0, 1.0], # Green for PARCON_TARGET_TO_COMPUTE
])
legend_handles = [
    mpl.lines.Line2D([0], [0], c=color_array[0], marker="s", ls=""),
    mpl.lines.Line2D([0], [0], c=color_array[1], marker="s", ls=""),
    mpl.lines.Line2D([0], [0], c=color_array[2], marker="s", ls="")
]
bounds = [-0.5, 0.5, 1.5, 2.5]

cmap_true_false = mpl.colors.ListedColormap(color_array[:-1])
norm_true_false = mpl.colors.BoundaryNorm(bounds[:-1], cmap_true_false.N)
legend_labels_true_false = ["False", "True"]

cmap_parcon = mpl.colors.ListedColormap(color_array)
norm_parcon = mpl.colors.BoundaryNorm(bounds, cmap_parcon.N)
legend_labels_parcon = ["No PC", "Comp. -> Tar.", "Tar. -> Comp."]

# Flip y-axis (=Z) if axis is flipped in equi. This namelist only exist in the
# params_in.txt file because it is a parallax namelist
params_in_file = args.path[0] + "params_in.txt"
flip_Z = nml_to_bool(get_param(params_in_file, "flip_Z"))
if flip_Z:
    for ax in axes:
        ax.invert_yaxis()

axes_prep(init=True)
plot_all()
plt.show()
