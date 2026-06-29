import h5netcdf
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import argparse
from misc import load_RZ, load_phi, load_velspace, get_param, nml_to_bool
from plot_style import set_mpl_rcParams, get_boxlims
from dependencies import check_dependencies

check_dependencies()
set_mpl_rcParams()

def display_header(grid_file, verbosity):
    ds = h5netcdf.File(grid_file, 'r')

    print("File {}:".format(grid_file))

    for name in ds.attrs:
        print('\t%s: %s' % (name, ds.attrs[name]))

    dimnames = ["{}({})".format(dimname, len(ds.dimensions[dimname]))
                for dimname in ds.dimensions.keys()]
    print("\tdimensions(sizes):", ", ".join(dimnames))

    for group_name, group in ds.groups.items():
        print("\tgroup: {}".format(group_name))

        if len(group.dimensions.keys()) > 0:
            dimnames = ["{}({})".format(dimname, len(group.dimensions[dimname]))
                        for dimname in group.dimensions.keys()]
            print("\t\tdimensions(sizes):", ", ".join(dimnames))

        if verbosity >= 2 and len(group.variables.keys()) > 0:
            varnames = [str(group.variables[varname].dtype)+' '+str(varname)+
                        ((str(group.variables[varname].dimensions)).replace(",)",")")).replace("'","")
                        for varname in group.variables.keys()]
            print("\t\tvariables(dimensions):")
            for var in varnames:
                print("\t\t\t" + var)

        if len(group.groups.keys()) > 0:
            subgroup_list = list(group.groups.keys())
            unique_prefixes = np.unique([subgroup_name.split('_plane')[0]
                                        for subgroup_name in subgroup_list])
            for prefix in unique_prefixes:
                group_list = [subgroup for subgroup in subgroup_list
                                               if subgroup.startswith(prefix)]
                is_sorted = all([int(subgroup.split('_plane')[-1]) == i
                                for subgroup, i in zip(group_list,
                                                       np.arange(1, len(group_list) + 1))])
                if is_sorted and len(group_list) > 1:
                    print("\t\tgroups: {}..{}".format(group_list[0],
                                           group_list[-1].split('_plane')[-1]))
                else:
                    print("\t\tgroups:", ", ".join([group for group in group_list]))

def plot_mesh():
    # Plot the mesh points in RZ space
    global meshplot

    # Save plot collection on init
    if meshplot[0] == 0:
        # Use the plot function with blank linestyle since scatter doesn't have
        # the routines `set_x / ydata` for updating the plane number
        meshplot[0] = axes['mesh'].plot(R[plane, is_compute[plane, :]], \
                                        Z[plane, is_compute[plane, :]], \
                                        linestyle='', marker='+', c='b', \
                                        label='Compute points')
        meshplot[1] = axes['mesh'].plot(R[plane, ~not_ghost[plane, :]], \
                                        Z[plane, ~not_ghost[plane, :]], \
                                        linestyle='', marker='+', c='y', \
                                        label='Ghost points')
        meshplot[2] = axes['mesh'].plot(R[plane, ~not_in_target[plane, :]], \
                                        Z[plane, ~not_in_target[plane, :]], \
                                        linestyle='', marker='+', c='m', \
                                        label='Target points')
        axes['mesh'].legend()
        axes['mesh'].set_xlabel('$R$')
        axes['mesh'].set_ylabel('$Z$')
        axes['mesh'].set(**get_boxlims(R, Z))

    # Otherwise update collection data when plane is updated
    else:
        meshplot[0][0].set_xdata(R[plane, is_compute[plane, :]])
        meshplot[0][0].set_ydata(Z[plane, is_compute[plane, :]])
        meshplot[1][0].set_xdata(R[plane, ~not_ghost[plane, :]])
        meshplot[1][0].set_ydata(Z[plane, ~not_ghost[plane, :]])
        meshplot[2][0].set_xdata(R[plane, ~not_in_target[plane, :]])
        meshplot[2][0].set_ydata(Z[plane, ~not_in_target[plane, :]])
    axes['mesh'].set_title(r'Mesh at $\varphi={:.03f}\pi$'.format(phi[plane]))

def plot_phi_vp_mu():
    # Plot 1D grids for phi, vp, and mu
    for ax_label, data, title in zip(['phi', 'vp', 'mu'], \
                                     [phi, vp, mu], \
                                     [r"$\varphi$ / $\pi$", "$v_{||}$", "$\mu$"]):
        ax = axes[ax_label]
        for val in data:
            ax.axvline(x = val, c='k')
            ax.set_title(title)
            ax.yaxis.set_ticks([])

def plot_masks():
    # Combine and plot not_ghost and not_filler masks, to show where ghost and
    # filler points lie in a 2D buffer array

    # Create data array - value of 0 for compute points, 2 for ghosts, and
    # 4 for filler points
    data = np.zeros_like(not_filler, dtype=np.int32)
    data[~not_ghost] = 2
    data[~not_in_target] = 4
    data[~not_filler] = 6

    # Discrete colormap with color for each category
    cmap = mpl.colors.ListedColormap(['b', 'y', 'm', 'k'])
    norm = mpl.colors.BoundaryNorm([-1, 1, 3, 5, 7], cmap.N)

    # Plot as "image" - set interpolation to none so that edges will be sharp
    c = axes['masks'].imshow(data.T, \
                       norm=norm, cmap=cmap, \
                       origin='lower', aspect='auto', interpolation='none')

    # Colorbar to show categories
    cbar = fig.colorbar(c, ax=axes['masks'])
    cbar.set_ticks([0, 2, 4, 6])
    cbar.ax.set_yticklabels(["Computation point", "Ghost point", "Target point", "Filler point"])

    axes['masks'].set(xlabel="Phi index", ylabel="RZ index", title="Masks")
    # Since the axes are array indices, the ticks should be integers
    axes['masks'].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    axes['masks'].yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

def plot_phi_highlight():
    # Plot red lines on the mask plot and phi grid showing which plane is
    # currently displayed
    global phi_highlights

    shift = 0.4
    color = 'r'
    # Save plot collection on init
    if phi_highlights[-1] == 0:
        phi_highlights[0] = axes['masks'].axvline(x = plane - shift, c=color)
        phi_highlights[1] = axes['masks'].axvline(x = plane + shift, c=color)
        phi_highlights[2] = axes['phi'].axvline(x = phi[plane], c=color, zorder=3)
    # Otherwise update data when plane is updated
    else:
        phi_highlights[0].set_xdata(x = plane - shift)
        phi_highlights[1].set_xdata(x = plane + shift)
        phi_highlights[2].set_xdata(x = phi[plane])

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
    plot_mesh()
    plot_phi_highlight()
    fig.canvas.draw()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--plane", type=int,
                    help="display given poloidal plane")
parser.add_argument("-v", action='count', default=0,
                    help="print mesh file header to console and exit")
parser.add_argument('path', type=str, nargs=1,
                    help='path to the output directory')
args = parser.parse_args()

grid_file = args.path[0] + "mesh.nc"

if args.v > 0:
    display_header(grid_file, args.v)
    quit()

# Get grids
# Load RZ grids, masking filler points
R, Z, xind, yind, not_filler, not_ghost, not_in_target = load_RZ(
    grid_file, mask_ghosts=False, mask_target=False, return_masks=True
)
is_compute = np.logical_and(not_filler, np.logical_and(not_ghost, not_in_target))
phi = load_phi(grid_file)
vp, mu = load_velspace(grid_file)

# Container for plot collections which change with phi
meshplot = [0,0,0]
phi_highlights = [0,0,0]

# Get plane number
n_planes = R.shape[0]
if args.plane is None:
    plane = n_planes // 2
else:
    plane = (args.plane - 1) % n_planes
print("     Displaying poloidal plane", plane + 1, "of", n_planes)

# Create figure and axes
width = 15
height = 9
fig = plt.figure(figsize=(width, height))
axes = {}
axes['masks'] = plt.subplot2grid((3,4), (0,0), rowspan=3, colspan=1)
axes['mesh'] = plt.subplot2grid((3,4), (0,1), rowspan=3, colspan=2, aspect='equal')
axes['phi'] = plt.subplot2grid((3,4), (0,3), aspect=(phi.max() - phi.min())/2)
axes['vp'] = plt.subplot2grid((3,4), (1,3), aspect=(vp.max() - vp.min())/2)
axes['mu'] = plt.subplot2grid((3,4), (2,3), aspect=(mu.max() - mu.min())/2)
fig.canvas.mpl_connect('key_press_event', press)

plot_mesh()
plot_masks()
plot_phi_highlight()
plot_phi_vp_mu()

# Flip y-axis (=Z) if axis is flipped in equi. This namelist only exist in the
# params_in.txt file because it is a parallax namelist
params_in_file = args.path[0] + "params_in.txt"
flip_Z = nml_to_bool(get_param(params_in_file, "flip_Z"))
if flip_Z:
    axes['mesh'].invert_yaxis()

plt.show()
