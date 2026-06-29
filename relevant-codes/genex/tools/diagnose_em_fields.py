import sys
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import argparse
import scipy.stats
from misc import load_RZ, load_phi, get_mesh_extents, create_field_2d, \
                 get_param, nml_to_bool
from plot_style import set_mpl_rcParams, get_boxlims
from part import load_from_parts
from dependencies import check_dependencies

check_dependencies()
set_mpl_rcParams()

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--record", type=int,
                    help="display given record")
parser.add_argument("-p", "--plane", type=int,
                    help="display given poloidal plane")
parser.add_argument('path', type=str, nargs=1,
                    help='path to the output directory')
args = parser.parse_args()

params_file = args.path[0] + "params_out.txt"
grid_file = args.path[0]  + "mesh.nc"

# read parameter file
param_T_ref = float(get_param(params_file, "T_REF"))
param_L_ref = float(get_param(params_file, "L_REF"))
param_B_ref = float(get_param(params_file, "B_REF"))
param_m_ref = float(get_param(params_file, "M_REF"))

param_rho_ref = (9.787151386e3 * np.sqrt((param_T_ref * 1e3) / param_m_ref)
                 * param_m_ref / param_B_ref * 1.043968492e-8)

em_x = load_from_parts(args.path[0], "em_fields.nc")

time = em_x.dim_time.data
n_t = len(time)

if args.record is None:
    t = n_t - 1
else:
    t = (args.record - 1) % n_t

# get and denormalize grid
x, y, xind, yind = load_RZ(grid_file)
phi_div_pi = load_phi(grid_file)
x *= param_L_ref
y *= param_L_ref
boxlims = get_boxlims(x, y)
extents = get_mesh_extents(x, y)

# determine the number of planes
n_planes = np.shape(x)[0]
if args.plane is None:
    plane = n_planes // 2
else:
    plane = (args.plane - 1) % n_planes

print("     Displaying plane {} of {}, timestep {} of {}".format( \
        plane + 1, n_planes, t + 1, n_t))

quad_meshes = {}

def arg_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def plot_em_fields(clear_axes=False):
    # Plot phi, A_par and B_par at the globally specified plane and timestep. If
    # clear_axes = False, the previous plot container can be reused. Otherwise,
    # Otherwise, the previous plots need to be cleared

    if clear_axes:
        for ax in axes.flatten():
            ax.cla()
            ax.set(**boxlims)

    # es_pot
    data = em_x.es_pot.data
    field = data[t][plane][:] * param_T_ref
    field_2d = create_field_2d(xind[plane, :], yind[plane, :], field)

    if clear_axes:
        c = axes[0].imshow(field_2d, extent=extents[plane], \
                           cmap="RdBu", norm=colors.CenteredNorm(),
                           origin="lower")
        fig.colorbar(c, cax=cbar_axes[0])
        axes[0].set(title = "    $\mathregular{\phi_1}$ / kV",
            xlabel = "R / m", ylabel = "Z / m")
        quad_meshes["es_pot"] = c
    else:
        quad_meshes["es_pot"].set_array(field_2d)
        quad_meshes["es_pot"].set(cmap="RdBu", norm=colors.CenteredNorm())
        quad_meshes["es_pot"].autoscale()

    # A_par
    data = em_x.A_par.data
    field = data[t][plane][:] * param_rho_ref * param_B_ref
    field_2d = create_field_2d(xind[plane, :], yind[plane, :], field)

    if clear_axes:
        c = axes[1].imshow(field_2d, extent=extents[plane], \
                           cmap="RdBu", norm=colors.CenteredNorm(),
                           origin="lower")
        fig.colorbar(c, cax=cbar_axes[1])
        axes[1].set(title = "     $A_{1,\|\|}$ / (Tm)",
            xlabel = "R / m", ylabel = "Z / m")
        quad_meshes["A_par"] = c
    else:
        quad_meshes["A_par"].set_array(field_2d)
        quad_meshes["A_par"].set(cmap="RdBu", norm=colors.CenteredNorm())
        quad_meshes["A_par"].autoscale()

    if "B_par" in em_x.data_vars:
        data = em_x.B_par.data
        field = data[t][plane][:] * param_B_ref
        field_2d = create_field_2d(xind[plane, :], yind[plane, :], field)

        if clear_axes:
            c = axes[2].imshow(field_2d, extent=extents[plane], \
                            cmap="RdBu", norm=colors.CenteredNorm(),
                            origin="lower")
            fig.colorbar(c, cax=cbar_axes[2])
            axes[2].set(title = "     $B_{1,\|\|}$ / T",
                xlabel = "R / m", ylabel = "Z / m")
            quad_meshes["B_par"] = c
        else:
            quad_meshes["B_par"].set_array(field_2d)
            quad_meshes["B_par"].set(cmap="RdBu", norm=colors.CenteredNorm())
            quad_meshes["B_par"].autoscale()
    fig.suptitle((r"Electromagnetic fields for $\varphi = ${:.3f}$\pi$, "
                  r"t = {:.5f} $\tau$").format(phi_div_pi[plane], time[t]))

def press(event):
    global plane, t
    sys.stdout.flush()

    if event.key == 'down':
        if t >= 1:
            t = t - 1
            plot_em_fields()
        print("     Displaying timestep", t + 1)
    elif event.key == 'up':
        if t < n_t - 1:
            t = t + 1
            plot_em_fields()
        print("     Displaying timestep", t + 1)
    elif event.key == 'right':
        plane = (plane + 1) % n_planes
        plot_em_fields(clear_axes=True)
        print("     Displaying plane", plane + 1)
    elif event.key == 'left':
        plane = (plane - 1) % n_planes
        plot_em_fields(clear_axes=True)
        print("     Displaying plane", plane + 1)
    fig.canvas.draw()

# Plot fields
width = 20
height = 6
fig, axes = plt.subplots(1, len(em_x.data_vars), figsize=(width, height), \
                         sharex=True, sharey=True, \
                         subplot_kw={'aspect':'equal'})

cbar_axes = np.empty_like(axes, dtype=object)
for i in range(len(axes)):
    ax_divider = make_axes_locatable(axes[i])
    cbar_axes[i] = ax_divider.append_axes("right", size="7%", pad="2%")
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
fig.canvas.mpl_connect('key_press_event', press)

# Flip y-axis (=Z) if axis is flipped in equi. This namelist only exist in
# the params_in.txt file because it is a parallax namelist
params_in_file = args.path[0] + "params_in.txt"
flip_Z = nml_to_bool(get_param(params_in_file, "flip_Z"))
if flip_Z:
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()

plot_em_fields(clear_axes=True)
plt.show()
