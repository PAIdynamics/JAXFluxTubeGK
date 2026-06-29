import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import argparse
from misc import load_RZ, load_phi, load_velspace, create_mesh_2d, \
                 create_field_2d, get_n_points, get_nearest_uind, get_param, \
                 nml_to_float, nml_to_bool
from plot_style import set_mpl_rcParams, get_boxlims
from part import load_checkpoints_from_parts
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

grid_file = args.path[0] + "mesh.nc"
params_file = args.path[0] + "params_out.txt"

# get species names. we need to use the order stored in the parameter file
# to access the correct species indices in dist_func
species = get_param(params_file, "names")

# get if use vspectral enabled
use_vspec = nml_to_bool(get_param(params_file, "use_vspectral"))

if use_vspec:
    print("Diagnose spectral velocity-space")
else:
    print("Diagnose grid velocity-space")

dataset = load_checkpoints_from_parts(args.path[0], "checkpoint*.nc")
dist_func = dataset.dist_func.data
print("Received shape: ", np.shape(dist_func))

(n_RZ, n_planes, n_vp, n_mu, n_sp) = get_n_points(grid_file, params_file)

time = dataset.dim_time.data
n_t = len(time)

if args.record is None:
    t = n_t - 1
else:
    t = (args.record - 1) % n_t

# determine the number of planes
if args.plane is None:
    plane = n_planes // 2
else:
    plane = (args.plane - 1) % n_planes
print("    Displaying plane {} of {}, \
           timestep {} of {}".format(plane + 1, n_planes, t + 1, n_t))
plane_prev = plane

# get grid
spacing_RZ = nml_to_float(get_param(params_file, "spacing_RZ"))

x, y, xind, yind = load_RZ(grid_file)
phi_div_pi = load_phi(grid_file)
vp, mu = load_velspace(grid_file)

(vp_2d, mu_2d) = np.meshgrid(vp, mu)

if not use_vspec:
    # offset vp to have the correct labels with pcolormesh
    vp_2d = vp_2d - 0.5 * (vp[1] - vp[0])

width = 13
height = 11
fig, axes = plt.subplots(3, 2, figsize=(width, height),
    sharex=False, sharey=False)

# Flip y-axis (=Z) if axis is flipped in equi. This namelist only exist in the
# params_in.txt file because it is a parallax namelist
params_in_file = args.path[0] + "params_in.txt"
flip_Z = nml_to_bool(get_param(params_in_file, "flip_Z"))

quad_meshes = {}
quad_meshes['real'] = {}
quad_meshes['vel'] = {}
plots  = [0,0]
lines  = [0,0]

# points
points = {}
points['real'] = [0,0]
points['vel'] = [0,0]

# colorbar objects must be array not dict. init with 2 elements.
colbars_pos = [0,0]
colbars = {}
colbars['real'] = [0,0]
colbars['vel'] = [0,0]

mu_ind = 0
vp_ind = n_vp // 2
RZ_ind = 0

def prep_plane_change():
    global points, RZ_ind

    for ax in axes[0, :]:
        ax.cla()
        ax.set(aspect='equal', **get_boxlims(x, y))
        if flip_Z:
            # Only flip the upper 2 subplots which contain real space
            ax.invert_yaxis()

    # If possible, keep real-space point at same location
    if points['real'][0] != 0:
        x_prev = x[plane_prev, RZ_ind]
        y_prev = y[plane_prev, RZ_ind]
        RZ_ind = get_nearest_uind(x[plane, :].compressed(),
                                              y[plane, :].compressed(),
                                              x_prev, y_prev,
                                              spacing_RZ)
    points['real'] = [0,0]

def plot_record(t, plane):
    """
        Plots distribution function velocity space and real-space slices at a
        time point t on a particular plane.
        This function is used on first execution of the program and if the
        timestep or plane is updated.
    """
    x_2d, y_2d = create_mesh_2d(x[plane, :], \
                                y[plane, :])

    fig.suptitle((r"Data for $\varphi = ${:.3f}$\pi$, " \
                  r"t = {:.5f} $\tau$").format(phi_div_pi[plane], time[t]))

    for sp in range(2):

        # 1st row:

        ax = axes[0, sp]
        # plot the dist funct at (vp, mu) = (0, 0)
        record = dist_func[t, sp, :, :, plane, :]
        record_2d = create_field_2d(xind[plane, :], \
                                    yind[plane, :], \
                                    record[mu_ind, vp_ind, :])
        cmap = 'plasma'

        if use_vspec:
            ax.set(title="$N^{" + str(vp_ind) \
                   + str(mu_ind) + "}$ - " + species[sp],
                   xlabel = "R / m",
                   ylabel = "Z / m")
        else:
            ax.set(title="Distribution function (" + species[sp] + ")",
                   xlabel = "R", ylabel = "Z")

        c = ax.pcolormesh(x_2d, y_2d, record_2d, cmap = cmap)
        quad_meshes['real'][species[sp]] = c

        if colbars['real'][sp] == 0: #create colorbar axis only on init
            colbars['real'][sp] = fig.colorbar(c, ax=ax)
        else:
            colbars['real'][sp].ax.cla()
        colbars['real'][sp] = fig.colorbar(c, cax=colbars['real'][sp].ax)

        if use_vspec:
            colbars['real'][sp].set_label("$N^{" + str(vp_ind)
                                       + str(mu_ind) + "}$(R, Z) / 1")
        else:
            colbars['real'][sp].set_label(r"f(R, Z, $v_{||}$ = " +
                 "{:.3f}, $\mu$ = {:.3f}) / 1".format(vp[vp_ind], \
                                                      mu[mu_ind]))

        #create point only on init
        if points['real'][sp] == 0:
                    points['real'][sp] = ax.plot(x[plane, RZ_ind], \
                                                 y[plane, RZ_ind], \
                                                'ro', markersize = 3, \
                                                mfc = 'r')

        # 2nd row:
        ax = axes[1, sp]

        record_2d = record[:, :, RZ_ind]
        norm = None
        cmap = "viridis"

        if use_vspec:
            record_2d = np.abs(record_2d)
            norm = LogNorm(vmin=record_2d.min(), \
                           vmax=record_2d.max())
            cmap = "inferno"

        quad_meshes['vel'][species[sp]] = \
                  ax.pcolormesh(vp_2d, mu_2d, record_2d, cmap=cmap,
                                norm=norm)

        if colbars['vel'][sp] == 0: #create colorbar axis only on init
            colbars['vel'][sp] = fig.colorbar(\
                                         quad_meshes['vel'][species[sp]], \
                                         ax=ax)
        else:
            colbars['vel'][sp].ax.cla()
            colbars['vel'][sp] = fig.colorbar(\
                                         quad_meshes['vel'][species[sp]], \
                                         cax=colbars['vel'][sp].ax)

        if use_vspec:
            # plot amplitude spectrum evaluated at (x,y)
            ax.set(title="Spectrum - " + species[sp],
                   xlabel = "$\mathregular{p}$ / 1",
                   ylabel = "$\mathregular{j}$ / 1")

            colbars['vel'][sp].set_label("$|N^{pj}$( R = " + \
                   "{:.3f}, Z = {:.3f})| / 1".format(x[plane, RZ_ind], \
                                                     y[plane, RZ_ind]))

            if points['vel'][sp] == 0:
                points['vel'][sp] = ax.plot(vp[vp_ind], \
                                            mu[mu_ind], \
                                           'rx', \
                                            markersize = 10, \
                                            mfc = 'r')

        else:
            ax.set(title="Velocity space (" + species[sp] + ")",
            xlabel = "$\mathregular{v_{||}}$ / 1",
            ylabel = "$\mathregular{\mu}$ / 1")

            colbars['vel'][sp].set_label(\
                "f($\mathregular{v_{||}, \mu, R}$ = " + \
                "{:.3f}, Z = {:.3f}) / 1".format(x[plane, RZ_ind], \
                y[plane, RZ_ind]))

        if lines[sp] == 0: #create line only on init
            lines[sp] = ax.plot(ax.get_xlim(), [mu[mu_ind], \
                                                mu[mu_ind]],
                                "-g", linewidth = 2)
        else:
            lines[sp][0].set_ydata([mu[mu_ind], mu[mu_ind]])

        # 3rd row:
        ax = axes[2,sp]

        # plot mu slice
        vp_slice = record_2d[mu_ind, :]
        if use_vspec:
            vp_slice = np.abs(vp_slice / vp_slice[0])

        if plots[sp] == 0: # Create plot only on init
            if use_vspec:
                plots[sp] = ax.semilogy(vp, vp_slice, '-og')
                ax.set(xlabel = "$\mathregular{p}$ / 1",
                       ylabel = "$| N^{p" + str(mu_ind) + "} / N^{0" \
                              +  str(mu_ind)+  "} |$")
            else:
                plots[sp] = ax.plot(vp, vp_slice, '-g')
                ax.set(title="Velocity space, $\mathregular{\mu}$-slice (" + \
                       species[sp] + ")",
                       xlabel = "$\mathregular{v_{||}}$ / 1",
                       ylabel="f($\mathregular{v_{||}, \mu}$ = " + \
                       "{:.3f}) / 1".format(mu[mu_ind]))

        else:
            plots[sp][0].set_ydata(vp_slice)
            ax.relim()
            ax.autoscale_view()

def refresh_vel_space(RZ_ind_new, mu_ind_new):
    """
        Refresh the velocity-space plots if a new unstructured RZ index or mu
        index is selected. If one of these indices is unchanged, pass None.
    """

    update_RZ = (RZ_ind_new is not None)
    # Always update the mu line plot if the point in RZ has changed
    update_mu = (mu_ind_new is not None) or update_RZ

    x_new = x[plane, RZ_ind]
    y_new = y[plane, RZ_ind]

    for sp in range(2):
        record = dist_func[t, sp, :, :, plane, RZ_ind]
        if use_vspec:
            record = np.abs(record)

        if update_RZ:
            # Update the point indicating the RZ location of the velocity-space
            # plot
            points['real'][sp][0].set_xdata([x_new])
            points['real'][sp][0].set_ydata([y_new])
            if use_vspec:
                colbars['vel'][sp].set_label("$| N^{pj}$(R = " + \
                    "{:.3f}, Z = {:.3f})| / 1".format(x_new, y_new))
            else:
                colbars['vel'][sp].set_label( \
                    "f($\mathregular{v_{||}, \mu, R}$ = " + \
                    "{:.3f}, Z = {:.3f}) / 1".format(x_new, y_new))

            # Update the 2D velocity-space plot
            quad_meshes['vel'][species[sp]].set_array(np.asarray(record.ravel()))
            quad_meshes['vel'][species[sp]].autoscale()

        if update_mu:
            # Update the 1D vp slice
            vp_slice = record[mu_ind, :]
            if use_vspec:
                vp_slice /= vp_slice[0]

            plots[sp][0].set_ydata(vp_slice)
            lines[sp][0].set_ydata([mu[mu_ind], mu[mu_ind]])

            ax = axes[2,sp]
            if use_vspec:
                ax.set_ylabel("$| N^{p" + str(mu_ind) + "} / N^{0" \
                              +  str(mu_ind)+  "} |$")
            else:
                ax.set(ylabel="f($\mathregular{v_{||}, \mu}$ = " + \
                                            "{:.3f}) / 1".format(mu[mu_ind]))
            ax.relim()
            ax.autoscale_view()

def refresh_real_space(vp_ind, mu_ind):
    """
        Refresh the real-space plot if a new vp and/or mu index is selected.
    """

    if use_vspec:
        for sp in range(2):
            points['vel'][sp][0].set_xdata([vp[vp_ind]])
            points['vel'][sp][0].set_ydata([mu[mu_ind]])

            ax = axes[0, sp]
            colbars['real'][sp].set_label("$N^{" + str(vp_ind)
                                                 + str(mu_ind) + "}$(R, Z) / 1")
            ax.set(title="$N^{" + str(vp_ind) \
                                + str(mu_ind) + "}$ - " + species[sp],
                   xlabel = "R / m",
                   ylabel = "Z / m")

            # update real space plot with new velocity space indices
            record = dist_func[t, sp, mu_ind, vp_ind, plane, :]

            record_2d = create_field_2d(xind[plane, :], \
                                        yind[plane, :], \
                                        record)

            pcm = quad_meshes['real'][species[sp]]
            pcm.set_array(record_2d.ravel())
            max_clim = np.max(record_2d)
            min_clim = np.min(record_2d)
            pcm.set_clim(min_clim, max_clim)
            fig.canvas.draw()
    else:
        # TODO: Implement this functionality for the grid-version as well
        pass

def onclick(event):
    global RZ_ind

    # do only if click was in upper row (real space)
    if event.inaxes == axes[0, 0] or event.inaxes == axes[0, 1]:
        RZ_ind = get_nearest_uind(x[plane, :].compressed(),
                                  y[plane, :].compressed(),
                                  event.xdata, event.ydata,
                                  spacing_RZ)

        refresh_vel_space(RZ_ind, None)

        x_click = x[plane, RZ_ind]
        y_click = y[plane, RZ_ind]
        print("     Switching to real space point \
                    (R, Z) = ({:.6f}, {:.6f})".format(\
                        x_click, y_click))
        fig.canvas.draw()

def press(event):
    global vp_ind, mu_ind, t, plane, plane_prev
    sys.stdout.flush()

    # Increase mu index
    if event.key == 'w':
        if not use_vspec:
            print("Warning: Mu update not fully implemented for grid!")
        if(mu_ind < n_mu - 1):
            mu_ind = mu_ind + 1

            refresh_real_space(vp_ind, mu_ind)
            refresh_vel_space(None, mu_ind)
        print("     Switching to mu = {:.6f}".format(mu[mu_ind]))

    # Decrease mu index
    elif event.key == 'x':
        if not use_vspec:
            print("Warning: Mu update not fully implemented for grid!")
        if(mu_ind >= 1):
            mu_ind = mu_ind - 1

            refresh_real_space(vp_ind, mu_ind)
            refresh_vel_space(None, mu_ind)
        print("     Switching to mu = {:.6f}".format(mu[mu_ind]))

    # Increase vp index
    elif event.key == 'd':
        if not use_vspec:
            print("Warning: Vp update not implemented for grid!")
        if(vp_ind < n_vp - 1):
            vp_ind = vp_ind + 1

            refresh_real_space(vp_ind, mu_ind)
        print("     Switching to vp = {:.6f}".format(vp[vp_ind]))

    # Decrease vp index
    elif event.key == 'a':
        if not use_vspec:
            print("Warning: Vp update not implemented for grid!")
        if(vp_ind >= 1):
            vp_ind = vp_ind - 1

            refresh_real_space(vp_ind, mu_ind)
        print("     Switching to vp = {:.6f}".format(vp[vp_ind]))

    # Increase plane
    elif event.key == 'right':
        if use_vspec:
            print("Warning: Plane update not tested for vspec!")
        plane_prev = plane
        plane = (plane + 1) % n_planes

        prep_plane_change()
        plot_record(t, plane)
        print("     Switching to plane", plane + 1)

    # Decrease plane
    elif event.key == 'left':
        if use_vspec:
            print("Warning: Plane update not tested for vspec!")
        plane_prev = plane
        plane = (plane - 1) % n_planes

        prep_plane_change()
        plot_record(t, plane)
        print("     Switching to plane", plane + 1)

    # Increase time
    elif event.key == 'up':
        if use_vspec:
            print("Warning: Time update not tested for vspec!")
        if(t < n_t - 1):
            t = t + 1

            plot_record(t, plane)
        print("     Displaying timestep =", t + 1)

    # Decrease time
    elif event.key == 'down':
        if use_vspec:
            print("Warning: Time update not tested for vspec!")
        if(t >= 1):
            t = t - 1

            plot_record(t, plane)
        print("     Displaying timestep =", t + 1)

    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', press)

prep_plane_change()
plot_record(t, plane)
plt.show()
