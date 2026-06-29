import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
from misc import load_RZ, create_mesh_2d, create_field_2d, \
                 get_param, nml_to_bool
from plot_style import set_mpl_rcParams
from part import load_from_parts
from dependencies import check_dependencies
from matplotlib.animation import FuncAnimation, writers

check_dependencies()
set_mpl_rcParams()

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, nargs=1,
                    help='path to the output directory')
parser.add_argument("-s", "--species", type=str, default = "ions",
                    help="animate moment of given species")
parser.add_argument("-m", "--moment", type=str, default = "n",
                    help="animate given moment")
parser.add_argument("-p", "--plane", type=int,
                    help="animate given poloidal plane")
parser.add_argument("-f", "--fps", type=float,
                    default=1, help="frames per second for animation")
args = parser.parse_args()

grid_file = args.path[0] + "mesh.nc"
params_file = args.path[0] + "params_out.txt"

x, y, xind, yind = load_RZ(grid_file)

species = args.species
moment = args.moment
fps = args.fps

all_species = get_param(params_file, "names")
print("We received the following species: ", species)
if(species not in all_species):
    print("Error: Species", species, "not contained in the moments file!")
    sys.exit()
if(moment not in ["n", "u_par", "E_par",
                  "E_perp", "Q_par", "Q_perp"]):
    print("Error: Moment", moment, "not contained in the moments file!")
    sys.exit()

spec_data = load_from_parts(args.path[0],
                "mom_2d.nc", group=species)
field_data = getattr(spec_data, moment).data

n_planes = np.shape(field_data)[1]
if args.plane is None:
    plane = n_planes // 2
else:
    plane = (args.plane - 1) % n_planes

x_2d, y_2d = create_mesh_2d(x[plane, :], y[plane, :])

print("Welcome to the GENE-X animation tool!")
print("")
print("We are animating:")
print("    The", species, "species")
print("    The", moment, "moment")
print("    Poloidal plane", plane + 1, "out of", n_planes)

time_series = field_data[:, plane, :]
time = load_from_parts(args.path[0], "mom_0d.nc").dim_time.data

fig, ax = plt.subplots(figsize=(12.8, 7.2))

# Flip y-axis (=Z) if axis is flipped in equi. This namelist only exist in the
# params_in.txt file because it is a parallax namelist
params_in_file = args.path[0] + "params_in.txt"
flip_Z = nml_to_bool(get_param(params_in_file, "flip_Z"))
if flip_Z:
    ax.invert_yaxis()

record_2d = create_field_2d(xind[plane, :], yind[plane, :], time_series[0, :])

cax = ax.pcolormesh(x_2d, y_2d, record_2d,
                    vmin = np.min(time_series[:, :]),
                    vmax = np.max(time_series[:, :]),
                    cmap='plasma')
ax.set_aspect('equal')
fig.colorbar(cax)
suptitle = plt.suptitle("tau={:5.3e}".format(time[0]))

Writer = writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='The GENE-X Team'), bitrate=3600)

pause = False

def init():
    return animate(0)

def animate(i):
    record_2d = create_field_2d(xind[plane, :], \
                                yind[plane, :], \
                                time_series[i, :])
    cax.set_array(record_2d.ravel())
    suptitle.set_text("tau={:5.3e}".format(time[i]))
    #cax.autoscale()
    return cax, suptitle,

animation = FuncAnimation(
    fig,
    animate,
    frames = np.shape(time_series)[0],
    interval = 1,
    blit = False,
    repeat = False,
    init_func = init
)

def onClick(event):
    global pause
    if not pause:
        animation.event_source.stop()
        pause = True
    else:
        animation.event_source.start()
        pause = False

fig.canvas.mpl_connect('button_press_event', onClick)

plt.show()

animation.save("animate_{}_{}_plane{}.mp4".format( \
        moment, species, plane), writer=writer, dpi=100)
