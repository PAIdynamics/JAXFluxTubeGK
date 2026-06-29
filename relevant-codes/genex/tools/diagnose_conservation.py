import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
from misc import get_param
from part import load_from_parts
from dependencies import check_dependencies

check_dependencies()

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, nargs=1,
                    help='path to the output directory')
args = parser.parse_args()

params_file = args.path[0] + "params_out.txt"

time = load_from_parts(args.path[0], "mom_0d.nc").dim_time.data

species = get_param(params_file, "names")
print("Received species: ", species)

if(len(species) != 2):
    print("Error: At the moment the diagnostics only supports two species")
    sys.exit()


# read the particle number per species
particle_number = {}
for n in range(len(species)):
    moments = load_from_parts(args.path[0], "mom_0d.nc", group=species[n])

    particle_number[species[n]] = np.array(moments.n.data)

# calculate total energy per species
energy = {}
for n in range(len(species)):
    moments = load_from_parts(args.path[0], "mom_0d.nc", group=species[n])

    energy_par = moments.E_par.data
    energy_perp = moments.E_perp.data
    energy_es = moments.E_es.data
    energy[species[n]] = np.array(energy_par) + np.array(energy_perp) \
                                              + np.array(energy_es)

# read canonical toroidal momentum
momentum_can_tor = {}
for n in range(len(species)):
    moments = load_from_parts(args.path[0], "mom_0d.nc", group=species[n])

    momentum_can_tor[species[n]] = np.array(moments.p_phi.data)

# calculate momentum
param_mass = get_param(params_file, "masses")
masses = {}
for n in range(len(species)):
    masses[species[n]] = float(param_mass[n])

momentum = np.zeros_like(time)
for n in range(len(species)):
    moments = load_from_parts(args.path[0], \
                    "mom_0d.nc", group=species[n])
    momentum = momentum + masses[species[n]] * np.array(moments.u_par.data)

# normalize quantities to plot in the same plot
momentum = momentum / momentum[0]
for n in range(len(species)):
    energy[species[n]] = energy[species[n]] / energy[species[n]][0]
    particle_number[species[n]] = particle_number[species[n]] \
                                / particle_number[species[n]][0]
    momentum_can_tor[species[n]] = momentum_can_tor[species[n]] \
                                / momentum_can_tor[species[n]][0]

# calculate totals
# index the totals by name "total" to use the same interface as the multi
# species plot
particle_number_total = {"total": 0.0}
energy_total = {"total": 0.0}
momentum_can_tor_total = {"total": 0.0}
for n in range(len(species)):
    particle_number_total["total"] = particle_number_total["total"] + \
        particle_number[species[n]]
    energy_total["total"] = energy_total["total"] + energy[species[n]]
    momentum_can_tor_total["total"] = momentum_can_tor_total["total"] + \
        momentum_can_tor[species[n]]

# calculate rates
particle_number_rate = {}
energy_rate = {}
momentum_can_tor_rate = {}
for n in range(len(species)):
    energy_rate[species[n]] = \
        np.gradient(energy[species[n]], time)
    particle_number_rate[species[n]] = \
        np.gradient(particle_number[species[n]], time)
    momentum_can_tor_rate[species[n]] = \
        np.gradient(momentum_can_tor[species[n]], time)

# calculate rates of totals
particle_number_rate_total = {}
energy_rate_total = {}
momentum_can_tor_rate_total = {}
particle_number_rate_total["total"] = \
    np.gradient(particle_number_total["total"], time)
energy_rate_total["total"] = \
    np.gradient(energy_total["total"], time)
momentum_can_tor_rate_total["total"] = \
    np.gradient(momentum_can_tor_total["total"], time)

def toggle_total(is_total):
    # clear axes of all subplots
    shp = np.shape(subs)
    for i in range(shp[0]):
        for j in range(shp[1]):
            subs[i, j].cla()
    # either plot species values
    if(is_total == True):
        is_total = False
        plot_all(fig, subs, time, species, particle_number,
            particle_number_rate, energy, energy_rate,
            momentum_can_tor, momentum_can_tor_rate)
    # or totals
    else:
        is_total = True
        plot_all(fig, subs, time, ["total"], particle_number_total,
            particle_number_rate_total, energy_total, energy_rate_total,
            momentum_can_tor_total, momentum_can_tor_rate_total)
    return is_total

def press(event):
    global fig, is_total
    sys.stdout.flush()

    if event.key == 't':
        is_total = toggle_total(is_total)
        print("     Switching to total mode = ", is_total)

    fig.canvas.draw()

def plot_single(subs, i, j, time, quant, tit, lab, leg, linestyle):
    subs[i,j].plot(time, quant, linestyle, label=leg)
    subs[i,j].legend()
    subs[i,j].set(title=tit, ylabel=lab)
    if i == 1:
        subs[i,j].set(xlabel="t / $\mathregular{\\tau}$")

def plot_all(fig, subs, time, species, n, dndt, E, dEdt, psi0, dpsi0dt):

    # line styles to cycle through in plot loop
    lines = ["-","--","-.",":"]

    for sp in range(len(species)):
        #1st row: n, E, psi0
        plot_single(subs, 0, 0, time, n[species[sp]],
            tit="density",
            lab="n / 1",
            leg="n (" + species[sp] + ")",
            linestyle="b" + lines[sp])

        plot_single(subs, 0, 1, time, E[species[sp]],
            tit="energy",
            lab="E / 1",
            leg="E (" + species[sp] + ")",
            linestyle="r" + lines[sp])

        plot_single(subs, 0, 2, time, psi0[species[sp]],
            tit="canonical toroidal momentum",
            lab="$\mathregular{\psi_0}$ / 1",
            leg="$\mathregular{\psi_0}$ (" + species[sp] + ")",
            linestyle="g" + lines[sp])

        #2nd row: dn/dt, dE/dt, dpsi0/dt
        plot_single(subs, 1, 0, time, dndt[species[sp]],
            tit="density change",
            lab="dn/dt / $\\tau^{-1}$",
            leg="dn/dt (" + species[sp] + ")",
            linestyle="b" + lines[sp])

        plot_single(subs, 1, 1, time, dEdt[species[sp]],
            tit="energy change",
            lab="dE/dt / $\\tau^{-1}$",
            leg="dE/dt (" + species[sp] + ")",
            linestyle="r" + lines[sp])

        plot_single(subs, 1, 2, time, dpsi0dt[species[sp]],
            tit="canonical toroidal momentum change",
            lab="d$\mathregular{\psi_0}$/dt / $\\tau^{-1}$",
            leg="d$\mathregular{\psi_0}$/dt (" + species[sp] + ")",
            linestyle="g" + lines[sp])

is_total = False

#create plot and listen to key press
fig, subs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
plot_all(fig, subs, time, species, particle_number, particle_number_rate,
            energy, energy_rate, momentum_can_tor, momentum_can_tor_rate)

fig.canvas.mpl_connect('key_press_event', press)
plt.show()
