import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# Parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, nargs=1, help="path to output directory")
parser.add_argument("-n", "--n_checkpoints", type=int, default=3,
                    help=("number of checkpoints written by MMS test, "
                          "equal to the number of part folders"))
parser.add_argument("-f", "--final_time", type=float, default=0.625,
                    help="expected final time of MMS simulation")
parser.add_argument("-e", "--expected_scaling", type=int, default=2,
                    help="expected exponent of error scaling")
parser.add_argument("-nres", "--n_resolutions", type=int, default=4,
                    help="number of resolutions")
parser.add_argument("-d", "--deep", default=False, action='store_true',
                    help="Option to read the output from deep MMS testing")
parser.add_argument("-m", "--runtime_mode",  type=str, default="cpu",
                    help="runtime mode, i.e. cpu, gpu, cxx, acc, ompx")

args = parser.parse_args()
path = args.path[0]
n_checkpoints = args.n_checkpoints
final_time = args.final_time
expected_scaling = args.expected_scaling
n_res = args.n_resolutions
read_mms_deep_output = args.deep
mode = args.runtime_mode

# Function that defines exact scaling for reference
def exact_scaling(x, y0):
    return 2.0**expected_scaling * y0 / x**expected_scaling

scaling_word = {1:'First', 2:'Second', 3:'Third', 4:'Fourth', 5:'Fifth'}

# Functions to read the result from the file given by fname
def extract_mms_data(fname, res_id=0):
    # Returns the correct line with the mms results
    fid = open(fname)
    data = fid.readlines()
    res_counter = 0
    for i in range(len(data)):
        if "Result of the MMS verification:" in data[i]:
            if res_counter == res_id:
                # Skip the table header and the intermediate table entries
                result = data[i+1+n_checkpoints].split()
                result = np.array(result).astype(np.float64)
                if not np.isclose(result[0], final_time):
                    print(("\tWarning: Final time is not as expected!"),
                    ("Expected = {:.4f}, Actual = {:.4f}").format(final_time,
                                                                  result[0]))
                return result
            res_counter += 1

def get_scaling(fname, res_id=0):
    print("Retrieving result from " + fname)
    try:
        data_ref = extract_mms_data(fname, res_id)
        return data_ref[1:].reshape(2, 6)
    except:
        print("Could not read result from file, skipping!")
        return float("nan")

# Collect results from files assuming fixed directory structure
mms_err_slab      = np.zeros((n_res, 2, 6), dtype=np.float64)
mms_err_circular  = np.zeros((n_res, 2, 6), dtype=np.float64)
mms_err_salpha    = np.zeros((n_res, 2, 6), dtype=np.float64)
mms_err_dommaschk = np.zeros((n_res, 2, 6), dtype=np.float64)
if read_mms_deep_output:
    fname = "mms_deep_" + mode + ".out"
    for i in range(n_res):
        rel = "resolution_" + str(i+1) + "/job.out"
        mms_err_slab[i,:,:]      = get_scaling(path + "slab/" + fname, i)
        mms_err_circular[i,:,:]  = get_scaling(path + "circular/" + fname, i)
        mms_err_salpha[i,:,:]    = get_scaling(path + "salpha/" + fname, i)
        mms_err_dommaschk[i,:,:] = get_scaling(path + "dommaschk/" + fname, i)

else:
    for i in range(n_res):
        rel = "resolution_" + str(i+1) + "/" + mode + "/job.out"
        mms_err_slab[i,:,:]      = get_scaling(path + "slab/" + rel)
        mms_err_circular[i,:,:]  = get_scaling(path + "circular/" + rel)
        mms_err_salpha[i,:,:]    = get_scaling(path + "salpha/" + rel)
        mms_err_dommaschk[i,:,:] = get_scaling(path + "dommaschk/" + rel)

# Calculate error ratio
rat_err_slab      = mms_err_slab[:-1,:,:] / mms_err_slab[1:,:,:]
rat_err_circular  = mms_err_circular[:-1,:,:] / mms_err_circular[1:,:,:]
rat_err_salpha    = mms_err_salpha[:-1,:,:] / mms_err_salpha[1:,:,:]
rat_err_dommaschk = mms_err_dommaschk[:-1,:,:] / mms_err_dommaschk[1:,:,:]

titles = ["ions", "electrons", "es_pot", "A_par", "B_par", "E_par"]

print("")
print("Printing MMS analysis...")
print("------------------------")
fstr = "\t{:11}{:34} | {}"
print(fstr.format("", "L2 error ratio", "Linf error ratio"))
for i in range(len(titles)):
    print("Error ratio " + titles[i] + ":")
    print(fstr.format("slab:",
                      np.array2string(rat_err_slab[:, 0, i]),
                      np.array2string(rat_err_slab[:, 1, i])))
    print(fstr.format("circular:",
                      np.array2string(rat_err_circular[:, 0, i]),
                      np.array2string(rat_err_circular[:, 1, i])))
    print(fstr.format("salpha:",
                      np.array2string(rat_err_salpha[:, 0, i]),
                      np.array2string(rat_err_salpha[:, 1, i])))
    print(fstr.format("dommaschk:",
                      np.array2string(rat_err_dommaschk[:, 0, i]),
                      np.array2string(rat_err_dommaschk[:, 1, i])))

# Create plot
norm_type = ("2", "\infty")
# Line styles for norm types
line_styles = ("--", ":")
# Color cycle for geometries
colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
# Grey for reference lines in background
ref_color = "0.9"
nrow = 2
ncol = 3
fig, ax = plt.subplots(nrow, ncol, figsize=(16,9))

for i in range(nrow):
    for j in range(ncol):
        ind = i * ncol + j

        # We need to break because we have more subplots than data
        if (ind >= len(titles)):
            fig.delaxes(ax[i,j]) # delete empty subplot
            break

        ax[i,j].set_xlabel("Resolution")
        ax[i,j].set_ylabel("$\\frac{||f - f_{MMS}||_p}{||f_{MMS}||_p}$",
                           rotation=0, fontsize=14)
        ax[i,j].yaxis.set_label_coords(-0.1,0.9)
        ax[i,j].set_xscale("log")
        ax[i,j].set_yscale("log")

        ax[i,j].xaxis.set_major_formatter(NullFormatter())
        ax[i,j].xaxis.set_minor_formatter(NullFormatter())
        ax[i,j].xaxis.set_tick_params(which='minor', size=0)
        ax[i,j].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        x = 2**np.array(range(1, n_res+1))
        ax[i,j].set_xticks(x)
        xticks = np.array(np.log2(x), dtype=int)
        ax[i,j].set_xticklabels(xticks)
        ax[i,j].set_title(titles[ind])

        for k in range(2):
            ax[i,j].plot(x, mms_err_slab[:,k,ind],
                         marker="o", ls=line_styles[k], c=colors[0])
            ax[i,j].plot(x, mms_err_circular[:,k,ind],
                         marker="o", ls=line_styles[k], c=colors[1])
            ax[i,j].plot(x, mms_err_salpha[:,k,ind],
                         marker="o", ls=line_styles[k], c=colors[2])
            ax[i,j].plot(x, mms_err_dommaschk[:,k,ind],
                         marker="o", ls=line_styles[k], c=colors[3])

        # Create legend, including only geometries where a result was found
        legend_elements = [mpl.lines.Line2D([0], [0], c=ref_color,
                                            label="{} order reference".format(
                                               scaling_word[expected_scaling])),
                           mpl.lines.Line2D([0], [0], c="k", ls=line_styles[0],
                                            label='p={} error'.format(
                                               norm_type[0])),
                           mpl.lines.Line2D([0], [0], c="k", ls=line_styles[1],
                                            label=r'p=${}$ error'.format(
                                               norm_type[1]))]

        if (~np.isnan(mms_err_slab[:,:,ind])).any():
            legend_elements.append(mpl.lines.Line2D([0], [0], marker="o",
                                                    c=colors[0],
                                                    label='Slab'))
        if (~np.isnan(mms_err_circular[:,:,ind])).any():
            legend_elements.append(mpl.lines.Line2D([0], [0], marker="o",
                                                    c=colors[1],
                                                    label='Circular'))
        if (~np.isnan(mms_err_salpha[:,:,ind])).any():
            legend_elements.append(mpl.lines.Line2D([0], [0], marker="o",
                                                    c=colors[2],
                                                    label='Salpha'))
        if (~np.isnan(mms_err_dommaschk[:,:,ind])).any():
            legend_elements.append(mpl.lines.Line2D([0], [0], marker="o",
                                                    c=colors[3],
                                                    label='Dommaschk'))

        ax[i,j].legend(handles=legend_elements, loc="upper right")

        # Add reference lines in the background
        # Force auto-scale to update data limits
        ax[i,j].autoscale_view()
        ax[i,j].set_autoscale_on(False)
        ref_min = np.nanmin([mms_err_slab[:, :, ind], \
                             mms_err_circular[:, :, ind], \
                             mms_err_salpha[:, :, ind], \
                             mms_err_dommaschk[:, :, ind]])
        ref_max = 2**(expected_scaling * (n_res - 1)) \
                * np.nanmax([mms_err_slab[:, :, ind], \
                             mms_err_circular[:, :, ind], \
                             mms_err_salpha[:, :, ind], \
                             mms_err_dommaschk[:, :, ind]])
        n_ref = 10
        if ref_min <= 0:
            ref_min = 1e-4
            print("Warning: Minimum reference value is non-positive." \
            "Setting ref_min to: ", ref_min)
        if ref_max <= 0:
            ref_max = 2e2
            print("Warning: Maximum reference value is non-positive." \
            "Setting ref_max to: ", ref_max)
        ref_line_y0s = np.geomspace(ref_min, ref_max, n_ref, endpoint=True)
        for n in range(n_ref):
            reference = exact_scaling(x, ref_line_y0s[n])
            ax[i,j].plot(x, reference, ref_color, zorder=-5)

plt.tight_layout()
plt.show()
