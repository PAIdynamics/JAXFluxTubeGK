import numpy as np
import argparse
import sys

spacing_between_runs = 2

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=1,
                    help="index of multi-run in the input file >= 1")
parser.add_argument("--skip", type=int, default=17,
                    help="number of skipped rows")
parser.add_argument("fname_ref", nargs=1, type=str, help="reference filename")
parser.add_argument("fname_test", nargs=1, type=str, help="test filename")
parser.add_argument("atol", type=float, help="absolute tolerance")
parser.add_argument("rtol", type=float, help="relative tolerance")
args = parser.parse_args()

class bcolors:
    Reset      = '\033[m'
    BoldRed    = '\033[1;31m'
    BoldGreen  = '\033[1;32m'
    Yellow     = '\033[33m'
    BoldYellow = '\033[1;33m'

def filter_whitelist(fname, verbose=False):
    # Function to read a file and filter the lines based on the whitelist.
    # NOTE: The whitelist is meant to be a temporary fix for deploying CI to a
    #       new machine. The whitelist should be an active issue well known by
    #       the support group of the machine and has no mitigation solution.
    #       The whitelist needs to be removed once such solution is provided.

    # Izum Vega-specifc whitelist
    whitelist = \
        ["slurmstepd: error: _get_joules_task: can't get info from slurmd",
         "Before the prereq statement in cuda default",
         "After the prereq statement in cuda default",
         "Removing old paths from PATH, LD_LIBRARY_PATH, CPATH"]

    # Read the file
    with open(fname, 'r') as file:
        lines = file.read().splitlines()

    # Filtering
    filtered_out = set()
    filtered_in = []
    for line in lines:
        if line in whitelist:
            filtered_out.add(line)
        else:
            filtered_in.append(line)

    # Print filtered out lines
    if(filtered_out and verbose):
        print(bcolors.BoldYellow + "Warning: the following lines match " +
              "the whitelist and are filtered out from the test:" +
              bcolors.Reset)
        for line in filtered_out:
            print(bcolors.Yellow + "> " + line + bcolors.Reset)

    return filtered_in

def print_failed(data_test, data_ref, res):
    for i, r in enumerate(res):
        if(not r):
            diff_abs = data_ref[i] - data_test[i]
            diff_rel = 100.0 * diff_abs / data_ref[i]
            print(f"Element {i:2d} was {data_test[i]:.4f}, " \
                 +f"should be {data_ref[i]:.4f}, " \
                 +f"difference is {diff_abs:>7.4f} " \
                 +f"({diff_rel:>8.2f} %)")

def print_err_reading(fname, skip):
    fid = open(fname)
    content = fid.readlines()
    print(bcolors.BoldRed + f"Failed to parse line {skip} of file {fname}:\n" +
          bcolors.Reset, content[skip])
    fid.close()

# In case of an output file with multiple runs, this returns the number of lines
# which indicate the end of each of these run as dictionary with the run index
# as the key
def get_end_of_runs(fname):
    file_test = filter_whitelist(fname, verbose=True)
    run_end_lines = {}
    id = 1
    i = 0
    for line in file_test:
        if "Simulation finished" in line:
                run_end_lines[id] = i
                id += 1
        i += 1
    return run_end_lines

skip = args.skip
fname_ref  = args.fname_ref[0].strip()
fname_test = args.fname_test[0].strip()
perform_test = True

run_end_lines = get_end_of_runs(fname_test)
n_runs        = len(run_end_lines)

if args.id > n_runs:
    print(bcolors.BoldRed + f"Error: the given --id={args.id} is more than " + \
          f"the number of runs {n_runs} detected in file {fname_test}!" + \
          bcolors.Reset)
    sys.exit(1)
else:
    # Skiprows for test file
    if args.id == 1:
        skip_test = skip
    else:
        skip_test = run_end_lines[args.id - 1] + spacing_between_runs + skip

# Try to read in the data

try:
    data_ref = np.loadtxt(fname_ref,
                          dtype=np.float64, skiprows=skip, max_rows=1)
except:
    print_err_reading(fname_ref, skip)
    perform_test = False

try:
    file_test = filter_whitelist(fname_test)
    data_test = np.loadtxt(file_test,
                           dtype=np.float64, skiprows=skip_test, max_rows=1)
except:
    print_err_reading(fname_test, skip_test)
    perform_test = False

if(not perform_test):
    print(bcolors.BoldRed + "Check failed: file could not be read!" +
          bcolors.Reset)
    sys.exit(1)

# Perform the check

print("test line:\n", data_test)
print("against:\n", data_ref)
res = np.isclose(data_test, data_ref, rtol=args.rtol, atol=args.atol)

if(np.all(res)):
    print(bcolors.BoldGreen + "Check passed!" + bcolors.Reset)
    sys.exit(0)
else:
    print(bcolors.BoldRed + "Check failed: no match!" + bcolors.Reset)
    print_failed(data_test, data_ref, res)
    sys.exit(1)
