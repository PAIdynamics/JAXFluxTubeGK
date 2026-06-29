import sys
import shutil
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from dependencies import check_dependencies

check_dependencies()

parser = argparse.ArgumentParser()

parser.add_argument('-a', '--onlyaggregate', default=False, action='store_true',
                    help="Only aggregate trace data and avoid plotting")

parser.add_argument('-k', '--keeporiginal', default=False, action='store_true',
                    help="Avoid the removal of the original trace data")

parser.add_argument('-n','--noshowplot', default=False, action='store_true',
                    help="Avoid displaying the figures")

parser.add_argument("-r", "--rank", type=int,
                    help="Select specific MPI process to plot")

parser.add_argument('path', type=str, nargs=1,
                    help='path to the output directory')

args = parser.parse_args()

if args.onlyaggregate:
    aggregate_traces = True
    plot_traces      = False
    show_plot        = False
else:
    aggregate_traces = True
    plot_traces      = True
    if args.noshowplot:
        show_plot = False
    else:
        show_plot = True

if args.keeporiginal:
    keep_original = True
else:
    keep_original = False

rank = 0
if args.rank:
    rank = args.rank

out_dir = args.path[0]

class device_memory_traces_t:
    """
    Handler class for device memory traces
    """

    def __init__(self, out_dir, keep_original_traces=False):
        if os.path.exists(out_dir + '/traces_dmem.csv'):
            print('    Info: Reading and processing ' + out_dir + '/traces_dmem.csv')
            self.df = pd.read_csv(out_dir + '/traces_dmem.csv')
        else:
            self.df = {"Rank": [],
                       "Region name": [],
                       "Start timestamp [s]": [],
                       "End timestamp [s]": [],
                       "Memory size [B]": [],
                       "Memory start [B]": [],
                       "Total memory usage [B]": []}
            self.df = pd.DataFrame(self.df)
            self._read_tracefiles(out_dir)
            self._save_complete_trace(out_dir)
            if not keep_original_traces:
                self._delete_original_traces(out_dir)

        self.num_ranks = self.df["Rank"].max() + 1

    def _store_entry(self, rank, name, tstart, tend, msize, mstart, musage):
        rowid = len(self.df)
        self.df.loc[rowid] = ""
        self.df.loc[rowid, "Rank"] = rank
        self.df.loc[rowid, "Region name"] = name
        self.df.loc[rowid, "Start timestamp [s]"] = tstart
        self.df.loc[rowid, "End timestamp [s]"] = tend
        self.df.loc[rowid, "Memory size [B]"] = msize
        self.df.loc[rowid, "Memory start [B]"] = mstart
        self.df.loc[rowid, "Total memory usage [B]"] = musage

    def _extract_data_by_region(self, df, region_name, rank):
        df_reg = df[df["Region name"] == region_name]

        while not df_reg.empty:
            mem_size = df_reg.iloc[0]["Memory size [B]"]

            # Get all entries with the same memory size (absolute value)
            df_mem = df_reg[df_reg["Memory size [B]"].abs() == mem_size]

            while not df_mem.empty:
                id_start = df_mem[df_mem["Memory size [B]"] > 0].index[0]
                id_end   = df_mem[df_mem["Memory size [B]"] < 0].index[0]
                if id_start == 0:
                    mem_start = 0
                else:
                    mem_start = df.loc[id_start - 1]["Total memory usage [B]"]
                self._store_entry(
                    rank   = rank,
                    name   = df_mem.loc[id_start]["Region name"],
                    tstart = df_mem.loc[id_start]["Timestamp [s]"],
                    tend   = df_mem.loc[id_end]["Timestamp [s]"],
                    msize  = mem_size,
                    mstart = mem_start,
                    musage = df_mem.loc[id_start]["Total memory usage [B]"])
                df_mem = df_mem.drop(id_start)
                df_mem = df_mem.drop(id_end)
                df_reg = df_reg.drop(id_start)
                df_reg = df_reg.drop(id_end)

    def _save_complete_trace(self, out_dir):
        self.df.to_csv(out_dir + '/traces_dmem.csv', index=False)

    def _delete_original_traces(self, out_dir):
        traces_dmem_dir = out_dir + '/traces_dmem/'
        print('    Info: Deleting ' + traces_dmem_dir)
        shutil.rmtree(traces_dmem_dir)

    def _read_tracefiles(self, out_dir):
        # Data frame aggregation rule
        aggregation_rule = {'Region name': 'last',
                            'Memory size [B]': 'sum',
                            'Total memory usage [B]': 'last',
                            'Timestamp [s]': 'last'}

        traces_dmem_dir = out_dir + '/traces_dmem/'

        # Get all trace files in the specified directory
        files = os.listdir(traces_dmem_dir)
        filenames  = [f for f in files if os.path.isfile(os.path.join(traces_dmem_dir, f))]
        tracefiles = [f for f in filenames if 'report' in f and '.csv' in f]
        ranks      = [f.split('.csv')[0] for f in tracefiles]
        ranks      = [int(r.split('_')[-1]) for r in ranks]
        del files
        del filenames

        for i in range(len(tracefiles)):
            print('    Info: Reading and processing ' + traces_dmem_dir + tracefiles[i])

            # Read a CSV file into a DataFrame
            df = pd.read_csv(traces_dmem_dir + tracefiles[i])
            df = df.groupby('Region ID').agg(aggregation_rule).reset_index()

            region_names  = df["Region name"].unique()

            for region in region_names:
                self._extract_data_by_region(df, region, ranks[i])

def plot_dmem_trace(dmem_traces, rank):
    # Fetch data
    print(f'    Info: Plotting the memory trace of GPU #{rank}')
    df = dmem_traces.df
    df = df[df["Rank"] == rank]

    nblocks       = df.shape[0]
    reg_names     = df["Region name"].unique()
    nregions      = len(reg_names)
    max_mem_usage = df["Total memory usage [B]"].max()
    max_time      = df["End timestamp [s]"].max()

    # Define color mapping for each memory block
    reg_colors = {}
    for i in range(nregions):
        reg_colors[reg_names[i]] = "C" + str(i)

    for i in range(nblocks):
        dfrow  = df.iloc[i]
        height = dfrow["Memory size [B]"] / 1e9
        width  = dfrow["End timestamp [s]"] - dfrow["Start timestamp [s]"]
        xpos   = dfrow["Start timestamp [s]"]
        ypos   = dfrow["Memory start [B]"] / 1e9
        color  = reg_colors[dfrow["Region name"]]
        ax.add_patch(mpl.patches.Rectangle((xpos, ypos), width, height, ls="-", ec=None, fc=color))

    ax.set_xlim(left   = -0.01*max_time, right = 1.01*max_time)
    ax.set_ylim(bottom = 0             , top   = 1.03*max_mem_usage/1e9)
    ax.set_xlabel('Runtime [s]')
    ax.set_ylabel(f'GPU #{rank} memory usage [GB]')

    handles, labels = ax.get_legend_handles_labels()
    for reg in reg_names:
        labels.append(reg)
        handles.append(mpl.patches.Rectangle([0, 0], 0, 0, color=reg_colors[reg], label=reg))
    handles, labels = list(reversed(handles)), list(reversed(labels))
    lgd = ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1.1, 0.45), fontsize=10, ncol=1)

    fig.tight_layout()

    fig.savefig(out_dir + f'/traces_dmem_{rank}.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig(out_dir + f'/traces_dmem_{rank}.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def press(event):
    global dmem_traces, rank, fig, ax
    sys.stdout.flush()
    ax.cla()

    if event.key == 'left':
        rank = (rank - 1) % dmem_traces.num_ranks
        plot_dmem_trace(dmem_traces, rank)
    elif event.key == 'right':
        rank = (rank + 1) % dmem_traces.num_ranks
        plot_dmem_trace(dmem_traces, rank)

    fig.canvas.draw()

# Read and process the tracefiles
dmem_traces = device_memory_traces_t(out_dir, keep_original_traces=keep_original)

if plot_traces:
    # Plot device memory trace
    figsize = (8, 3)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_dmem_trace(dmem_traces, rank)
    fig.canvas.mpl_connect('key_press_event', press)

    if show_plot:
        plt.show()
