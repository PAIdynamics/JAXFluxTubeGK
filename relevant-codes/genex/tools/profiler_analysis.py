import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from plot_style import set_mpl_rcParams
from profiler_tools import load_profiler_dataframe

"""
This program loads profiler data from the job file specified via the command line
and plots pie charts of the main profiler regions ('genex', 'initialize', and
'timestep'). It also contains helper functions to do further analysis on the
profiler data, such as summing all regions which contain a particular key.
"""

set_mpl_rcParams()

def pie_plot_level(df, column, key='genex', level=0, ax=None):
    """
    Takes the profiler data from df[column] and plots a pie chart from the
    region given by key and level. This chart shows all direct children of the
    given region.
    """
    # Select all direct children (i.e. the + 2 level is still a blank string)
    if level + 2 < len(df.index.names):
        keys = (key, '')
        levels = (level, level + 2)
    else:
        keys = (key)
        levels = (level)
    selection = df.xs(key=keys, level=levels)[column]

    # The first row will correspond to the top-level entry itself. Normalize all
    # the remaining data to that row and remove it
    title = "{} ({} = {:.2f})".format(key, column, selection.iloc[0])
    selection = selection[1:] / selection.iloc[0]

    # If the normalized column values are greater than one, re-normalize them
    # from the sum. This is done because for timestep, the sum of the profiler
    # regions tends to be < 1% larger than the top-level time, probably due to
    # rounding errors.
    ssum = selection.sum()
    if ssum > 1.0:
        print("Warning: Normalized sum of values in column", column, "for key",
               key, "is {:.05f}! Renormalizing to sum of 1".format(ssum))
        selection = selection / ssum

    # NOTE: We let remaining inaccuracies that lead to a sum slightly larger
    #       than one be handled by the pie plotting routine by using normalize
    labels = selection.index.get_level_values('lvl_' + str(level + 1))
    selection.plot(kind='pie', autopct="%1.2f%%",
                   ax=ax, explode=[0.1]*len(selection),
                   normalize=True, labels=labels, title=title, ylabel='')

def aggregate_by_key(df, key):
    """
    Finds all rows which include the given key at any level, and aggregates the
    column values with the appropriate logic (sum for number of calls, total
    time, and time percentage, and weighted average for time / call.)

    Useful to find the contribution of functions which can occur in various
    profiler regions, like 'mpi_allreduce'.
    """
    n_lvls = len(df.index.names)
    selection = pd.DataFrame(columns=df.columns)
    for lvl in range(n_lvls):
        try:
            # For levels before the final one: select only the top-level region
            # which includes the given key
            if lvl < n_lvls - 1:
                keys = (key, '')
                levels = (lvl, lvl + 1)
            else:
                keys = key
                levels = lvl
            lvl_selection = df.xs(keys, level=levels, drop_level=False)
            selection = pd.concat([selection, lvl_selection])
        # Key does not exist in current level
        except KeyError:
            continue

    # The pd.concat doesn't maintain the MultiIndex of the original dataframe
    multi_index = pd.MultiIndex.from_tuples(selection.index, names=df.index.names)
    selection.set_index(multi_index, inplace=True)

    aggregate = np.zeros(len(df.columns))
    for i, col in enumerate(df.columns):
        # For time per call column, aggregate via weighted average
        if col == col_time_per_call:
            if len(selection) > 0:
                aggregate[i] = np.average(selection[col],
                                          weights=selection[col_num_calls])
            else:
                aggregate[i] = np.nan
        # Otherwise aggregate via sum
        else:
            aggregate[i] = np.sum(selection[col])

    index = tuple(['aggregate'] + [''] * (n_lvls - 1))
    selection.loc[index, :] = aggregate

    return selection

parser = argparse.ArgumentParser()
parser.add_argument('fname', type=str, nargs=1,
                    help='file with profiler output')
parser.add_argument('-c', '--column', type=str,
                    help="profiler column to plot (default 'time / s')")
parser.add_argument('-a', '--aggregate', type=str, nargs='*',
                    help='give region keys to aggregate and print the result')
args = parser.parse_args()

if args.column is None:
    column = 'time / s'
else:
    column = args.column

profiler_data = load_profiler_dataframe(args.fname[0])

for col in profiler_data.columns:
    # Identify time per call and #calls column for use in aggregate function
    if col == 'time/call' or col == 'time per call':
        col_time_per_call = col
    elif col == '#calls':
        col_num_calls = col

if args.aggregate is not None:
    for key in args.aggregate:
        result = aggregate_by_key(profiler_data, key)
        print("Resulting aggregate for key:", key)
        print(result)
        print("")

mosaic = [['genex',    'initialize'],
          ['timestep', 'timestep'  ],
          ['timestep', 'timestep'  ]]
fig, axes = plt.subplot_mosaic(mosaic, figsize=(8,12))

for key, level in zip(['genex', 'initialize', 'timestep'],
                      [0,       1,            1]):
    pie_plot_level(profiler_data, column, key, level, ax=axes[key])

fig.suptitle("Profiler analysis")
plt.show()
