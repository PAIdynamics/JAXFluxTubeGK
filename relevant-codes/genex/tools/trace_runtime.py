import argparse
import os
import sys
import json
import pandas as pd
import plotly.express as px
from dependencies import check_dependencies, check_dependency

check_dependencies()
check_dependency("plotly >= 4.9")

parser = argparse.ArgumentParser()

parser.add_argument('-s','--showplot', default=False, action='store_true',
                    help="Display the generated figures")

parser.add_argument('tracefile', type=str,
                    help='File containing the trace data')

args = parser.parse_args()

show_plot = False
if args.showplot:
    show_plot = True

tracefile = args.tracefile

def create_perfetto_json(df, filename):
    """
    Create JSON file from the trace file compatible with Perfetto UI
    """

    perfetto_df = df.copy()
    perfetto_df['ph'] = 'X'
    perfetto_df['cat'] = 'GENE-X'
    perfetto_df['dur'] = perfetto_df['Duration [s]'] * 1e6
    perfetto_df = perfetto_df.rename(columns={'Start timestamp [s]': 'ts',
                                              'Rank': 'pid',
                                              'Region name': 'name'})
    perfetto_df['ts'] = perfetto_df['ts'] * 1e6
    perfetto_df['tid'] = perfetto_df['pid']
    perfetto_df.drop(['Region path', 'Region level', 'End timestamp [s]', 'Duration [s]'],
                     axis=1, inplace=True)

    perfetto_json = perfetto_df.to_json(orient='records')
    perfetto_json = '{"schemaVersion":1,"traceName":"sample.json","traceEvents":' + perfetto_json + '}'
    perfetto_json = json.loads(perfetto_json)
    with open(filename, 'w') as file:
        json.dump(perfetto_json, file)

    return perfetto_json

def create_builtin_csv(perfetto_json, filename):
    """
    Create CSV trace file supported by the built-in visualizer from raw JSON file with Perfetto schema
    """

    df = pd.DataFrame.from_dict(perfetto_json['traceEvents'])
    df = df.rename(columns={'ts': 'Start timestamp [s]',
                            'pid' : 'Rank'})
    df['Start timestamp [s]'] = df['Start timestamp [s]'] / 1e6
    df['Duration [s]'] = df['dur'] / 1e6
    df['End timestamp [s]'] = df['Start timestamp [s]'] + df['Duration [s]']
    df['Region path'] = 'genex'
    df.drop(['ph', 'cat', 'tid', 'dur'], axis=1, inplace=True)
    df['Region level'] = df['name'].str.count('/')

    region_name = df['name'].str.split('/')
    region_name = region_name.str[-1]
    df['Region name'] = region_name

    region_path = df.apply(lambda row: row['name'].replace(row['Region name'], ''), axis=1)
    df['Region path'] = region_path.str.rstrip('/')
    df.drop('name', axis=1, inplace=True)

    df.to_csv(filename, index=False)

    return df

# Check if given trace file exists
if not os.path.exists(tracefile):
    sys.exit(f'    Error: Trace file {tracefile} cannot be found!')

# Read the runtime trace
print(f'    Info: Reading trace file {tracefile}')
if '.csv' in tracefile:
    # Read the trace from a supported CSV file
    df = pd.read_csv(tracefile)

elif '.json' in tracefile:
    # Read the trace from a supported JSON file with Perfetto schema
    with open(tracefile) as file:
        perfetto_json = json.load(file)

        # Convert Perfetto schema into supported DataFrame format
        print('    Info: Converting the raw Perfetto format into compatible DataFrame')
        df = create_builtin_csv(perfetto_json, 'traces_runtime.csv')
        del perfetto_json

else:
    sys.exit(f'    Error: {tracefile} is not supported!')

# Convert the runtime trace into Perfetto UI compatible JSON file
# NOTE: Compared to the original one, this is more readable.
if not os.path.exists('/traces_perfetto.json'):
    print('    Info: Converting the runtime trace into Perfetto UI compatible JSON file')
    create_perfetto_json(df, 'traces_perfetto.json')

# Convert timestamps [s] into datetime format
df.insert(loc=0, column='start', value=pd.to_datetime(df["Start timestamp [s]"], unit='s'))
df.insert(loc=0, column='end',   value=pd.to_datetime(df["End timestamp [s]"],   unit='s'))

region_levels = df['Region level'].unique()

for level in region_levels:
    # Plot the runtime trace for specific profiling region level
    print(f'    Info: Plotting the runtime trace of level {level} profiling regions')
    fig = px.timeline(df[df["Region level"] == level],
                      x_start="start", x_end="end", y="Rank",
                      hover_data=['Rank', 'Region name', 'Region path',
                                  'Region level', 'Duration [s]'],
                      color='Region name',
                      height=700, width=1400)

    # Save the plot as an HTML file
    fig.write_html(f"traces_runtime_lv{level}.html")
    del fig

# Plot the runtime trace of all profiling region levels
print('    Info: Plotting the runtime trace of all profiling levels')

# df_trace = df[df["Region level"] != 0].copy()
df['Region level x Rank compund axis'] = (max(df['Region level']) - df['Region level'] - 1) * (max(df['Rank']) + 1) + df['Rank']

fig = px.timeline(df, x_start="start", x_end="end", y="Region level x Rank compund axis",
                  hover_data=['Rank', 'Region name', 'Region path',
                              'Region level', 'Duration [s]'],
                  color='Region name',
                  height=700, width=1400)

# Save the plot as an HTML file
fig.write_html("traces_runtime.html")

# Display the interactive plot
if show_plot:
    fig.show()
