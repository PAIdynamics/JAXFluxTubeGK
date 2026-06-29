import pandas as pd
import numpy as np
import io

sep = '|'

def read_runinfo_from_debug_list(outdir: str) -> dict:
    """
    Reads the debug_list.txt file from the given output directory and returns
    a dictionary with the run information.
    """
    runinfo = {}
    fname = outdir + '/debug_log.txt'
    try:
        with open(fname) as f:
            skiplines=0
            gdim={'RZ':0,'phi':0,'vpar':0,'mu':0,'spec':0}
            ldim_inner={'RZ':0,'phi':0,'vpar':0,'mu':0,'spec':0}
            for line in f:
                if skiplines==0:
                    if line.strip().startswith("Global problem size"):
                        skiplines=1
                    elif line.strip().startswith("|"):
                        parts = [part.strip() for part in line.split('|')]
                        gdim['RZ']=int(parts[1])
                        gdim['phi']=int(parts[2])
                        gdim['vpar']=int(parts[3])
                        gdim['mu']=int(parts[4])
                        gdim['spec']=int(parts[5])
                    elif line.strip().startswith("non-ghost"):
                        parts = [part.strip() for part in line.split('|')]
                        ldim_inner['RZ']  =int(parts[1])
                        ldim_inner['phi'] =int(parts[2])
                        ldim_inner['vpar']=int(parts[3])
                        ldim_inner['mu']  =int(parts[4])
                        ldim_inner['spec']=int(parts[5])
                else:
                    skiplines-=1
            runinfo['gdim']=gdim
            runinfo['ldim_inner']=ldim_inner
            runinfo['parallelization']={'RZ':0,'phi':0,'vpar':0,'mu':0,'spec':0}
            for part in ['RZ','phi','vpar','mu','spec']:
                runinfo['gdim'][part]=int(gdim[part])
                runinfo['ldim_inner'][part]=int(ldim_inner[part])
                if ldim_inner[part] != 0:
                    runinfo['parallelization'][part]=int(gdim[part] / ldim_inner[part])
                else:
                    runinfo['parallelization'][part]=1
    except FileNotFoundError:
        print(f"Warning: Could not find runinfo file '{fname}'")
    return runinfo

def extract_performance_block(fname : str) -> tuple[int,io.StringIO,int,list,int,dict]:
    skiprows = None
    nrows = None
    names = None
    max_lvl = 0
    prefix = '0: '

    with open(fname) as f:
        lines=io.StringIO()
        for line in f:
            lines.write(line.removeprefix(prefix))
        lines.seek(0,io.SEEK_SET)
        runinfo=None
        for i, line in enumerate(lines):
            # We look for the output_directory line to get the additional data
            # for the run from the files therein
            if line.strip().startswith('output is written to '):
                outdir = line.strip().removeprefix('output is written to ')
                runinfo = read_runinfo_from_debug_list(outdir)

            # The beginning of the profiler table starts with the 'region' header
            if line.strip().startswith('region'):
                # Collect column headers
                names = [name.strip() for name in line.replace("\n", "").split(sep)]
                # The actual data starts two lines later
                skiprows = i + 2

            # A profiler data table has been found, determine how long it is and
            # how many region levels it has
            if skiprows is not None:
                # The beginning and end of the table are indicated with dashed lines
                if line.strip().startswith('----'):
                    # Beginning of data table
                    if i == skiprows - 1:
                        continue
                    # End of data table - exit line loop
                    else:
                        nrows = i - skiprows
                        break
                # From all data table lines, determine the maximum level
                else:
                    region = line.split(sep)[0]
                    # In the profiler table, levels are indicated with two spaces
                    lvl = int((len(region) - len(region.lstrip(' '))) / 2)
                    max_lvl = max(max_lvl, lvl)

    if skiprows is None or nrows is None or names is None:
        raise Exception("Unexpected file format, could not identify profiler table!")

    return skiprows,lines,nrows,names,max_lvl,runinfo


def load_profiler_dataframe(fname : str):
    """
    Loads profiler data from file 'fname'. The profiler regions are converted
    into a pandas MultiIndex to simplify selecting a particular profiler level
    or region.
    """
    skiprows,lines,nrows,names,max_lvl,runinfo = extract_performance_block(fname)

    lines.seek(0,io.SEEK_SET)
    df = pd.read_csv(lines, sep=sep, skiprows=skiprows, nrows=nrows, names=names)
    # Build pandas MultiIndex from profiler regions, where each MultiIndex level
    # corresponds to a profiler level. For top-level regions, the lower MultiIndex
    # levels will be filled with blank strings, e.g.
    #
    # multi_index = np.array(
    #     ['genex', '',           '',            ''],
    #     ['genex', 'initialize', '',            ''], # first child of 'genex' region
    #     ['genex', 'initialize', 'op_bnd_cond', ''],
    #     ...
    #     ['genex', 'timestep',   '',            ''], # second child of 'genex' region
    #     ['genex', 'timestep',   'copy',        ''],
    #     ...
    # )

    regions = np.full((len(df), max_lvl + 1), fill_value='', dtype=object)
    current_region = regions[0, :].copy()
    prev_lvl = 0
    for i, region in enumerate(df['region']):
        lvl = int((len(region) - len(region.lstrip(' '))) / 2)
        # If we've gone back to a higher level, fill the lower levels of the
        # index with blank strings
        if lvl < prev_lvl:
            current_region[lvl:] = ''
        current_region[lvl] = region.strip()
        regions[i, :] = current_region
        prev_lvl = lvl

    multi_index = pd.MultiIndex.from_arrays(regions.T,
        names=['lvl_{:01d}'.format(i) for i in range(max_lvl + 1)])
    df.set_index(multi_index, inplace=True)

    # Remove dataframe region column since the information is now stored in the
    # index
    df.drop('region', axis=1, inplace=True)

    return df


def load_profiler_tree(fname : str) -> dict:
    """
    Loads profiler data from file 'fname'. The profiler regions are converted
    into a pandas MultiIndex to simplify selecting a particular profiler level
    or region.
    """
    skiprows,lines,nrows,names,max_lvl,runinfo = extract_performance_block(fname)

    # Build a tree map of profiler regions from the buffered lines
    lines_data = lines.getvalue().splitlines()
    table_lines = lines_data[skiprows: skiprows + nrows]

    region_map = {}   # maps tuple path -> node dict
    roots = []
    stack = []

    for i, raw_line in enumerate(table_lines):
        parts = raw_line.split(sep)
        region_field = parts[0]
        lvl = int((len(region_field) - len(region_field.lstrip(' '))) / 2)
        name = region_field.strip()

        # Collect data fields using header names (skip the region column)
        data = {}
        for col_idx, colname in enumerate(names[1:], start=1):
            data[colname] = parts[col_idx].strip() if col_idx < len(parts) else ''

        node = {'name': name, 'self': data, 'children': []}

        # Ensure stack length
        if len(stack) <= lvl:
            stack.extend([None] * (lvl + 1 - len(stack)))

        # Attach into tree
        if lvl == 0:
            roots.append(node)
        else:
            parent = stack[lvl - 1]
            if parent is None:
                # Malformed indentation: treat as root
                roots.append(node)
            else:
                parent['children'].append(node)

        # Set current level in stack and truncate deeper levels
        stack[lvl] = node
        stack = stack[:lvl + 1]

    # Expose the constructed structures for later use
    profiler_tree = {'roots': roots,'runinfo':runinfo}

    return profiler_tree
