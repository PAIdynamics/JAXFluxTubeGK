import numpy as np
import re
import xarray as xr
import h5py
from pathlib import Path
from glob import glob

def get_groups(path_to_file: Path):
    # Returns a list of all groups in the h5 file
    f = h5py.File(str(path_to_file.absolute()), 'r')
    keys = []
    f.visit(lambda key : keys.append(key) \
            if isinstance(f[key], h5py.Group) else None)
    f.close()
    return keys

def findall_files(directory_path: Path, filename: str):
    # Returns a list of all occurences of "filename" in "directory_path" and
    # sub-paths. Type of the list elements is Path.
    #res = directory_path.rglob(filename)
    res = glob(str(directory_path.resolve()) + "/**/" + filename, \
               recursive=True)
    return [Path(elem) for elem in res]

def has_parts(directory_path: Path):
    # Checks whether the directory has a structure that conforms with the
    # GENEX part structure.
    test = findall_files(directory_path, 'part_*')
    if len(test) > 0:
        return True
    else:
        return False

def load_file_into_dataset(path_to_file: Path, group=None, is_checkpoint=False):
    # Loads the specified file into an xarray dataset with all dimensions
    # except RZ as coordinates
    try:
        # Internal errors by dask may not be raised, leading to a frozen
        # behaviour. Therefore we force it and set it back to default
        # afterwards
        err_default = np.geterr()
        np.seterr(divide="raise")
        dataset = xr.open_dataset(path_to_file.absolute(), chunks='auto',
                                  decode_cf=False)
    finally:
        np.seterr(divide=err_default["divide"])

    # Replace dim_time by time since the former is only an index and
    # combination would not work
    dataset = dataset.rename({'time':'dim_time'})
    # If group is chosen we have to make correction to the time dimension
    # because in the group it is an index and we need the value
    if(not group is None):
        timedata = dataset["dim_time"]
        dataset = xr.open_dataset(path_to_file.absolute(), group=group,
                                  chunks='auto', decode_cf=False)
        dataset.coords["dim_time"] = timedata
    # We get all properties of the dataset and go through the list. When we
    # find a dimension property we add it as a coordinate
    properties = dir(dataset)
    regex = re.compile('dim_')
    for prop in properties:
        if(regex.match(prop)):
            if (not is_checkpoint):
                # RZ is not a coordinate per default
                if('RZ' in prop): continue
                propdata = dataset[prop]
            else:
                # Checkpoints need to be handled differently because they are
                # split between mpi processes
                name = path_to_file.stem
                number = int(name.split('_')[-1])

                # Read boundaries without ghost cells from checkpoint_files
                lb = dataset.data_vars["lb_stripped"].values
                ub = dataset.data_vars["ub_stripped"].values + 1
                # +1 is due to how Python range works w.r.t. Fortran
                # i.e. np.arange(3) results in [1,2] not [1,2,3]

                # We need to manually assign the correct dimension index of
                # lb, ub to the dimension name in the nc file
                if('RZ' in prop):
                    propdata = np.arange(lb[0], ub[0], dtype = int)
                elif('phi' in prop):
                    propdata = np.arange(lb[1], ub[1], dtype = int)
                elif('vp' in prop):
                    propdata = np.arange(lb[2], ub[2], dtype = int)
                elif('mu' in prop):
                    propdata = np.arange(lb[3], ub[3], dtype = int)
                elif('sp' in prop):
                    propdata = np.arange(lb[4], ub[4], dtype = int)
                else:
                    propdata = dataset[prop]

            dataset = dataset.assign_coords({prop : propdata})
    return dataset

def load_from_parts(dir_path: str, fname: str, group=None):
    # Returns an xarray dataset that contains all the data from the desired
    # file "fname" in all part_* subdirectories of "dir_path".
    # If file contains netcdf group structure, a call without the group
    # parameter will only return datasets that are common to all groups.
    # Does not work for checkpoint files. See load_checkpoints_from_parts.
    if(has_parts(Path(dir_path))):
        path_to_files = findall_files(Path(dir_path), fname)
        data = []
        for p in path_to_files:
            try:
                data.append(load_file_into_dataset(p, group))
            except Exception:
                print("Warning: Failed loading data from " + str(p))
        dataset = xr.combine_by_coords(data)
    else:
        raise Exception("File has no part structure!")

    return dataset

def load_checkpoints_from_parts(dir_path: str, fname: str, group=None):
    # Returns an xarray dataset that contains all the data from the desired
    # checkpoint file "fname" in all part_* subdirectories of "dir_path".
    # Needs the total number of points and total number of procs
    if(has_parts(Path(dir_path))):
        path_to_files = findall_files(Path(dir_path), fname)
        data = []
        for p in path_to_files:
            try:
                data.append(load_file_into_dataset(p, group, \
                                                   is_checkpoint=True))
            except Exception:
                print("Warning: Failed loading data from " + str(p))
        dataset = xr.combine_by_coords(data)
    else:
        raise Exception("File has no part structure!")

    return dataset

def files_exist(dir_path: str, pattern: str):
    # Checks whether the directory has a structure that conforms with the
    # GENEX part structure and contains files matching the given pattern.
    directory_path = Path(dir_path)
    if not has_parts(directory_path):
        return False
    test = findall_files(directory_path, pattern)
    if len(test) > 0:
        return True
    else:
        return False
