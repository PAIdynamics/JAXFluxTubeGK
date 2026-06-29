import numpy as np
import h5py
from numba import jit

def load_RZ(fpath, mask_ghosts=True, mask_target=True,
            return_masks=False):
    """
    Loads the R and Z coordinates from the given mesh file path. The filler
    points are automatically masked; with mask_ghosts=True or mask_target=True,
    you can also mask ghost or target points. With return_masks=True, the
    filler, ghost, and target masks are also returned.
    """
    gridfile = h5py.File(fpath, 'r')
    RZ_grid = gridfile["RZ_grid"]
    not_filler = RZ_grid["not_filler"][()].astype(bool)
    not_ghost = RZ_grid["not_ghost"][()].astype(bool)
    not_in_target = gridfile["parcon"]["not_in_target"][()].astype(bool)
    # Load RZ grids, masking filler points and optionally also ghosts
    mask = ~not_filler
    if mask_ghosts:
        mask = np.logical_or(mask, ~not_ghost)
    if mask_target:
        mask = np.logical_or(mask, ~not_in_target)
    R    = np.ma.array(RZ_grid["R"][()], mask=mask)
    Z    = np.ma.array(RZ_grid["Z"][()], mask=mask)
    xind = np.ma.array(RZ_grid["x_ind"][()], mask=mask)
    yind = np.ma.array(RZ_grid["z_ind"][()], mask=mask)

    arrays_to_return = [R, Z, xind, yind]
    if return_masks:
        arrays_to_return += [not_filler, not_ghost, not_in_target]

    return arrays_to_return

def load_phi(fpath):
    """
    Loads the phi coordinates from the given mesh file path.
    """
    gridfile = h5py.File(fpath, 'r')
    return gridfile['phi_grid']['phi'][()] / np.pi

def load_velspace(fpath):
    """
    Loads the velspace coordinates vp, mu from the given mesh file path.
    """
    gridfile = h5py.File(fpath, 'r')
    return gridfile['vp_grid']['vp'][()], gridfile['mu_grid']['mu'][()]

def load_magfield(fpath):
    """
    Loads the magnetic field from the given mesh file path.
    """
    gridfile = h5py.File(fpath, 'r')
    return gridfile['magnetic_field']

def load_parcon(fpath):
    """
    Loads the parallel connection from the given mesh file path.
    """
    gridfile = h5py.File(fpath, 'r')
    return gridfile['parcon']

def load_bufzone(fpath):
    """
    Loads the buffer zone from the given mesh file path.
    """
    gridfile = h5py.File(fpath, 'r')
    return gridfile["RZ_grid"]['buf_zone'][()]

def get_mesh_extents(R: np.ma.array, Z: np.ma.array, flip_Z=False):
    # Returns minimum and maximum R and Z values of given 3D mesh. For use
    # with plt.imshow
    n_planes = R.shape[0]
    extents = [None] * n_planes
    for k in range(n_planes):
        if not R[k, :].mask.all():
            if flip_Z:
                extents[k] = [np.min(R[k, :].compressed()), \
                              np.max(R[k, :].compressed()), \
                              -np.max(Z[k, :].compressed()), \
                              -np.min(Z[k, :].compressed())]
            else:
                extents[k] = [np.min(R[k, :].compressed()), \
                              np.max(R[k, :].compressed()), \
                              np.min(Z[k, :].compressed()), \
                              np.max(Z[k, :].compressed())]
    return extents

def create_mesh_2d(x: np.ma.array, y: np.ma.array):
    # Creates a 2d cartesian mesh from the given unstructured 1d x and y
    # coordinates.
    x_vals = np.unique(x.compressed())
    y_vals = np.unique(y.compressed())
    x_2d, y_2d = np.meshgrid(x_vals, y_vals)

    return (x_2d, y_2d)

@jit(nopython=True, cache=True)
def get_nearest_uind(x, y, xval, yval, spacing):
    # Returns the unstructured index of the nearest grid point given by x, y
    # coordinates from user-defined xval and yval, assuming an equidistant grid.
    # If the point is outside the coordinate range, zero is returned.
    box_x = (xval - 0.5*spacing, xval + 0.5*spacing)
    box_y = (yval - 0.5*spacing, yval + 0.5*spacing)
    uind_inbox = np.where((x >= box_x[0]) & (x <= box_x[1]) &\
                          (y >= box_y[0]) & (y <= box_y[1]))[0]
    if len(uind_inbox) > 0:
        return uind_inbox[0]
    else:
        return 0

def create_field_2d(xind: np.ma.array, yind: np.ma.array, field: np.array):
    # Creates a two 2d field for the from the given unstructured 1d field data
    # and the 1d unstructured indices
    xind_vals, xind_idx = np.unique(xind.compressed(), return_inverse=True)
    yind_vals, yind_idx = np.unique(yind.compressed(), return_inverse=True)

    field_2d = np.full(yind_vals.shape + xind_vals.shape, fill_value=np.nan)
    field_2d[yind_idx, xind_idx] = field[~np.ma.getmask(xind)]

    mask_array = np.ma.array(field_2d, mask=np.isnan(field_2d))

    return mask_array

def get_n_points(fpath, params_file):
    # Returns an array with the number of grid points in each dimension of
    # the distribution function
    mesh_file = h5py.File(fpath, 'r')
    dim_RZ  = np.shape(mesh_file["RZ_grid"]["R"])[1]
    dim_phi = get_param(params_file, "n_points_phi")
    dim_vp  = get_param(params_file, "n_points_vp")
    dim_mu  = get_param(params_file, "n_points_mu")
    species = get_param(params_file, "names")
    return np.array([dim_RZ, dim_phi, dim_vp, dim_mu, np.size(species)], \
                    dtype=int)

def get_n_procs(params_file : str):
    # Returns an array with the number of total processes in each dimension
    n_phi = get_param(params_file, "n_procs_phi")
    n_vp  = get_param(params_file, "n_procs_vp")
    n_mu  = get_param(params_file, "n_procs_mu")
    n_sp  = get_param(params_file, "n_procs_sp")
    return np.array([1, n_phi, n_vp, n_mu, n_sp], dtype=int)

def get_param(fname, pname):
    # Function to read a param value from a Fortran input file containing
    # namelists. When duplicate parameters from multiple namelists exist
    # the first one is returned. When a specific namelist should be searched
    # use get_param_from_namelist instead.
    #
    # Returns a single string or an array of strings that must be casted
    # with meta knowledge about the parameter.
    # Only returns a parameter that has been set, i.e. is not a comment!

    # Read lines of file
    with open(fname) as f:
        lines = f.readlines()

    # Search for the parameter in read-in lines
    return search_param_in_lines(pname, lines)

def get_param_from_namelist(fname, nname, pname):
    # Function to read a param value (pname) from a
    # Fortran input file (fname) from a specific chosen namelist (nname).
    #
    # Returns a single string or an array of strings that must be casted
    # with meta knowledge about the parameter.
    # Only returns a parameter that has been set, i.e. is not a comment!

    # Read lines of file
    with open(fname) as f:
        lines = f.readlines()

    # Add starting symbol of namelist
    if nname[0] != "&":
        nname = "&" + nname

    # Search for the namelist in the parameter file - get start and end index
    # of the list of lines
    i_start = -1
    i_end   = -1
    for i in range(len(lines)):
        if(nname.lower() in lines[i].lower()):
            i_start = i
        if(i_start != -1 and "/" in lines[i].lower()):
            i_end = i
            break

    # Search for the parameter in read-in lines
    return search_param_in_lines(pname, lines[i_start:i_end])

def search_param_in_lines(pname, lines):
    # Function that searches for a specific parameter in a list of input file
    # lines.

    # Check if param substring is found, if yes return the parameter
    for l in lines:
        if(pname.lower() in l.lower()):

            # Parse fortran line
            ret = parse_fortran_line(l)

            # Return single element if array has 1 entry
            if(len(ret)) == 1:
                return ret[0]
            # Else return whole array
            else:
                return ret

def parse_fortran_line(expr):
    # Function that parses a fortran namelist line to a string array

    arr = []

    # First, check if line is a comment
    expr = expr.lstrip()
    if expr[0] == "!":
        return arr

    # Remove appending comments
    spl = expr.split('!')
    expr = spl[0]

    # For better parsing, add spaces around commas to split them apart
    expr = expr.replace(',', ' , ')

    # Split equal sign apart if they stick to a name or number
    # (created by gcc compiled code)
    expr = expr.replace('=', ' = ')
    # Remove quotation marks in strings (created by gcc compiled code)
    expr = expr.replace('"', ' ')

    # Split line by space
    spl = expr.split()

    # Try to parse each array entry of the splitted line
    # Start at index 2 because index 0 is the name and index 1 the = sign
    for l in spl[2:]:
        # If line is a float, add it to the array
        if(isfloat(l)):
            arr.append(l)
        # If fortran * notation, add the correct amount of the value
        elif('*' in l):
            n = l.split('*')
            for i in range(int(n[0])):
                arr.append(n[1])
        # If not only a comma left, add the string
        elif(l != ','):
            arr.append(l)

    # Remove empty array elements if any exist
    while '' in arr:
        arr.remove('')

    return arr

def isfloat(s):
    # Function that checks if a string is a float
    try:
        float(s)
        return True
    except ValueError:
        return False

def nml_to_bool(fstring):
    # Converts a namelist line to a boolean. If empty, returns [] by default.

    if not fstring:
        return []

    if ".true." in fstring or "T" in fstring:
        return True
    elif ".false." in fstring or "F" in fstring:
        return False
    else:
        return []

def nml_to_float(fstring):
    # Converts a namelist line to a float. If empty, returns np.nan by default.

    if not fstring:
        return np.nan

    if isfloat(fstring):
        return float(fstring)
    else:
        return np.nan
