from copy import deepcopy
from confix import merge_namelist_dicts

def scale_parameters(config):
    mms_config    = config["mms_configurations"]
    common_params = config["common_parameters"]

    # Check if optional parameters are prescribed
    geom_params = {}
    res_params  = {}
    expl_params = {}
    if config["geometric_parameters"]:
        geom_params = config["geometric_parameters"]
    if config["resolution_parameters"]:
        res_params  = config["resolution_parameters"]
    if config["explicit_parameters"]:
        expl_params = config["explicit_parameters"]

    geometries     = mms_config["geometries"]
    scaling_factor = mms_config["scaling_factor"]
    upscale_vars   = mms_config["vars_to_upscale"]
    downscale_vars = mms_config["vars_to_downscale"]
    max_limits     = mms_config["max_limits"]
    min_limits     = mms_config["min_limits"]
    resolutions    = range(1, mms_config["n_resolutions"] + 1)

    scaled_params = {}

    for geom in geometries:
        scaled_params[geom] = {}
        for res in resolutions:
            # Handle the base resolution
            if res == 1:
                # Deep copy common parameters as the base resolution
                scaled_params[geom][1] = deepcopy(common_params)

                # Merge with geometric-specific parameters if prescribed
                if geom_params:
                    if geom_params[geom]:
                        scaled_params[geom][1] = \
                            merge_namelist_dicts(scaled_params[geom][1],
                                                 geom_params[geom])

            # Handle the scaling for each resolution
            else:
                # Deep copy parameters from the previous resolution
                scaled_params[geom][res] = deepcopy(scaled_params[geom][res-1])

                # Upscale the eligible parameters
                for namelist in upscale_vars:
                    for var in upscale_vars[namelist]:
                        svar = scaled_params[geom][res][namelist][var] \
                             * scaling_factor
                        if max_limits:
                            if var in max_limits and svar > max_limits[var]:
                                svar = max_limits[var]
                        scaled_params[geom][res][namelist][var] = svar

                # Downscale the eligible parameters
                for namelist in downscale_vars:
                    for var in downscale_vars[namelist]:
                        svar = scaled_params[geom][res][namelist][var] \
                             / scaling_factor
                        if min_limits:
                            if var in min_limits and svar < min_limits[var]:
                                svar = min_limits[var]
                        scaled_params[geom][res][namelist][var] = svar

            # Merge with resolution-specific parameters if prescribed
            if res_params:
                for namelist in res_params:
                    for var in res_params[namelist]:
                        scaled_params[geom][res][namelist][var] = \
                            res_params[namelist][var][res-1]

            # Merge with explicit parameters if prescribed
            if expl_params:
                for namelist in expl_params:
                    for var in expl_params[namelist]:
                        scaled_params[geom][res][namelist][var] = \
                            expl_params[namelist][var][geom][res-1]

    return scaled_params
