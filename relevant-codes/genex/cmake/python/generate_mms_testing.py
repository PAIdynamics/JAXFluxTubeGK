import argparse
from os import listdir, getenv
from copy import deepcopy
from confix import yml2dict, dict2namelist, list2script, \
                   merge_dicts, merge_namelist_dicts
from genex_prebuilder import decompose_procs, \
                             scale_parameters, \
                             create_mms_dir_tree, \
                             setup_slurm_job_info, \
                             create_slurm_script, \
                             setup_slurm_job_info_gpu, \
                             create_slurm_script_gpu, \
                             setup_slurm_job_info_broad, \
                             create_slurm_script_broad, \
                             setup_slurm_job_info_deep, \
                             create_slurm_script_deep, \
                             create_mms_assertion_script, \
                             create_mms_run_by_parts

parser = argparse.ArgumentParser()
parser.add_argument("source_dir", nargs=1, type=str,
                    help="Path to source directory")
parser.add_argument("build_dir", nargs=1, type=str,
                    help="Path to build directory")
parser.add_argument("build_type", nargs=1, type=str,
                    help="Type of the build")
parser.add_argument("runtime_modes", nargs='*', type=str,
                    help="Selection of runtime modes separated by semicolon")
args = parser.parse_args()

source_dir = args.source_dir[0].strip()
build_dir = args.build_dir[0].strip()
build_type = args.build_type[0].strip()
runtime_modes = args.runtime_modes

def generate_mms_testing(source_dir, build_dir, build_type, rtime_modes):
    mms_config_dir = source_dir + "/"
    config_files = [f for f in listdir(mms_config_dir) \
                      if "mms_configurations" in f]

    for f in config_files:
        # Read yml files and aggregate them for MMS configuration
        config_dict = yml2dict(mms_config_dir + f)
        for dep in config_dict["mms_configurations"]["file_dependencies"]:
            dep_dir = config_dict["mms_configurations"]["file_dependencies_dir"]
            temp_dict = yml2dict(mms_config_dir + dep_dir + "/" + dep)
            config_dict = merge_dicts(config_dict, temp_dict)

        # Overwrite the list of runtime modes based on current build support
        eff_rtime_modes = []
        for mode in rtime_modes:
            if mode in config_dict["job_configurations"]["general"] \
                                  ["runtime_modes"]:
                eff_rtime_modes.append(mode)
        config_dict["job_configurations"]["general"]["runtime_modes"] = \
            eff_rtime_modes

        mms_config    = config_dict["mms_configurations"]
        job_config    = config_dict["job_configurations"]
        common_params = config_dict["common_parameters"]

        destination_dir = build_dir + "/mms-testing/" \
                        + mms_config["subdir"] + "/"
        geometries      = mms_config["geometries"]
        n_resolutions   = mms_config["n_resolutions"]
        resolutions     = range(1, n_resolutions + 1)
        n_species       = len(common_params["params_species"]["charges"])
        runtime_modes   = job_config["general"]["runtime_modes"]

        # Create the directory tree under output path
        create_mms_dir_tree(mms_config, source_dir, destination_dir, geometries,
                            n_resolutions, runtime_modes)

        # Scale the common and explicit parameters for each resolutions
        scaled_params = scale_parameters(config_dict)

        # Slurm job info
        job_info = setup_slurm_job_info(job_config, geometries, n_resolutions)

        for mode in runtime_modes:
            # Fetch the GPU and other job related parameters
            gpu_params = {}
            for key in job_config[mode]:
                if "params_" in key:
                    gpu_params[key] = deepcopy(job_config[mode][key])

            for geom in geometries:
                for res in resolutions:
                    # Generate parallelization namelist
                    n_points = job_config[mode]["n_procs"][geom][res-1]
                    params_mesh = scaled_params[geom][res]["params_mesh"]
                    par_params = decompose_procs(n_points, params_mesh,
                                                 n_species=n_species)

                    # Merge scaled and parallel namelists
                    full_params = merge_namelist_dicts(scaled_params[geom][res],
                                                       par_params)

                    # Merge the GPU and other job related parameters
                    if gpu_params:
                        full_params = merge_namelist_dicts(full_params,
                                                           gpu_params)

                    # Set absolute path of the parameter file
                    params_file = destination_dir + geom + "/resolution_" + \
                                  str(res) + "/" + mode + "/params_in.txt"

                    # Write parameter file
                    dict2namelist(params_file, full_params)

                    # Set absolute path of the slurm job script file
                    script_file = destination_dir + geom + "/resolution_" + \
                                  str(res) + "/" + mode + "/submit.sh"

                    # Generate the contents of slurm job script file
                    slurm_script = create_slurm_script(build_dir, job_config,
                                    job_info[mode][geom][res], runtime_modes,
                                    mode)

                    # Write slurm job script file
                    list2script(script_file, slurm_script)

                    # Delete local dictionaries and lists
                    del par_params
                    del full_params
                    del slurm_script

        # Aggregate slurm job info for GPU
        gpu_job_info = setup_slurm_job_info_gpu(job_info, geometries,
                                                n_resolutions, runtime_modes)

        if gpu_job_info:
            for geom in geometries:
                for res in resolutions:
                    # Set absolute path of the slurm job script file
                    script_file = destination_dir + geom + "/resolution_" + \
                                  str(res) + "/submit_gpu.sh"

                    # Generate the contents of slurm job script file
                    slurm_script = create_slurm_script_gpu(build_dir,
                                    job_config, gpu_job_info[geom][res],
                                    runtime_modes, geom, res)

                    # Write slurm job script file
                    list2script(script_file, slurm_script)

                    # Delete local lists
                    del slurm_script

        # Aggregate slurm job info for broad MMS testing
        broad_job_info = setup_slurm_job_info_broad(runtime_modes,
                                                    job_info["cpu"],
                                                    gpu_job_info, geometries)

        # Aggregate slurm job info for deep MMS testing
        deep_job_info = setup_slurm_job_info_deep(job_info, runtime_modes,
                                                  geometries, resolutions)

        # Generate assertion scripts for broad MMS testing
        assert_script = create_mms_assertion_script(geometries, resolutions,
                                                    runtime_modes)

        for mode in broad_job_info:
            # Set absolute path of the slurm job script file
            script_file = destination_dir + "/submit_mms_broad_" + mode + ".sh"

            # Generate the contents of slurm job script file
            slurm_script = create_slurm_script_broad(build_dir, job_config, \
                           broad_job_info[mode], geometries, runtime_modes)

            # Write slurm job script file
            list2script(script_file, slurm_script)

            # Set absolute path of the assertion script file
            script_file = destination_dir + "/mms_assert_broad_" + mode + ".sh"

            # Write assertion script file
            list2script(script_file, assert_script["broad"][mode])

            # Delete local lists
            del slurm_script

        for geom in geometries:
            for mode in runtime_modes:
                # Set absolute path of the slurm job script file
                script_file = destination_dir + geom + "/submit_mms_deep_" + \
                              mode + ".sh"

                # Generate the contents of slurm job script file
                slurm_script = create_slurm_script_deep(build_dir, job_config, \
                               deep_job_info[mode][geom], resolutions, \
                               runtime_modes, mode, geom)

                # Write slurm job script file
                list2script(script_file, slurm_script)

                # Delete local lists
                del slurm_script

            # Set absolute path of the assertion script file
            script_file = destination_dir + geom + "/mms_assert_deep.sh"

            # Write assertion script file
            list2script(script_file, assert_script["deep"][geom])

        # This part is a special MMS broad GPU setup in MPCDF Raven that submits
        # and asserts the job in parts in order to increase the CI throughput
        pname_by_parts = []
        tlim_by_parts  = []

        if (getenv('MACH_NAME', None) == 'MPCDF Raven'):
            # Job distribution in case of Release build
            if (build_type.lower() == 'release'):
                parts = {0: ['cxx/slab', 'cxx/circular', 'cxx/salpha',
                             'acc/slab', 'acc/circular', 'acc/salpha',
                             'ompx/slab', 'ompx/circular', 'ompx/salpha'],
                         1: ['cxx/dommaschk', 'acc/dommaschk', 'ompx/dommaschk']}

            # Job distribution in case of Debug build and other build types
            else:
                parts = {0: ['cxx/slab', 'acc/slab', 'ompx/slab'],
                         1: ['cxx/circular', 'cxx/salpha'],
                         2: ['acc/circular'],
                         3: ['ompx/circular'],
                         4: ['acc/salpha'],
                         5: ['ompx/salpha'],
                         6: ['cxx/dommaschk'],
                         7: ['acc/dommaschk'],
                         8: ['ompx/dommaschk']}

            # Manually reset the SLURM partition and time limit
            for i in parts:
                pname_by_parts.append("gpu")
                tlim_by_parts.append("00:30:00")

        elif (getenv('MACH_NAME', None) == 'MPCDF Viper-GPU'):
            # Job distribution in case of Release build
            if (build_type.lower() == 'release'):
                parts = {0: ['cxx/slab', 'cxx/circular',
                             'ompx/slab', 'ompx/circular'],
                         1: ['cxx/salpha', 'ompx/salpha'],
                         2: ['cxx/dommaschk', 'ompx/dommaschk']}

            # Job distribution in case of Debug build and other build types
            else:
                parts = {0: ['cxx/slab', 'ompx/slab'],
                         1: ['cxx/circular'],
                         2: ['ompx/circular'],
                         3: ['cxx/salpha'],
                         4: ['ompx/salpha'],
                         5: ['cxx/dommaschk'],
                         6: ['ompx/dommaschk']}

            # Manually reset the SLURM partition and time limit
            for i in parts:
                pname_by_parts.append("apu")
                tlim_by_parts.append("00:30:00")

        gpu_modes = [mode for mode in runtime_modes if mode != 'cpu']

        # Check if the running machine is MPCDF Raven and only filter GPU jobs
        if (getenv('MACH_NAME', None) == 'MPCDF Raven' or
            getenv('MACH_NAME', None) == 'MPCDF Viper-GPU') and \
            'gpu' in broad_job_info and gpu_modes != ['cxx']:

            for key in parts:
                suffix = str(key)
                modes, geoms = zip(*(s.split("/") for s in parts[key]))
                modes = list(dict.fromkeys(modes))
                geoms = list(dict.fromkeys(geoms))

                # Set absolute path of the slurm job script file
                script_file = destination_dir + "/submit_mms_broad_gpu_" + \
                              suffix + ".sh"

                # Aggregate slurm job info for broad MMS testing by parts
                bpart_job_info = setup_slurm_job_info_broad(modes, \
                                 job_info["cpu"], gpu_job_info, geoms)

                # Override some values
                bpart_job_info["gpu"]["partition_type"] = "gpu"
                bpart_job_info["gpu"]["partition_name"] = pname_by_parts[key]
                bpart_job_info["gpu"]["time_limit"]     = tlim_by_parts[key]

                # Generate the contents of slurm job script file
                slurm_script = create_slurm_script_broad(build_dir, \
                               job_config, bpart_job_info["gpu"], \
                               geometries=geoms, runtime_modes=modes, \
                               is_mms_by_parts=True, outfile_suffix=suffix)

                # Write slurm job script file
                list2script(script_file, slurm_script)

                # Delete local dicts and lists
                del bpart_job_info
                del slurm_script

                # Generate assertion scripts for broad MMS testing
                assert_script = \
                    create_mms_assertion_script(geoms, resolutions, modes, \
                    is_mms_by_parts=True, outfile_suffix=suffix)

                # Set absolute path of the assertion script file
                assert_file = destination_dir + "/mms_assert_broad_gpu_" + \
                              suffix + ".sh"

                # Write assertion script file
                list2script(assert_file, assert_script)

            # Generate the main shell script to run the whole MMS broad GPU by parts
            script = create_mms_run_by_parts(pname_by_parts)

            # Set absolute path of the script file
            script_file = destination_dir + "/run_mms_broad_gpu_by_parts.sh"

            # Write script file
            list2script(script_file, script)

    return None

generate_mms_testing(source_dir, build_dir, build_type, runtime_modes)
