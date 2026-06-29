from os import getenv

def create_slurm_script_broad(build_dir, job_config, broad_job_info, \
                              geometries, runtime_modes, \
                              is_mms_by_parts=False, outfile_suffix=None):
    job_name         = job_config["general"]["job_name"]
    output_file      = job_config["general"]["output_file"]
    partition_type   = broad_job_info["partition_type"]
    partition_name   = broad_job_info["partition_name"]
    qos              = broad_job_info["qos"]
    n_nodes          = broad_job_info["n_nodes"]
    ntasks_per_node  = broad_job_info["ntasks_per_node"]
    time_limit       = broad_job_info["time_limit"]
    cpus_per_task    = broad_job_info["cpus_per_task"]
    slurm_gpu_gres   = broad_job_info["slurm_gpu_gres"]
    memory           = broad_job_info["memory"]
    is_exclusive     = broad_job_info["is_exclusive"]
    slurm_constraint = broad_job_info["slurm_constraint"]
    slurm_gpu_only   = broad_job_info["slurm_gpu_only"]
    mem_per_cpu      = int(memory / (cpus_per_task * ntasks_per_node))
    modes = {}
    modes["cpu"] = ["cpu"]
    modes["gpu"] = [mode for mode in runtime_modes if mode != "cpu"]

    # Fallback to only use CPU partition when it's only cpu and cxx modes
    cpu_fallback = False
    if "gpu" not in partition_type or modes["gpu"] == ["cxx"] \
        or slurm_gpu_gres == "none":
        cpu_fallback = True

    # Avoid CPU fallback fallback in case of MMS testing by parts in MPCDF Raven
    if is_mms_by_parts:
        cpu_fallback = False

    script = []

    script = ['#!/bin/bash -l']
    script.append('# Job Name')
    script.append('#SBATCH -J ' + job_name)
    script.append('# Standard output and error')
    if is_mms_by_parts and outfile_suffix:
        script.append('#SBATCH -o mms_broad_' + partition_type + '_' + \
                      outfile_suffix + '.out')
    else:
        script.append('#SBATCH -o mms_broad_' + partition_type + '.out')

    if is_exclusive:
        script.append('#SBATCH --exclusive')

    billing_account = getenv('GXBILL', None)
    if billing_account:
        script.append('#SBATCH -A ' + billing_account)

    if not cpu_fallback or slurm_gpu_only:
        if slurm_constraint:
            script.append('#SBATCH --constraint=' + slurm_constraint)
        else:
            script.append('#SBATCH --constraint=gpu')

    # Quick fix for machines that require setting partition as constraint.
    # In this case, no partition should be written as this would lead to
    # invalid batch file.
    if broad_job_info["partition_as_constraint"]:
        script.append('#SBATCH --constraint=' + partition_name)
    else:
        script.append('#SBATCH --partition=' + partition_name)

    if qos:
        script.append('#SBATCH --qos=' + qos)

    script.append('# Number of nodes')
    script.append('#SBATCH --nodes=' + str(n_nodes))
    script.append('#SBATCH --ntasks-per-node=' + str(ntasks_per_node))

    if not cpu_fallback or slurm_gpu_only:
        script.append('# Number of gpus per node')
        script.append('#SBATCH --gres=' + slurm_gpu_gres + str(ntasks_per_node))
    script.append('#SBATCH --mem=' + str(memory))

    script.append('# Number of openmp threads')
    script.append('#SBATCH --cpus-per-task=' + str(cpus_per_task))
    script.append('#SBATCH --time=' + time_limit)

    script.append('##SBATCH --mail-type=ALL          ' +
                  '# Uncomment, enter your email and remove')
    script.append('##SBATCH --mail-user=<YOUR_EMAIL> ' +
                  '# these comments to receive job notifications')
    script.append('')

    for line in job_config["general"]["environment_commands"]:
            if "<GENEX_BUILD_DIR>" in line:
                sline = line.replace("<GENEX_BUILD_DIR>", build_dir)
                script.append(sline)
            else:
                script.append(line)

    if not cpu_fallback:
        for mode in modes["gpu"]:
            if "environment_commands" in job_config[mode]:
                for line in job_config[mode]["environment_commands"]:
                    script.append(line)
    script.append('')

    for geom in geometries:
        exec_dir = "./" + geom + "/resolution_1/"
        for line in job_config["general"]["execution_commands"]:
            for mode in modes[partition_type]:
                sline = line.replace("srun", "srun --mem-per-cpu=" + \
                                             str(mem_per_cpu))
                n_procs = job_config[mode]["n_procs"][geom][0]
                if n_procs == 1:
                    sline = sline.replace("srun", "srun -N 1 -n 1")
                else:
                    sline = sline.replace("srun", "srun -n " + str(n_procs))
                if mode == "cpu":
                    sline = sline.replace("<GPU_FLAG> ", "")
                else:
                    sline = sline.replace("<GPU_FLAG>", "-" + mode)
                sline = sline.replace("<OUT_DIR>", exec_dir + mode + "/")
                sline = sline.replace("<PARAMS_IN_FILE>",
                                      exec_dir + mode + "/params_in.txt")
                sline += " 2>&1 |& tee " + exec_dir + mode + "/" + output_file

                script.append("( cd " + exec_dir + mode + "; " \
                              "rm -rf debug_log.txt mesh.nc part* *out.txt )")
                script.append(sline)
                script.append("sleep 1")
            script.append('')

    if is_mms_by_parts:
        script.append('touch mms_broad_' + partition_type + '_' + \
                      outfile_suffix + '.done')

    return script
