from os import getenv

def create_slurm_script_deep(build_dir, job_config, deep_job_info, \
                             resolutions, runtime_modes, mode, geom):
    job_name         = job_config["general"]["job_name"]
    output_file      = job_config["general"]["output_file"]
    partition_type   = deep_job_info["partition_type"]
    partition_name   = deep_job_info["partition_name"]
    qos              = deep_job_info["qos"]
    n_nodes          = deep_job_info["n_nodes"]
    ntasks_per_node  = deep_job_info["ntasks_per_node"]
    time_limit       = deep_job_info["time_limit"]
    cpus_per_task    = deep_job_info["cpus_per_task"]
    slurm_gpu_gres   = deep_job_info["slurm_gpu_gres"]
    memory           = deep_job_info["memory"]
    is_exclusive     = deep_job_info["is_exclusive"]
    slurm_constraint = deep_job_info["slurm_constraint"]
    slurm_gpu_only   = deep_job_info["slurm_gpu_only"]
    mem_per_cpu      = int(memory / (cpus_per_task * ntasks_per_node))
    gpu_modes = [mode for mode in runtime_modes if mode != "cpu"]

    # Fallback to only use CPU partition
    cpu_fallback = False
    if "gpu" not in partition_type or gpu_modes == ["cxx"] \
        or slurm_gpu_gres == "none":
        cpu_fallback = True

    script = []

    script = ['#!/bin/bash -l']
    script.append('# Job Name')
    script.append('#SBATCH -J ' + job_name)
    script.append('# Standard output and error')
    script.append('#SBATCH -o mms_deep_' + mode + '.out')

    if slurm_gpu_only:
        script.append('#SBATCH --gres=' + slurm_gpu_gres + str(ntasks_per_node))
        if slurm_constraint:
            script.append('#SBATCH --constraint=' + slurm_constraint)
        else:
            script.append('#SBATCH --constraint=gpu')

    if is_exclusive:
        script.append('#SBATCH --exclusive')

    billing_account = getenv('GXBILL', None)
    if billing_account:
        script.append('#SBATCH -A ' + billing_account)

    if not cpu_fallback:
        script.append('#SBATCH --constraint=gpu')

    # Quick fix for machines that require setting partition as constraint.
    # In this case, no partition should be written as this would lead to
    # invalid batch file.
    if deep_job_info["partition_as_constraint"]:
        script.append('#SBATCH --constraint=' + partition_name)
    else:
        script.append('#SBATCH --partition=' + partition_name)

    if qos:
        script.append('#SBATCH --qos=' + qos)

    script.append('# Number of nodes')
    script.append('#SBATCH --nodes=' + str(n_nodes))
    script.append('#SBATCH --ntasks-per-node=' + str(ntasks_per_node))

    if not cpu_fallback:
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
        if "environment_commands" in job_config[mode]:
            for line in job_config[mode]["environment_commands"]:
                script.append(line)
    script.append('')

    for res in resolutions:
        exec_dir = "./resolution_" + str(res) + "/"
        for line in job_config["general"]["execution_commands"]:
            sline = line.replace("srun", "srun --mem-per-cpu=" + \
                                         str(mem_per_cpu))
            n_procs = job_config[mode]["n_procs"][geom][res - 1]
            n_nodes = n_procs // ntasks_per_node if n_procs > 1 else 1
            sline = sline.replace("srun", "srun -N {} -n {}".format(n_nodes, n_procs))
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

    return script
