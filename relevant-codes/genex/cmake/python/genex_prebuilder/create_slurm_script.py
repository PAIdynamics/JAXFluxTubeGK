from os import getenv

def create_slurm_script(build_dir, job_config, job_info, runtime_modes, mode):
    job_name         = job_config["general"]["job_name"]
    output_file      = job_config["general"]["output_file"]
    partition_type   = job_info["partition_type"]
    partition_name   = job_info["partition_name"]
    qos              = job_info["qos"]
    n_nodes          = job_info["n_nodes"]
    ntasks_per_node  = job_info["ntasks_per_node"]
    time_limit       = job_info["time_limit"]
    cpus_per_task    = job_info["cpus_per_task"]
    slurm_gpu_gres   = job_info["slurm_gpu_gres"]
    memory           = job_info["memory"]
    is_exclusive     = job_info["is_exclusive"]
    slurm_constraint = job_info["slurm_constraint"]
    slurm_gpu_only   = job_info["slurm_gpu_only"]
    gpu_modes = [mode for mode in runtime_modes if mode != "cpu"]

    # Fallback to only use CPU partition when it's only cpu and cxx modes
    cpu_fallback = False
    if "gpu" not in partition_type or gpu_modes == ["cxx"] \
        or slurm_gpu_gres == "none":
        cpu_fallback = True

    script = []

    script = ['#!/bin/bash -l']
    script.append('# Job Name')
    script.append('#SBATCH -J ' + job_name)
    script.append('# Standard output and error')
    script.append('#SBATCH -o ' + output_file)

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
    if job_info["partition_as_constraint"]:
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

    if "environment_commands" in job_config[mode]:
        for line in job_config[mode]["environment_commands"]:
            script.append(line)
    script.append('')

    for line in job_config["general"]["execution_commands"]:
        if "gpu" in partition_type:
            sline = line.replace("<GPU_FLAG>", "-" + mode)
        else:
            sline = line.replace("<GPU_FLAG> ", "")
        sline = sline.replace("<OUT_DIR>", "./")
        sline = sline.replace("<PARAMS_IN_FILE>", "params_in.txt")
        script.append("rm -rf debug_log.txt mesh.nc part* *out.txt")
        script.append(sline)

    return script
