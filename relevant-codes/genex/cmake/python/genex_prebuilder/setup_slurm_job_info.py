import time
from confix import get_machine_info, get_partition_dataframe, \
                   autoselect_partition

def setup_slurm_job_info(job_config, geometries, n_resolutions):
    general_config = job_config["general"]
    rt_modes = general_config["runtime_modes"]
    env_comm = general_config["environment_commands"]
    exe_comm = general_config["execution_commands"]
    resolutions = range(1, n_resolutions + 1)

    machine_info     = get_machine_info()
    partition_df     = get_partition_dataframe()
    cpus_per_node    = machine_info["cpus_per_node"]
    gpus_per_node    = machine_info["gpus_per_node"]
    numa_per_node    = machine_info["numa_per_node"]
    slurm_gpu_gres   = machine_info["slurm_gpu_gres"]
    slurm_constraint = None
    if "slurm_constraint" in machine_info:
        slurm_constraint = machine_info["slurm_constraint"]
    slurm_gpu_only   = False
    if "slurm_gpu_only" in machine_info:
        slurm_gpu_only = machine_info["slurm_gpu_only"]

    # Support for machine that require setting the partition as a constraint
    partition_as_constraint = None
    if "partition_as_constraint" in machine_info:
        partition_as_constraint \
            = machine_info["partition_as_constraint"]

    # Fallback to only use CPU partition when it's only cpu and cxx modes
    cpu_fallback = False
    if set(["cpu", "cxx"]) == set(rt_modes):
        cpu_fallback = True
        buffer_ptype = job_config["cxx"]["partition_type"]
        job_config["cxx"]["partition_type"] = "cpu"
        # if machine_info["gpu_only"]:
        #     buffer_ptype = "gpu"

    # GENE-X specific restriction
    gpus_per_task = 1

    job_info = {}

    for mode in rt_modes:
        job_info[mode] = {}
        all_n_procs    = job_config[mode]["n_procs"]
        all_tlimits    = job_config[mode]["time_limits"]
        ptype          = job_config[mode]["partition_type"]

        # if machine_info["gpu_only"]:
        #     ptype = "gpu"

        for geom in geometries:
            job_info[mode][geom] = {}

            for res in resolutions:
                job_info[mode][geom][res] = {}
                n_procs = all_n_procs[geom][res-1]
                tlim    = all_tlimits[geom][res-1]

                if "gpu" in ptype or slurm_gpu_only:
                    ntasks_per_node = int(gpus_per_node / gpus_per_task)
                else:
                    ntasks_per_node = numa_per_node

                if n_procs >= ntasks_per_node:
                    n_nodes = int(n_procs / ntasks_per_node)
                else:
                    n_nodes = 1
                    ntasks_per_node = n_procs

                if "gpu" in ptype or slurm_gpu_only:
                    cpus_per_task = int(cpus_per_node * gpus_per_task / \
                                        gpus_per_node)
                else:
                    cpus_per_task = int(cpus_per_node / ntasks_per_node)

                is_exclusive = False
                if cpus_per_task * ntasks_per_node == cpus_per_node:
                    is_exclusive = True

                pname = autoselect_partition(partition_df, ptype, n_nodes, tlim)
                for p in machine_info["partitions"]:
                    if machine_info["partitions"][p]["partition_name"] == pname:
                        partition = machine_info["partitions"][p]
                        break

                if isinstance(tlim, int):
                    if (tlim % 86400) == 0:
                        tlim = '24:00:00'
                    else:
                        tlim = time.strftime('%H:%M:%S', time.gmtime(tlim))

                if "mem_per_node" in partition:
                    mem_per_node = partition["mem_per_node"]
                else:
                    mem_per_node = machine_info["mem_per_node"]

                memory = int(mem_per_node * ntasks_per_node * \
                             cpus_per_task / cpus_per_node)

                try:
                    qos = partition["qos"]
                except:
                    qos = None

                job_info[mode][geom][res]["partition_type"]   = ptype
                job_info[mode][geom][res]["partition_name"]   = pname
                job_info[mode][geom][res]["qos"]              = qos
                job_info[mode][geom][res]["n_nodes"]          = n_nodes
                job_info[mode][geom][res]["ntasks_per_node"]  = ntasks_per_node
                job_info[mode][geom][res]["time_limit"]       = tlim
                job_info[mode][geom][res]["cpus_per_task"]    = cpus_per_task
                job_info[mode][geom][res]["slurm_gpu_gres"]   = slurm_gpu_gres
                job_info[mode][geom][res]["memory"]           = memory
                job_info[mode][geom][res]["is_exclusive"]     = is_exclusive
                job_info[mode][geom][res]["slurm_constraint"] = slurm_constraint
                job_info[mode][geom][res]["slurm_gpu_only"]   = slurm_gpu_only
                job_info[mode][geom][res]["partition_as_constraint"] \
                    = partition_as_constraint

                if cpu_fallback and mode == "cxx":
                    job_info["cxx"][geom][res]["partition_type"] = buffer_ptype

    if cpu_fallback:
        job_config["cxx"]["partition_type"] = buffer_ptype

    return job_info
