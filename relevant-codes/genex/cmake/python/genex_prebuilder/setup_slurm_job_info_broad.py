import time, datetime
from copy import deepcopy
from confix import get_partition_dataframe, autoselect_partition

def setup_slurm_job_info_broad(runtime_modes, cpu_job_info, gpu_job_info,
                               geometries):
    partition_df = get_partition_dataframe()
    temp_info = {}
    temp_info["cpu"] = {}
    modes = ["cpu"]

    if gpu_job_info:
        temp_info["gpu"] = {}
        modes.append("gpu")

    for geom in geometries:
        temp_info["cpu"][geom] = deepcopy(cpu_job_info[geom][1])
        if gpu_job_info:
            temp_info["gpu"][geom] = deepcopy(gpu_job_info[geom][1])

    # Fallback to only use CPU partition when it's only cpu and cxx modes
    cpu_fallback = False
    if set(["cpu", "cxx"]) == set(runtime_modes):
        cpu_fallback = True

    broad_job_info = {}
    for mode in modes:
        broad_job_info[mode] = {}
        total_tlim = 0
        n_procs = []
        for geom in geometries:
            info = temp_info[mode][geom]
            tlim = info["time_limit"]
            if isinstance(tlim, str):
                if tlim == "24:00:00":
                    tlim = 86400
                else:
                    tlim = time.strptime(tlim,'%H:%M:%S')
                    tlim = datetime.timedelta(hours=tlim.tm_hour,
                           minutes=tlim.tm_min,
                           seconds=tlim.tm_sec).total_seconds()
            total_tlim += tlim
            n_procs.append(info["n_nodes"] * info["ntasks_per_node"])

        if total_tlim >= 86400:
            total_tlim = "24:00:00"
        else:
            total_tlim = time.strftime('%H:%M:%S', time.gmtime(total_tlim))
        max_geom = geometries[n_procs.index(max(n_procs))]
        broad_job_info[mode] = deepcopy(temp_info[mode][max_geom])
        broad_job_info[mode]["time_limit"] = total_tlim
        if cpu_fallback:
            ptype = "cpu"
        else:
            ptype = broad_job_info[mode]["partition_type"]
        n_nodes = broad_job_info[mode]["n_nodes"]
        broad_job_info[mode]["partition_name"] = \
            autoselect_partition(partition_df, ptype, n_nodes, total_tlim)
    return broad_job_info
