import time, datetime
from copy import deepcopy
from confix import get_partition_dataframe, autoselect_partition

def setup_slurm_job_info_deep(job_info, runtime_modes, geometries, resolutions):
    partition_df  = get_partition_dataframe()
    deep_job_info = {}

    # Fallback to only use CPU partition when it's only cpu and cxx modes
    cpu_fallback = False
    if set(["cpu", "cxx"]) == set(runtime_modes):
        cpu_fallback = True

    for mode in runtime_modes:
        deep_job_info[mode] = {}
        for geom in geometries:
            deep_job_info[mode][geom] = {}
            total_tlim = 0
            n_procs = []
            for res in resolutions:
                info = job_info[mode][geom][res]
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
            max_res = resolutions[n_procs.index(max(n_procs))]
            deep_job_info[mode][geom] = deepcopy(job_info[mode][geom][max_res])
            deep_job_info[mode][geom]["time_limit"] = total_tlim
            if cpu_fallback and mode == "cxx":
                ptype = "cpu"
            else:
                ptype   = deep_job_info[mode][geom]["partition_type"]
            n_nodes = deep_job_info[mode][geom]["n_nodes"]
            deep_job_info[mode][geom]["partition_name"] = \
                autoselect_partition(partition_df, ptype, n_nodes, total_tlim)
    return deep_job_info
