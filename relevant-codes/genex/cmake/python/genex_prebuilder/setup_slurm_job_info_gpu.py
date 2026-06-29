import time, datetime
from copy import deepcopy

def setup_slurm_job_info_gpu(job_info, geometries, n_resolutions, runtime_modes):
    gpu_modes   = [mode for mode in runtime_modes if mode != "cpu"]
    resolutions = range(1, n_resolutions + 1)

    gpu_job_info = {}

    if not gpu_modes:
        return gpu_job_info

    for geom in geometries:
        gpu_job_info[geom] = {}
        for res in resolutions:
            gpu_job_info[geom][res] = {}
            total_tlim = 0
            n_procs = []
            for mode in gpu_modes:
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
            max_mode = gpu_modes[n_procs.index(max(n_procs))]
            gpu_job_info[geom][res] = deepcopy(job_info[max_mode][geom][res])
            gpu_job_info[geom][res]["time_limit"] = total_tlim
    return gpu_job_info
