#ifndef RUNTIME_TRACER_HXX
#define RUNTIME_TRACER_HXX

#include <string>
#include <mpi.h>

// Namespace for the runtime tracer based on the profiling regions
namespace runtime_tracer
{
    // Enumerator for runtime tracer mode
    enum class mode_t: int
    {
        START = 1,
        END = 0
    };

    // Initialize the file stream for runtime trace reports
    void initialize(const MPI_Comm comm, const std::string trace_dir,
                    const double initial_timestamp);

    // Finalize the file stream for runtime trace reports
    void finalize();

    // Trace the profiling region
    void trace(const mode_t mode, const std::string& profile_path,
               const double tstamp);
}

#ifdef __cplusplus
extern "C" {
#endif

void cbind_runtime_tracer_init(const int32_t comm, const char* out_dir,
                               const char* trace_folder,
                               const double init_tstamp);

void cbind_runtime_tracer_fin(const int32_t comm);

void cbind_runtime_tracer_trace(const int32_t mode, const char* region_path,
                                const char* region_name, const double tstamp);

#ifdef __cplusplus
}
#endif

#endif
